"""Compare embedding models end to end on the tuning set, on the experiment branch only.

For each model: build its own index, choose a re-tuned `min_similarity` from the tuning set, then
run the whole pipeline (retrieval, refusal threshold, and gemma4:e4b answering) at 0.6 and at the
re-tuned value. Settings go to each step as FINE_PRINT_* environment variables, in a fresh
process, because Fine Print caches its settings and its embedding model per process.

The held-out set is never used here: Fine Print's eval/holdout is spent.

Documents are read in place from the main checkout's data/raw (--raw-dir), which this worktree
doesn't have because it is gitignored.

Usage: uv run python scripts/compare_embedding_models.py [--fine-tuned PATH] [--raw-dir PATH]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "runs" / "embedding_models"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_THRESHOLD = 0.6
STRATEGY = "section"


def models(fine_tuned: Path) -> dict[str, str]:
    """The models compared, by short name. All are BGE models, so all take the BGE prefix."""
    return {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        # Relative to this repo, which every step runs from, so results hold no machine paths.
        "bge-small-fine-tuned": os.path.relpath(fine_tuned, ROOT),
    }


def choose_threshold(answerable: list[float], unanswerable: list[float]) -> float:
    """Pick the threshold that handles the most tuning questions correctly.

    A question is kept when its best similarity is at least the threshold. Answerable questions
    should be kept and unanswerable ones refused. Every threshold strictly between two adjacent
    scores behaves the same, so each gap is tried at its midpoint; among the best gaps, the
    widest wins, which leaves the most room on both sides. This rule was fixed before any
    similarity was looked at.
    """
    scores = sorted(set(answerable + unanswerable))
    gaps = list(itertools.pairwise(scores))

    def correct(threshold: float) -> int:
        kept = sum(score >= threshold for score in answerable)
        refused = sum(score < threshold for score in unanswerable)
        return kept + refused

    best = max(correct((low + high) / 2) for low, high in gaps)
    low, high = max(
        ((low, high) for low, high in gaps if correct((low + high) / 2) == best),
        key=lambda gap: gap[1] - gap[0],
    )
    return round((low + high) / 2, 3)


def run_step(
    step: str, model: str, threshold: float, chroma: Path, raw_dir: Path, out: Path
) -> None:
    """Run one step in a fresh process with this configuration's settings."""
    env = os.environ | {
        "FINE_PRINT_RAW_DIR": str(raw_dir),
        "FINE_PRINT_EMBEDDING_MODEL": model,
        "FINE_PRINT_QUERY_PREFIX": BGE_PREFIX,
        "FINE_PRINT_MIN_SIMILARITY": str(threshold),
        "FINE_PRINT_CHROMA_DIR": str(chroma),
        "FINE_PRINT_DEFAULT_STRATEGY": STRATEGY,
    }
    if step == "index":
        cli = Path(sys.executable).parent / "fine-print"
        command = [str(cli), "index", "--strategy", STRATEGY]
    else:
        command = [sys.executable, __file__, "--child", step, "--out", str(out)]
    subprocess.run(command, env=env, check=True, cwd=ROOT)


def child(step: str, out: Path) -> None:
    """Measure, in this process, with the settings from the environment."""
    from fine_print import evaluate as ev  # noqa: PLC0415
    from fine_print.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    questions = ev.load_jsonl(settings.questions_path, ev.EvalQuestion)
    off_topic = ev.load_jsonl(settings.out_of_scope_path, ev.OutOfScopeQuestion)
    if step == "similarities":
        result = {
            "answerable": [ev._best_similarity(q.question, STRATEGY) for q in questions],
            "unanswerable": [ev._best_similarity(q.question, STRATEGY) for q in off_topic],
        }
    else:
        result = {
            "embedding_model": settings.embedding_model,
            "min_similarity": settings.min_similarity,
            "retrieval": ev.evaluate_strategy(STRATEGY, questions).model_dump(),
            "threshold": ev.evaluate_threshold(STRATEGY, questions, off_topic).model_dump(),
            "answers": ev.evaluate_answers(STRATEGY, questions, off_topic).model_dump(),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    partial.replace(out)  # Whole files only, so a resumed run can trust what exists.


def main() -> None:
    """Run every model at both thresholds."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tuned", type=Path, default=ROOT.parent / "embedding-model-bench" / "models" / "all"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT.parent / "fine-print" / "data" / "raw")
    parser.add_argument("--child")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.out)
        return
    if not (args.fine_tuned / "model.safetensors").exists():
        sys.exit(f"No fine-tuned model at {args.fine_tuned}")
    for name, model in models(args.fine_tuned).items():
        chroma = ROOT / "chroma_experiment" / name
        similarities = OUT / f"{name}__similarities.json"
        # Finished steps are skipped, so an interrupted run can be resumed.
        if not similarities.exists():
            print(f"== {name}: indexing", flush=True)
            run_step("index", model, DEFAULT_THRESHOLD, chroma, args.raw_dir, OUT)
            run_step("similarities", model, DEFAULT_THRESHOLD, chroma, args.raw_dir, similarities)
        scores = json.loads(similarities.read_text(encoding="utf-8"))
        tuned = choose_threshold(scores["answerable"], scores["unanswerable"])
        print(f"== {name}: re-tuned min_similarity {tuned}", flush=True)
        for threshold in dict.fromkeys([DEFAULT_THRESHOLD, tuned]):
            print(f"== {name}: answering at {threshold}", flush=True)
            out = OUT / f"{name}__{threshold}.json"
            if not out.exists():
                run_step("answers", model, threshold, chroma, args.raw_dir, out)


if __name__ == "__main__":
    main()
