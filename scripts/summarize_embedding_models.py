"""Summarize scripts/compare_embedding_models.py's results as Markdown, on the experiment branch.

Usage: uv run python scripts/summarize_embedding_models.py > docs/embedding_experiment_results.md
"""

import json
from math import comb
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "eval" / "runs" / "embedding_models"
MODELS = ("bge-small", "bge-base", "bge-small-fine-tuned")
BASELINE = "bge-small"
DEFAULT = 0.6


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test, ignoring ties."""
    total = wins + losses
    if total == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(total, i) for i in range(min(wins, losses) + 1)) / 2**total)


def load(name: str, threshold: float) -> dict:
    return json.loads((RUNS / f"{name}__{threshold}.json").read_text(encoding="utf-8"))


def tuned_threshold(name: str) -> float:
    """The re-tuned value is the one run file whose threshold isn't the default."""
    values = [float(p.stem.split("__")[1]) for p in RUNS.glob(f"{name}__0*.json")]
    others = [v for v in values if v != DEFAULT]
    return others[0] if others else DEFAULT


def correct_flags(run: dict) -> dict[str, bool]:
    """Whether each answerable question got a correct citation."""
    return {
        r["question"]: bool(r["correct_citation"])
        for r in run["answers"]["records"]
        if r["answerable"]
    }


def row(name: str, threshold: float, label: str, baseline_run: dict) -> str:
    run = load(name, threshold)
    retrieval, answers = run["retrieval"], run["answers"]
    records = [r for r in answers["records"] if r["answerable"]]
    unanswerable = [r for r in answers["records"] if not r["answerable"]]
    n, m = len(records), len(unanswerable)
    by_threshold = sum(r["refused"] and r["refusal_reason"] == "low_similarity" for r in records)
    by_model = sum(r["refused"] and r["refusal_reason"] == "model_declined" for r in records)
    correct = sum(bool(r["correct_citation"]) for r in records)
    refused_unanswerable = sum(r["refused"] for r in unanswerable)
    blocked = sum(r["refused"] and r["refusal_reason"] == "low_similarity" for r in unanswerable)
    ours, base = correct_flags(run), correct_flags(baseline_run)
    wins = sum(ours[q] and not base[q] for q in ours)
    losses = sum(base[q] and not ours[q] for q in ours)
    vs = "-" if name == BASELINE else f"{wins} / {losses} (p {sign_test_p(wins, losses):.2f})"
    recall = retrieval["recall_at_k"]
    return (
        f"| {name} | {label} | {round(recall['5'] * n)}/{n} | {retrieval['mrr']:.3f} "
        f"| {correct}/{n} | {vs} | {refused_unanswerable}/{m} ({blocked} threshold, "
        f"{refused_unanswerable - blocked} model) "
        f"| {by_threshold + by_model} ({by_threshold} threshold, {by_model} model) "
        f"| {answers['p95_latency_s']:.1f} s |"
    )


def main() -> None:
    lines = [
        "# Embedding models in Fine Print: tuning-set results",
        "",
        "Run by `scripts/compare_embedding_models.py`: section chunks, top 5 passages, "
        "gemma4:e4b with the `numbered` prompt. Tuning set only: 31 answerable and 15 "
        "out-of-scope questions. Fine Print's held-out set is spent and wasn't used.",
        "",
        '**Rows marked "re-tuned" chose their threshold on these same questions and were '
        "scored on them, so they are optimistic.** Rows at 0.6 use Fine Print's existing value.",
        "",
        "| Embedding model | min_similarity | Retrieval recall@5 | MRR | Correct citation "
        "| Correct-citation wins / losses vs bge-small at its matching threshold "
        "| Out-of-scope refused | Answerable wrongly refused | Answer p95 |",
        "|" + " --- |" * 9,
    ]
    for threshold_kind in ("default", "tuned"):
        for name in MODELS:
            threshold = DEFAULT if threshold_kind == "default" else tuned_threshold(name)
            label = (
                f"{threshold} (current)"
                if threshold_kind == "default"
                else f"{threshold} (re-tuned)"
            )
            base_threshold = DEFAULT if threshold_kind == "default" else tuned_threshold(BASELINE)
            lines.append(row(name, threshold, label, load(BASELINE, base_threshold)))
    lines += [
        "",
        "- **Retrieval recall@5** is Fine Print's keyword measure on the section index: the right "
        "document and every keyword in a top-5 chunk. It doesn't depend on the threshold.",
        "- **Correct citation** means the answer cites the section of a passage that passes the "
        "keyword test. Fine Print's own caveat applies: it doesn't check the answer is right.",
        "- **Wrongly refused** answerable questions are split by cause: below `min_similarity`, so "
        "the model never saw them, or declined by the model.",
        "- **Out-of-scope refused** is split the same way. Where the model declines them itself, a "
        "higher threshold adds nothing but wrong refusals.",
        "- One question is 3.2 points of 31 answerable questions; gaps of one or two are noise. "
        "Answer latency includes the real model, measured with no other apps open.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
