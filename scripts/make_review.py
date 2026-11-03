"""Write eval/review/v3_finalists.md from two saved tuning-set runs.

It uses the tuning questions only. Retrieval is repeatable, so the cited passages are looked up
again from each question.

Usage: uv run python scripts/make_review.py MODEL/PROMPT MODEL/PROMPT
"""

import argparse
from pathlib import Path

from compare_answers import run_path

from fine_print.config import PROJECT_ROOT, get_settings
from fine_print.evaluate import AnswerResult, EvalQuestion, load_jsonl
from fine_print.models import RetrievedChunk
from fine_print.rag import retrieve, select_passages
from fine_print.review import render_review

REVIEW_PATH = PROJECT_ROOT / "eval" / "review" / "v3_finalists.md"


def load_run(spec: str) -> AnswerResult:
    model, prompt = spec.split("/")
    return AnswerResult.model_validate_json(run_path(model, prompt).read_text(encoding="utf-8"))


def shown(question: str) -> list[RetrievedChunk]:
    return select_passages(retrieve(question))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finalists", nargs=2, metavar="MODEL/PROMPT")
    args = parser.parse_args()

    results = [load_run(spec) for spec in args.finalists]
    questions = load_jsonl(get_settings().questions_path, EvalQuestion)
    path = Path(REVIEW_PATH)
    if path.exists():
        raise SystemExit(f"{path} already exists, and may hold marks. Move it first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(results, questions, shown), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
