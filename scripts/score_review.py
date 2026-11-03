"""Score the marks in eval/review/v3_finalists.md.

Usage: uv run python scripts/score_review.py MODEL/PROMPT MODEL/PROMPT
"""

import argparse

from make_review import REVIEW_PATH, load_run

from fine_print.review import score_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finalists", nargs=2, metavar="MODEL/PROMPT")
    args = parser.parse_args()

    results = [load_run(spec) for spec in args.finalists]
    for score in score_review(REVIEW_PATH.read_text(encoding="utf-8"), results):
        print(
            f"{score.run}: {score.accuracy:.0%} correct of {score.marked} marked "
            f"({score.unmarked} unmarked)\n"
            f"  answerable: {score.answerable_correct}/{score.answerable_marked}\n"
            f"  unanswerable: {score.unanswerable_correct}/{score.unanswerable_marked}\n"
            f"  links out: {score.links_out}\n"
            f"  agrees with the automated verdict: {score.agrees_with_automated}/{score.marked}"
        )


if __name__ == "__main__":
    main()
