"""Build a file for marking answers by hand, and score the marks.

Automated scores can't tell whether an answer is right: a correct answer without a citation
fails, and a wrong answer citing the right section passes. So people read the replies too.
"""

import re
from collections.abc import Callable, Sequence

from fine_print.evaluate import AnswerRecord, AnswerResult, EvalQuestion
from fine_print.models import FrozenModel, RetrievedChunk

MARK_CELL = "[ ] correct / [ ] incorrect / [ ] links out"
MARK_ROW = "| **Your mark** |"
CHECKED_CORRECT = re.compile(r"\[x\]\s*correct", re.IGNORECASE)
CHECKED_INCORRECT = re.compile(r"\[x\]\s*incorrect", re.IGNORECASE)
CHECKED_LINKS_OUT = re.compile(r"\[x\]\s*links out", re.IGNORECASE)
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")

INSTRUCTIONS = """# v3 finalist review

For each question and finalist, read the reply against the cited passage, then replace `[ ]`
with `[x]` in the **Your mark** row:

- **correct**: the reply answers the question accurately, going by the policy. For an
  unanswerable question, refusing is correct. A refusal of an answerable question is incorrect.
- **incorrect**: anything else, including a reply that is partly wrong.
- **links out** (optional, on top of either mark): the reply points to an external page that
  the terms refer to, such as a help center. This is a plus and isn't required.

Mark correct or incorrect exactly once per cell. Then run:

`uv run python scripts/score_review.py {finalists}`

The expected phrase and automated verdict are there for comparison. Judge the reply itself,
because they can be wrong in both directions. A cited passage shows every shown passage under
the cited label, so a section split into several passages appears in full.
"""


class ReviewScore(FrozenModel):
    """One finalist's marks."""

    run: str
    marked: int
    unmarked: int
    answerable_correct: int
    answerable_marked: int
    unanswerable_correct: int
    unanswerable_marked: int
    links_out: int
    agrees_with_automated: int

    @property
    def accuracy(self) -> float:
        """Return the share of marked questions that are correct."""
        correct = self.answerable_correct + self.unanswerable_correct
        return correct / self.marked if self.marked else 0.0


def _cell(text: str) -> str:
    """Make text safe for one cell of a Markdown table."""
    escaped = text.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;")
    return "<br>".join(line for line in escaped.splitlines() if line.strip())


def automated_verdict(record: AnswerRecord) -> str:
    """Describe what the automated scoring concluded about a reply."""
    if not record.answerable:
        return "refused (pass)" if record.refused else "answered an unanswerable question (fail)"
    return "correct citation (pass)" if record.correct_citation else f"{record.failure} (fail)"


def automated_pass(record: AnswerRecord) -> bool:
    """Return whether the automated scoring passed a reply."""
    return record.refused if not record.answerable else bool(record.correct_citation)


def run_name(result: AnswerResult) -> str:
    """Name a model and prompt combination."""
    return f"{result.model} + {result.prompt}"


def _passages(record: AnswerRecord, shown: Callable[[str], list[RetrievedChunk]]) -> str:
    """Return the text of the passages under the labels a reply cited."""
    if not record.cited_labels:
        return "(none)"
    cited = [c for c in shown(record.question) if c.label in record.cited_labels]
    return "\n\n".join(f"{c.label} ({c.doc_type})\n{c.text}" for c in cited) or "(none)"


def render_review(
    results: Sequence[AnswerResult],
    questions: Sequence[EvalQuestion],
    shown: Callable[[str], list[RetrievedChunk]],
) -> str:
    """Lay out every question with each finalist's reply, cited passage and verdict.

    Args:
        results: The finalists' runs, which must cover the same questions in the same order.
        questions: The answerable questions, for their company and expected phrases.
        shown: Returns the passages the model was shown for a question.

    Returns:
        The Markdown file, with an empty mark for each reply.
    """
    details = {q.question: q for q in questions}
    header = " | ".join(run_name(r) for r in results)
    divider = " --- |" * (len(results) + 1)
    finalists = " ".join(f"{r.model}/{r.prompt}" for r in results)
    blocks = [INSTRUCTIONS.format(finalists=finalists)]
    for number, records in enumerate(zip(*(r.records for r in results), strict=True), start=1):
        first = records[0]
        if first.answerable:
            q = details[first.question]
            expected = "; ".join(q.keywords)
            title = f"## Q{number} (answerable, {q.company}, {q.doc_type})"
            intro = f"**{first.question}**\n\nExpected phrase: `{expected}`"
        else:
            title = f"## Q{number} (unanswerable)"
            intro = f"**{first.question}**\n\nThe corpus shouldn't answer this."
        rows = [
            f"| | {header} |",
            f"|{divider}",
            "| **Reply** | " + " | ".join(_cell(r.text) for r in records) + " |",
            "| **Cited passage** | "
            + " | ".join(_cell(_passages(r, shown)) for r in records)
            + " |",
            "| **Automated** | " + " | ".join(automated_verdict(r) for r in records) + " |",
            MARK_ROW + " " + " | ".join(MARK_CELL for _ in records) + " |",
        ]
        blocks.append(f"{title}\n\n{intro}\n\n" + "\n".join(rows))
    return "\n\n".join(blocks) + "\n"


def score_review(text: str, results: Sequence[AnswerResult]) -> list[ReviewScore]:
    """Count the marks in a review file.

    Args:
        text: The contents of the review file.
        results: The runs the file was built from, in the same order.

    Returns:
        One score per run. A cell with neither mark, or with both, counts as unmarked.
    """
    rows = [
        [cell.strip() for cell in UNESCAPED_PIPE.split(line.strip().removeprefix(MARK_ROW))]
        for line in text.splitlines()
        if line.startswith(MARK_ROW)
    ]
    scores = []
    for index, result in enumerate(results):
        tally = dict.fromkeys(("answerable", "unanswerable"), 0)
        marked = dict.fromkeys(("answerable", "unanswerable"), 0)
        links_out = agrees = 0
        for row, record in zip(rows, result.records, strict=True):
            cell = row[index]
            correct = bool(CHECKED_CORRECT.search(cell))
            incorrect = bool(CHECKED_INCORRECT.search(cell))
            links_out += bool(CHECKED_LINKS_OUT.search(cell))
            if correct == incorrect:
                continue
            kind = "answerable" if record.answerable else "unanswerable"
            marked[kind] += 1
            tally[kind] += correct
            agrees += correct == automated_pass(record)
        total = sum(marked.values())
        scores.append(
            ReviewScore(
                run=run_name(result),
                marked=total,
                unmarked=len(rows) - total,
                answerable_correct=tally["answerable"],
                answerable_marked=marked["answerable"],
                unanswerable_correct=tally["unanswerable"],
                unanswerable_marked=marked["unanswerable"],
                links_out=links_out,
                agrees_with_automated=agrees,
            )
        )
    return scores
