import pytest

from fine_print.evaluate import AnswerRecord, AnswerResult, EvalQuestion
from fine_print.models import RetrievedChunk
from fine_print.review import MARK_CELL, automated_verdict, render_review, score_review

QUESTION = EvalQuestion(question="Any fee?", company="acme", doc_type="terms", keywords=["$5"])
CHUNK = RetrievedChunk(
    text="A $5 | fee applies.",
    company="acme",
    doc_type="terms",
    section="Fees",
    chunk_id="id",
    strategy="section",
    start_char=0,
    end_char=18,
    similarity=0.8,
)
OTHER = CHUNK.model_copy(update={"section": "Other", "text": "Not cited."})


def answerable(**fields: object) -> AnswerRecord:
    defaults = {
        "question": "Any fee?",
        "answerable": True,
        "text": "It is $5\nper ride.",
        "refused": False,
        "refusal_reason": None,
        "cited_labels": ["[Acme – Fees]"],
        "correct_citation": True,
    }
    return AnswerRecord.model_validate(defaults | fields)


def unanswerable(**fields: object) -> AnswerRecord:
    defaults = {
        "question": "Off topic?",
        "answerable": False,
        "text": "Not found in the documents",
        "refused": True,
        "refusal_reason": "model_declined",
        "cited_labels": [],
    }
    return AnswerRecord.model_validate(defaults | fields)


def run(model: str, records: list[AnswerRecord]) -> AnswerResult:
    return AnswerResult(
        model=model,
        prompt="numbered",
        answered=1.0,
        cited_any=1.0,
        cited_relevant=1.0,
        unanswerable_refused=1.0,
        p95_latency_s=1.0,
        failures={},
        wrongly_refused=[],
        wrongly_answered=[],
        records=records,
    )


@pytest.fixture
def results() -> list[AnswerResult]:
    return [
        run("a", [answerable(), unanswerable()]),
        run(
            "b",
            [
                answerable(
                    correct_citation=False, failure="format_error", cited_labels=[], text="$5."
                ),
                unanswerable(refused=False, text="Paris."),
            ],
        ),
    ]


@pytest.fixture
def review(results: list[AnswerResult]) -> str:
    return render_review(results, [QUESTION], lambda question: [CHUNK, OTHER])


def test_review_shows_every_question_with_both_finalists_side_by_side(review: str) -> None:
    assert "| | a + numbered | b + numbered |" in review
    assert "## Q1 (answerable, acme, terms)" in review
    assert "**Any fee?**" in review
    assert "Expected phrase: `$5`" in review
    assert "## Q2 (unanswerable)" in review
    assert "**Off topic?**" in review
    assert review.count(MARK_CELL) == 4


def test_review_shows_only_the_cited_passage_escaped_for_a_table(review: str) -> None:
    assert "A $5 \\| fee applies." in review
    assert "[Acme – Fees] (terms)" in review
    assert "Not cited." not in review
    assert "(none)" in review


def test_replies_keep_their_line_breaks_in_one_table_row(review: str) -> None:
    assert "It is $5<br>per ride." in review


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (answerable(), "correct citation (pass)"),
        (answerable(correct_citation=False, failure="format_error"), "format_error (fail)"),
        (unanswerable(), "refused (pass)"),
        (unanswerable(refused=False), "answered an unanswerable question (fail)"),
    ],
)
def test_automated_verdict(record: AnswerRecord, expected: str) -> None:
    assert automated_verdict(record) == expected


def fill(review: str, marks: list[str]) -> str:
    for mark in marks:
        review = review.replace(MARK_CELL, mark, 1)
    return review


CORRECT = "[x] correct / [ ] incorrect / [ ] links out"
INCORRECT = "[ ] correct / [X] incorrect / [ ] links out"
CORRECT_LINKS = "[x] correct / [ ] incorrect / [x] links out"
BOTH = "[x] correct / [x] incorrect / [ ] links out"


def test_marks_are_scored_per_finalist(review: str, results: list[AnswerResult]) -> None:
    # Cells fill left to right: Q1 for a, Q1 for b, Q2 for a, Q2 for b.
    marked = fill(review, [CORRECT_LINKS, CORRECT, CORRECT, INCORRECT])

    first, second = score_review(marked, results)

    assert first.run == "a + numbered"
    assert (first.answerable_correct, first.answerable_marked) == (1, 1)
    assert (first.unanswerable_correct, first.unanswerable_marked) == (1, 1)
    assert first.accuracy == 1.0
    assert first.links_out == 1
    assert first.agrees_with_automated == 2

    assert (second.answerable_correct, second.answerable_marked) == (1, 1)
    assert (second.unanswerable_correct, second.unanswerable_marked) == (0, 1)
    assert second.accuracy == 0.5
    assert second.links_out == 0
    # The human passed an answer the automation failed, and agreed that Q2 was wrong.
    assert second.agrees_with_automated == 1


def test_review_names_the_finalists_in_the_scoring_command(review: str) -> None:
    assert "scripts/score_review.py a/numbered b/numbered" in review


def test_blank_and_contradictory_marks_count_as_unmarked(
    review: str, results: list[AnswerResult]
) -> None:
    first, second = score_review(fill(review, [BOTH]), results)

    assert (first.marked, first.unmarked) == (0, 2)
    assert first.accuracy == 0.0
    assert (second.marked, second.unmarked) == (0, 2)


def test_links_out_does_not_count_as_a_mark_on_its_own(
    review: str, results: list[AnswerResult]
) -> None:
    first, _ = score_review(fill(review, ["[ ] correct / [ ] incorrect / [x] links out"]), results)
    assert first.marked == 0
    assert first.links_out == 1
