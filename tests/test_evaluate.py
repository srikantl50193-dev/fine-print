import pytest

from fine_print.evaluate import (
    EvalQuestion,
    first_relevant_rank,
    is_relevant,
    load_questions,
    mean_reciprocal_rank,
    recall_at_k,
)
from fine_print.store import RetrievedChunk

QUESTION = EvalQuestion("Any fee?", "acme", ["$5", "cancellation"])


def make_chunk(text: str, company: str = "acme") -> RetrievedChunk:
    return RetrievedChunk(text, company, "terms", "Fees", "id", 0.5)


def test_relevant_needs_the_right_company_and_all_keywords() -> None:
    assert is_relevant(make_chunk("A $5 CANCELLATION fee."), QUESTION)
    assert not is_relevant(make_chunk("A $5 fee."), QUESTION)
    assert not is_relevant(make_chunk("A $5 cancellation fee.", "beta"), QUESTION)


def test_first_relevant_rank_is_one_based() -> None:
    chunks = [make_chunk("no"), make_chunk("$5 cancellation")]
    assert first_relevant_rank(chunks, QUESTION) == 2
    assert first_relevant_rank(chunks[:1], QUESTION) is None


def test_metrics_on_a_small_known_example() -> None:
    ranks = [1, 2, None, 4]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.5
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_shipped_questions_file_is_well_formed() -> None:
    questions = load_questions()
    assert len(questions) >= 10
    assert all(q.question and q.company and q.keywords for q in questions)
