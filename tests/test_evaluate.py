from pathlib import Path

import pytest
from pydantic import ValidationError

from fine_print import evaluate, rag
from fine_print.config import get_settings
from fine_print.evaluate import (
    EvalQuestion,
    OutOfScopeQuestion,
    classify_failure,
    evaluate_answers,
    evaluate_strategy,
    evaluate_threshold,
    first_relevant_rank,
    format_table,
    is_relevant,
    load_jsonl,
    mean_reciprocal_rank,
    recall_at_k,
)
from fine_print.models import Answer, RetrievedChunk
from fine_print.prompts import Prompt, load_prompt

QUESTION = EvalQuestion(
    question="Any fee?", company="acme", doc_type="terms", keywords=["$5", "cancellation"]
)


def make_chunk(
    text: str, company: str = "acme", doc_type: str = "terms", similarity: float = 0.8
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        company=company,
        doc_type=doc_type,
        section="Fees",
        chunk_id="id",
        strategy="section",
        start_char=0,
        end_char=len(text),
        similarity=similarity,
    )


def test_relevant_needs_the_right_document_and_all_keywords() -> None:
    assert is_relevant(make_chunk("A $5 CANCELLATION fee."), QUESTION)
    assert not is_relevant(make_chunk("A $5 fee."), QUESTION)
    assert not is_relevant(make_chunk("A $5 cancellation fee.", company="beta"), QUESTION)
    assert not is_relevant(make_chunk("A $5 cancellation fee.", doc_type="privacy"), QUESTION)


def test_keyword_broken_across_lines_still_matches() -> None:
    question = QUESTION.model_copy(update={"keywords": ["a cancellation fee"]})
    assert is_relevant(make_chunk("Charges a\ncancellation   fee."), question)


def test_first_relevant_rank_is_one_based() -> None:
    chunks = [make_chunk("no"), make_chunk("$5 cancellation")]
    assert first_relevant_rank(chunks, QUESTION) == 2
    assert first_relevant_rank(chunks[:1], QUESTION) is None


def test_metrics_on_a_small_known_example() -> None:
    ranks = [1, 2, None, 4]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.5
    assert recall_at_k(ranks, 4) == 0.75
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_evaluate_strategy_scores_every_question(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = EvalQuestion(question="Hit?", company="acme", doc_type="terms", keywords=["found"])
    miss = EvalQuestion(question="Miss?", company="acme", doc_type="terms", keywords=["absent"])
    requested = []

    def fake_retrieve(question: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        requested.append((strategy, top_k))
        return [make_chunk("other"), make_chunk("found it")]

    monkeypatch.setattr(evaluate, "retrieve", fake_retrieve)
    monkeypatch.setattr(evaluate, "collection_stats", lambda strategy: (7, 321.0))

    result = evaluate_strategy("section", [hit, miss])

    assert requested[0] == ("section", max(get_settings().eval_k_values))
    assert result.recall_at_k[1] == 0.0
    assert result.recall_at_k[3] == 0.5
    assert result.mrr == 0.25
    assert result.misses == ["Miss?"]
    assert (result.num_chunks, result.mean_chunk_chars) == (7, 321.0)


def test_threshold_counts_both_kinds_of_mistake(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {"Good": 0.8, "Weak": 0.3, "Off": 0.2, "Close": 0.6}

    def fake_retrieve(question: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        return [make_chunk("text", similarity=scores[question])]

    monkeypatch.setattr(evaluate, "retrieve", fake_retrieve)
    answerable = [
        EvalQuestion(question=q, company="acme", doc_type="terms", keywords=["x"])
        for q in ("Good", "Weak")
    ]
    unanswerable = [OutOfScopeQuestion(question=q) for q in ("Off", "Close")]

    result = evaluate_threshold("section", answerable, unanswerable)

    assert result.threshold == get_settings().min_similarity
    assert result.answerable_kept == 0.5
    assert result.unanswerable_refused == 0.5
    assert result.wrongly_refused == ["Weak"]
    assert result.wrongly_kept == ["Close"]


def test_empty_retrieval_counts_as_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "retrieve", lambda *args: [])
    answerable = [QUESTION]
    result = evaluate_threshold("section", answerable, [OutOfScopeQuestion(question="Off")])
    assert result.answerable_kept == 0.0
    assert result.unanswerable_refused == 1.0


def fake_answer(
    chunks: tuple[RetrievedChunk, ...] = (),
    cited: tuple[RetrievedChunk, ...] = (),
    **fields: object,
) -> Answer:
    return Answer.model_validate(
        {
            "question": "Q?",
            "text": "Answer.",
            "retrieved": list(chunks),
            "citations": rag.find_citations(" ".join(c.label for c in cited), list(cited)),
            **fields,
        }
    )


RELEVANT = make_chunk("$5 cancellation")
WRONG = make_chunk("Nothing useful.").model_copy(update={"section": "Other"})


def test_answers_score_every_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    both = (RELEVANT, WRONG)
    replies = {
        "Right?": fake_answer(both, cited=(RELEVANT,)),
        "Wrong?": fake_answer(both, cited=(WRONG,)),
        "Uncited?": fake_answer(both, cited=()),
        "Refused?": fake_answer(both, cited=(), refused=True, refusal_reason="model_declined"),
        "Missed?": fake_answer((WRONG,), cited=(), refused=True, refusal_reason="model_declined"),
        "Off topic?": fake_answer(cited=(), refused=True, refusal_reason="low_similarity"),
        "Off topic, answered?": fake_answer((WRONG,), cited=(WRONG,)),
    }
    calls = []

    def answer_fake(question: str, strategy: str, *, model: str, prompt: Prompt) -> Answer:
        calls.append((model, prompt.name))
        return replies[question]

    monkeypatch.setattr(evaluate, "answer_question", answer_fake)
    answerable = [
        QUESTION.model_copy(update={"question": q})
        for q in ("Right?", "Wrong?", "Uncited?", "Refused?", "Missed?")
    ]
    unanswerable = [OutOfScopeQuestion(question=q) for q in ("Off topic?", "Off topic, answered?")]

    result = evaluate_answers(
        "section", answerable, unanswerable, "qwen3:8b", load_prompt("numbered")
    )

    assert set(calls) == {("qwen3:8b", "numbered")}
    assert (result.model, result.prompt) == ("qwen3:8b", "numbered")
    assert result.answered == 0.6
    assert result.cited_any == 0.4
    assert result.cited_relevant == 0.2
    assert result.wrongly_refused == ["Refused?", "Missed?"]
    assert result.unanswerable_refused == 0.5
    assert result.wrongly_answered == ["Off topic, answered?"]
    assert result.failures == {
        "cited_wrong_passage": 1,
        "format_error": 1,
        "retrieved_but_ignored": 1,
        "not_retrieved": 1,
    }
    failures = {r.question: r.failure for r in result.records}
    assert failures["Right?"] is None
    assert failures["Wrong?"] == "cited_wrong_passage"
    assert failures["Uncited?"] == "format_error"
    assert failures["Refused?"] == "retrieved_but_ignored"
    assert failures["Missed?"] == "not_retrieved"


def test_passage_below_the_threshold_counts_as_not_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weak = make_chunk("$5 cancellation", similarity=0.1)
    answer = fake_answer((weak,), refused=True, refusal_reason="low_similarity")
    monkeypatch.setattr(evaluate, "answer_question", lambda *args, **kwargs: answer)

    result = evaluate_answers("section", [QUESTION], [OutOfScopeQuestion(question="Off?")])

    assert result.failures == {"not_retrieved": 1}
    assert result.p95_latency_s == 0.0


def test_latency_counts_only_questions_the_model_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            fake_answer((RELEVANT,), cited=(RELEVANT,)),
            fake_answer(refused=True, refusal_reason="low_similarity"),
        ]
    )
    monkeypatch.setattr(evaluate, "answer_question", lambda *args, **kwargs: next(replies))

    result = evaluate_answers("section", [QUESTION], [OutOfScopeQuestion(question="Off?")])

    assert result.records[0].latency_s is not None
    assert result.records[1].latency_s is None


@pytest.mark.parametrize(
    ("fields", "shown", "correct", "expected"),
    [
        ({}, True, True, None),
        ({}, False, False, "not_retrieved"),
        ({"refused": True}, False, False, "not_retrieved"),
        ({"refused": True}, True, False, "retrieved_but_ignored"),
        ({"cited": True}, True, False, "cited_wrong_passage"),
        ({}, True, False, "format_error"),
    ],
)
def test_classify_failure(
    fields: dict[str, bool], shown: bool, correct: bool, expected: str | None
) -> None:
    cited = (RELEVANT,) if fields.pop("cited", False) else ()
    answer = fake_answer((RELEVANT,), cited=cited, **fields)
    assert classify_failure(answer, shown, correct) == expected


def test_table_has_a_column_per_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "retrieve", lambda *args: [make_chunk("$5 cancellation")])
    monkeypatch.setattr(evaluate, "collection_stats", lambda strategy: (7, 321.4))

    table = format_table([evaluate_strategy("fixed", [QUESTION])])

    assert (
        "| Strategy | Chunks | Mean chars | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |"
        in table
    )
    assert "| fixed | 7 | 321 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |" in table


def test_load_jsonl_skips_blank_lines_and_rejects_bad_ones(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question": "One?"}\n\n{"question": "Two?"}\n')
    assert [q.question for q in load_jsonl(path, OutOfScopeQuestion)] == ["One?", "Two?"]

    path.write_text('{"question": ""}\n')
    with pytest.raises(ValidationError):
        load_jsonl(path, OutOfScopeQuestion)


def test_held_out_questions_are_well_formed_and_separate_from_the_tuning_set() -> None:
    settings = get_settings()
    questions = load_jsonl(settings.holdout_questions_path, EvalQuestion)
    off_topic = load_jsonl(settings.holdout_out_of_scope_path, OutOfScopeQuestion)
    assert len(questions) == 20
    assert len(off_topic) == 10
    assert {q.doc_type for q in questions} == {"terms", "privacy"}
    assert len({q.company for q in questions}) == 10

    tuning = {q.question for q in load_jsonl(settings.questions_path, EvalQuestion)}
    tuning |= {q.question for q in load_jsonl(settings.out_of_scope_path, OutOfScopeQuestion)}
    assert not tuning & {q.question for q in [*questions, *off_topic]}


def test_shipped_question_files_are_well_formed() -> None:
    settings = get_settings()
    questions = load_jsonl(settings.questions_path, EvalQuestion)
    assert len(questions) >= 30
    assert len({q.question for q in questions}) == len(questions)
    assert {q.doc_type for q in questions} == {"terms", "privacy"}
    assert len({q.company for q in questions}) == 10
    assert len(load_jsonl(settings.out_of_scope_path, OutOfScopeQuestion)) >= 10
