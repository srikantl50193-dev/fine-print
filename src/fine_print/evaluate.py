"""Measure retrieval quality for each chunking strategy, with no LLM involved.

A chunk counts as relevant if it belongs to the expected document and contains ALL of the
question's keywords. This is a stand-in for true relevance: it favours longer chunks, and it
can't see whether a passage really answers the question. So the table also reports each
strategy's chunk count and mean length, which any comparison has to take into account.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from fine_print.config import Strategy, get_settings
from fine_print.models import Answer, FrozenModel, RetrievedChunk
from fine_print.prompts import Prompt, load_prompt
from fine_print.rag import answer_question, retrieve, select_passages
from fine_print.store import collection_stats

P95 = 95

type FailureKind = Literal[
    "not_retrieved", "retrieved_but_ignored", "cited_wrong_passage", "format_error"
]


class EvalQuestion(FrozenModel):
    """A question the corpus can answer, and how to recognize a passage that answers it."""

    question: str = Field(min_length=1)
    company: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)


class OutOfScopeQuestion(FrozenModel):
    """A question the corpus cannot answer, which the system should refuse."""

    question: str = Field(min_length=1)


class StrategyResult(FrozenModel):
    """The retrieval scores of one chunking strategy."""

    strategy: Strategy
    num_chunks: int
    mean_chunk_chars: float
    recall_at_k: dict[int, float]
    mrr: float
    misses: list[str]


class ThresholdResult(FrozenModel):
    """How well `min_similarity` separates answerable from unanswerable questions."""

    threshold: float
    answerable_kept: float
    unanswerable_refused: float
    wrongly_kept: list[str]
    wrongly_refused: list[str]


class AnswerRecord(FrozenModel):
    """What happened to one question, kept so that failures can be read and classified."""

    question: str
    answerable: bool
    text: str
    refused: bool
    refusal_reason: str | None
    cited_labels: list[str]
    relevant_shown: bool | None = None
    correct_citation: bool | None = None
    failure: FailureKind | None = None
    latency_s: float | None = None


class AnswerResult(FrozenModel):
    """How the whole pipeline, including the model, treats the questions."""

    model: str
    prompt: str
    answered: float
    cited_any: float
    cited_relevant: float
    unanswerable_refused: float
    p95_latency_s: float
    failures: dict[str, int]
    wrongly_refused: list[str]
    wrongly_answered: list[str]
    records: list[AnswerRecord]


def load_jsonl[T: FrozenModel](path: Path, model: type[T]) -> list[T]:
    """Read a file holding one JSON object per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If a line does not match the model.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def is_relevant(chunk: RetrievedChunk, question: EvalQuestion) -> bool:
    """Check the document and that every keyword appears, ignoring case and line breaks."""
    text = " ".join(chunk.text.lower().split())
    return (
        chunk.company == question.company
        and chunk.doc_type == question.doc_type
        and all(keyword.lower() in text for keyword in question.keywords)
    )


def first_relevant_rank(chunks: list[RetrievedChunk], question: EvalQuestion) -> int | None:
    """Return the 1-based position of the first relevant chunk, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, question):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Return the share of questions with a relevant chunk in the top k."""
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average 1/rank over all questions, where a miss scores 0."""
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def evaluate_strategy(strategy: Strategy, questions: list[EvalQuestion]) -> StrategyResult:
    """Run every question against one strategy's index and score the results.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
    """
    k_values = get_settings().eval_k_values
    deepest = max(k_values)
    ranks = [first_relevant_rank(retrieve(q.question, strategy, deepest), q) for q in questions]
    num_chunks, mean_chars = collection_stats(strategy)
    return StrategyResult(
        strategy=strategy,
        num_chunks=num_chunks,
        mean_chunk_chars=mean_chars,
        recall_at_k={k: recall_at_k(ranks, k) for k in k_values},
        mrr=mean_reciprocal_rank(ranks),
        misses=[q.question for q, rank in zip(questions, ranks, strict=True) if rank is None],
    )


def _best_similarity(question: str, strategy: Strategy) -> float:
    """Return the score of the closest passage, which decides whether the model is asked."""
    return max((chunk.similarity for chunk in retrieve(question, strategy, 1)), default=-1.0)


def evaluate_threshold(
    strategy: Strategy,
    answerable: list[EvalQuestion],
    unanswerable: list[OutOfScopeQuestion],
) -> ThresholdResult:
    """Measure how `min_similarity` treats questions the corpus can and can't answer.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
    """
    threshold = get_settings().min_similarity
    refused = [q.question for q in answerable if _best_similarity(q.question, strategy) < threshold]
    kept = [q.question for q in unanswerable if _best_similarity(q.question, strategy) >= threshold]
    return ThresholdResult(
        threshold=threshold,
        answerable_kept=1 - len(refused) / len(answerable),
        unanswerable_refused=1 - len(kept) / len(unanswerable),
        wrongly_kept=kept,
        wrongly_refused=refused,
    )


def _cites_relevant_passage(
    answer: Answer, shown: list[RetrievedChunk], question: EvalQuestion
) -> bool:
    """Check whether the answer cites the section of a passage with the expected text.

    The check is by label, so that numbered citations, which name one passage, are held to
    the same standard as label citations, which name a whole section.
    """
    cited = {citation.label for citation in answer.citations}
    return any(c.label in cited and is_relevant(c, question) for c in shown)


def classify_failure(
    answer: Answer, relevant_shown: bool, correct_citation: bool
) -> FailureKind | None:
    """Say why an answerable question didn't get a correct citation, or None if it did.

    Args:
        answer: The pipeline's answer.
        relevant_shown: Whether the model was shown a passage with the expected text.
        correct_citation: Whether the answer cited that passage's section.
    """
    if correct_citation:
        return None
    if not relevant_shown:
        return "not_retrieved"
    if answer.refused:
        return "retrieved_but_ignored"
    return "cited_wrong_passage" if answer.citations else "format_error"


def _timed_answer(
    question: str, strategy: Strategy, model: str, prompt: Prompt
) -> tuple[Answer, float | None]:
    """Answer a question, timing it only if the model was called."""
    start = time.perf_counter()
    answer = answer_question(question, strategy, model=model, prompt=prompt)
    elapsed = time.perf_counter() - start
    return answer, None if answer.refusal_reason == "low_similarity" else elapsed


def _answerable_record(
    question: EvalQuestion, answer: Answer, latency: float | None
) -> AnswerRecord:
    """Score one answerable question."""
    shown = select_passages(answer.retrieved)
    relevant_shown = any(is_relevant(c, question) for c in shown)
    correct = _cites_relevant_passage(answer, shown, question)
    return AnswerRecord(
        question=question.question,
        answerable=True,
        text=answer.text,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        cited_labels=sorted({c.label for c in answer.citations}),
        relevant_shown=relevant_shown,
        correct_citation=correct,
        failure=classify_failure(answer, relevant_shown, correct),
        latency_s=latency,
    )


def evaluate_answers(
    strategy: Strategy,
    answerable: list[EvalQuestion],
    unanswerable: list[OutOfScopeQuestion],
    model: str | None = None,
    prompt: Prompt | None = None,
) -> AnswerResult:
    """Ask the model every question. This is slow, and needs Ollama running.

    A passage counts as cited when the answer names its section. Nothing here checks that the
    passage supports what the answer claims, so these are upper bounds on correctness.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
        OllamaError: If Ollama cannot answer.
    """
    settings = get_settings()
    model = model or settings.ollama_model
    prompt = prompt or load_prompt(settings.prompt_name)

    records = [
        _answerable_record(q, *_timed_answer(q.question, strategy, model, prompt))
        for q in answerable
    ]
    for q in unanswerable:
        answer, latency = _timed_answer(q.question, strategy, model, prompt)
        records.append(
            AnswerRecord(
                question=q.question,
                answerable=False,
                text=answer.text,
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
                cited_labels=sorted({c.label for c in answer.citations}),
                latency_s=latency,
            )
        )

    scored = [r for r in records if r.answerable]
    wrongly_answered = [r.question for r in records if not r.answerable and not r.refused]
    latencies = [r.latency_s for r in records if r.latency_s is not None]
    return AnswerResult(
        model=model,
        prompt=prompt.name,
        answered=sum(not r.refused for r in scored) / len(scored),
        cited_any=sum(bool(r.cited_labels) for r in scored) / len(scored),
        cited_relevant=sum(bool(r.correct_citation) for r in scored) / len(scored),
        unanswerable_refused=1 - len(wrongly_answered) / len(unanswerable),
        p95_latency_s=float(np.percentile(latencies, P95)) if latencies else 0.0,
        failures=dict(Counter(r.failure for r in scored if r.failure)),
        wrongly_refused=[r.question for r in scored if r.refused],
        wrongly_answered=wrongly_answered,
        records=records,
    )


def format_table(results: list[StrategyResult]) -> str:
    """Render the results as a Markdown table, ready to paste into the README."""
    k_values = get_settings().eval_k_values
    headers = ["Strategy", "Chunks", "Mean chars", *[f"Recall@{k}" for k in k_values], "MRR"]
    rows = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for result in results:
        recalls = [f"{result.recall_at_k[k]:.2f}" for k in k_values]
        cells = [
            result.strategy,
            str(result.num_chunks),
            f"{result.mean_chunk_chars:.0f}",
            *recalls,
            f"{result.mrr:.3f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
