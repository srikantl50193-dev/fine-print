"""Measure retrieval quality for each chunking strategy, with no LLM involved.

A chunk counts as relevant if it belongs to the expected document and contains ALL of the
question's keywords. This is a stand-in for true relevance: it favours longer chunks, and it
can't see whether a passage really answers the question. So the table also reports each
strategy's chunk count and mean length, which any comparison has to take into account.
"""

from pathlib import Path

from pydantic import Field

from fine_print.config import Strategy, get_settings
from fine_print.models import Answer, FrozenModel, RetrievedChunk
from fine_print.rag import answer_question, retrieve
from fine_print.store import collection_stats


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


class AnswerResult(FrozenModel):
    """How the whole pipeline, including the model, treats the questions."""

    answered: float
    cited_any: float
    cited_relevant: float
    unanswerable_refused: float
    wrongly_refused: list[str]
    wrongly_answered: list[str]


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


def _cites_relevant_passage(answer: Answer, question: EvalQuestion) -> bool:
    """Check whether a passage the answer cites is one that contains the expected text."""
    cited = {citation.chunk_id for citation in answer.citations}
    return any(c.chunk_id in cited and is_relevant(c, question) for c in answer.retrieved)


def evaluate_answers(
    strategy: Strategy,
    answerable: list[EvalQuestion],
    unanswerable: list[OutOfScopeQuestion],
) -> AnswerResult:
    """Ask the model every question. This is slow, and needs Ollama running.

    A passage counts as cited when the answer uses its label. Nothing here checks that the
    passage supports what the answer claims, so these are upper bounds on correctness.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
        OllamaError: If Ollama cannot answer.
    """
    answers = [answer_question(q.question, strategy) for q in answerable]
    replied = [a for a in answers if not a.refused]
    wrongly_answered = [
        q.question for q in unanswerable if not answer_question(q.question, strategy).refused
    ]
    return AnswerResult(
        answered=len(replied) / len(answerable),
        cited_any=sum(bool(a.citations) for a in replied) / len(answerable),
        cited_relevant=sum(
            _cites_relevant_passage(a, q) for a, q in zip(answers, answerable, strict=True)
        )
        / len(answerable),
        unanswerable_refused=1 - len(wrongly_answered) / len(unanswerable),
        wrongly_refused=[a.question for a in answers if a.refused],
        wrongly_answered=wrongly_answered,
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
