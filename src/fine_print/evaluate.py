"""Measure retrieval quality for each chunking strategy, with no LLM involved.

A chunk counts as relevant if it belongs to the expected company and contains
ALL of the question's keywords. Run with: uv run python -m fine_print.evaluate
"""

import json
from dataclasses import dataclass
from pathlib import Path

from fine_print import config
from fine_print.rag import retrieve
from fine_print.store import IndexNotBuiltError, RetrievedChunk, get_collection


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    company: str
    keywords: list[str]


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    num_chunks: int
    recall_at_k: dict[int, float]
    mrr: float
    misses: list[str]


def load_questions(path: Path = config.EVAL_QUESTIONS_PATH) -> list[EvalQuestion]:
    """Read the questions file, which holds one JSON object per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion(**json.loads(line)) for line in lines if line.strip()]


def is_relevant(chunk: RetrievedChunk, question: EvalQuestion) -> bool:
    """Check the company and that every keyword appears, ignoring case."""
    text = chunk.text.lower()
    return chunk.company == question.company and all(
        keyword.lower() in text for keyword in question.keywords
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


def evaluate_strategy(strategy: str, questions: list[EvalQuestion]) -> StrategyResult:
    """Run every question against one strategy's index and score the results."""
    deepest_k = max(config.EVAL_K_VALUES)
    ranks = [
        first_relevant_rank(retrieve(q.question, strategy, top_k=deepest_k), q)
        for q in questions
    ]
    return StrategyResult(
        strategy=strategy,
        num_chunks=get_collection(strategy).count(),
        recall_at_k={k: recall_at_k(ranks, k) for k in config.EVAL_K_VALUES},
        mrr=mean_reciprocal_rank(ranks),
        misses=[q.question for q, rank in zip(questions, ranks) if rank is None],
    )


def format_table(results: list[StrategyResult]) -> str:
    """Render the results as a Markdown table, ready to paste into the README."""
    recall_headers = [f"Recall@{k}" for k in config.EVAL_K_VALUES]
    headers = ["Strategy", "Chunks", *recall_headers, "MRR"]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + " --- |" * len(headers),
    ]
    for result in results:
        recalls = [f"{result.recall_at_k[k]:.2f}" for k in config.EVAL_K_VALUES]
        cells = [result.strategy, str(result.num_chunks), *recalls, f"{result.mrr:.3f}"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    questions = load_questions()
    try:
        results = [evaluate_strategy(s, questions) for s in config.STRATEGIES]
    except IndexNotBuiltError as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"\nEvaluated {len(questions)} questions\n\n{format_table(results)}")
    for result in results:
        for question in result.misses:
            print(f"Missed by '{result.strategy}' in top {max(config.EVAL_K_VALUES)}: {question}")


if __name__ == "__main__":
    main()
