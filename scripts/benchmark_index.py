"""Compare exact search (FAISS flat) with approximate search (FAISS HNSW).

The corpus is the chunk vectors already stored in Chroma, and the queries are the
questions in eval/questions.jsonl. Flat search checks every vector, so its results
are the ground truth that HNSW's recall is measured against.

Usage: uv run python scripts/benchmark_index.py --strategy section
"""

import argparse
import time
from dataclasses import dataclass

import faiss
import numpy as np

from fine_print import config
from fine_print.embed import embed_query
from fine_print.evaluate import load_questions
from fine_print.store import IndexNotBuiltError, get_all_embeddings


@dataclass(frozen=True)
class BenchRow:
    name: str
    ef_search: str
    recall: float
    latency_ms: float
    build_ms: float


def build_flat(vectors: np.ndarray) -> tuple[faiss.Index, float]:
    """Build an exact index. Inner product equals cosine on normalized vectors."""
    start = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, (time.perf_counter() - start) * 1000


def build_hnsw(vectors: np.ndarray) -> tuple[faiss.IndexHNSWFlat, float]:
    """Build an approximate graph index with the settings from config."""
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(vectors.shape[1], config.BENCH_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.BENCH_EF_CONSTRUCTION
    index.add(vectors)
    return index, (time.perf_counter() - start) * 1000


def measure_latency_ms(index: faiss.Index, queries: np.ndarray, k: int) -> float:
    """Return the mean milliseconds per query, searching one query at a time."""
    start = time.perf_counter()
    for _ in range(config.BENCH_LATENCY_REPEATS):
        for query in queries:
            index.search(query.reshape(1, -1), k)
    total_queries = config.BENCH_LATENCY_REPEATS * len(queries)
    return (time.perf_counter() - start) * 1000 / total_queries


def recall_against(truth: np.ndarray, found: np.ndarray) -> float:
    """Return the share of the true top-k neighbours that the index also found."""
    hits = sum(len(set(t) & set(f)) for t, f in zip(truth.tolist(), found.tolist()))
    return hits / truth.size


def run_benchmark(vectors: np.ndarray, queries: np.ndarray) -> tuple[list[BenchRow], int]:
    """Benchmark flat search, then HNSW at each efSearch value."""
    k = min(config.BENCH_K, len(vectors))
    flat, flat_build_ms = build_flat(vectors)
    _, truth = flat.search(queries, k)
    rows = [
        BenchRow("Flat", "-", 1.0, measure_latency_ms(flat, queries, k), flat_build_ms),
    ]

    hnsw, hnsw_build_ms = build_hnsw(vectors)
    for ef_search in config.BENCH_EF_SEARCH_VALUES:
        hnsw.hnsw.efSearch = ef_search
        _, found = hnsw.search(queries, k)
        rows.append(
            BenchRow(
                "HNSW",
                str(ef_search),
                recall_against(truth, found),
                measure_latency_ms(hnsw, queries, k),
                hnsw_build_ms,
            )
        )
    return rows, k


def format_table(rows: list[BenchRow], k: int) -> str:
    """Render the results as a Markdown table, ready to paste into the README."""
    lines = [
        f"| Index | efSearch | Recall@{k} vs flat | Query latency (ms) | Build time (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.ef_search} | {row.recall:.3f} "
            f"| {row.latency_ms:.4f} | {row.build_ms:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=config.STRATEGIES, default=config.DEFAULT_STRATEGY)
    strategy = parser.parse_args().strategy

    try:
        vectors = get_all_embeddings(strategy).astype(np.float32)
    except IndexNotBuiltError as error:
        raise SystemExit(f"Error: {error}") from error
    queries = np.stack([embed_query(q.question) for q in load_questions()]).astype(np.float32)

    # One thread, so latency reflects the algorithm and not the number of CPU cores.
    faiss.omp_set_num_threads(1)
    rows, k = run_benchmark(vectors, queries)

    print(f"\n{len(vectors)} chunks ('{strategy}' strategy), {len(queries)} queries, k={k}\n")
    print(format_table(rows, k))


if __name__ == "__main__":
    main()
