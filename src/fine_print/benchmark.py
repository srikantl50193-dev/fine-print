"""Compare exact search (FAISS flat) with approximate search (FAISS HNSW).

The corpus is the chunk vectors already stored in Chroma, and the queries are the evaluation
questions. Flat search checks every vector, so its results are the ground truth that HNSW's
recall is measured against. Chroma does its own HNSW search, so this measures the algorithm's
trade-off on our vectors, not Chroma's speed.
"""

import time
from pathlib import Path

import faiss
import matplotlib as mpl
import numpy as np

from fine_print.config import get_settings
from fine_print.models import FrozenModel

mpl.use("Agg")  # Writes image files, with no display needed.

from matplotlib import pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HNSW_COLOR = "#2a78d6"
FLAT_COLOR = "#eb6834"


class BenchRow(FrozenModel):
    """One index configuration and how it performed."""

    name: str
    ef_search: int | None
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
    """Build an approximate graph index with the same settings as the Chroma index."""
    settings = get_settings()
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(
        vectors.shape[1], settings.hnsw_max_neighbors, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = settings.hnsw_ef_construction
    index.add(vectors)
    return index, (time.perf_counter() - start) * 1000


def measure_latency_ms(index: faiss.Index, queries: np.ndarray, k: int) -> float:
    """Return the mean milliseconds per query, searching one query at a time."""
    repeats = get_settings().bench_latency_repeats
    start = time.perf_counter()
    for _ in range(repeats):
        for query in queries:
            index.search(query.reshape(1, -1), k)
    return (time.perf_counter() - start) * 1000 / (repeats * len(queries))


def recall_against(truth: np.ndarray, found: np.ndarray) -> float:
    """Return the share of the true top-k neighbours that the index also found."""
    hits = sum(len(set(t) & set(f)) for t, f in zip(truth.tolist(), found.tolist(), strict=True))
    return hits / truth.size


def run_benchmark(vectors: np.ndarray, queries: np.ndarray) -> tuple[list[BenchRow], int]:
    """Benchmark flat search, then HNSW at each efSearch value.

    Args:
        vectors: The corpus, as normalized float32 vectors.
        queries: The queries, as normalized float32 vectors.

    Returns:
        One row per configuration, and the k that was used.
    """
    settings = get_settings()
    # One thread, so latency reflects the algorithm and not the number of CPU cores.
    faiss.omp_set_num_threads(1)
    k = min(settings.bench_k, len(vectors))

    flat, flat_build_ms = build_flat(vectors)
    _, truth = flat.search(queries, k)
    rows = [
        BenchRow(
            name="Flat",
            ef_search=None,
            recall=1.0,
            latency_ms=measure_latency_ms(flat, queries, k),
            build_ms=flat_build_ms,
        )
    ]

    hnsw, hnsw_build_ms = build_hnsw(vectors)
    for ef_search in settings.bench_ef_search_values:
        hnsw.hnsw.efSearch = ef_search
        _, found = hnsw.search(queries, k)
        rows.append(
            BenchRow(
                name="HNSW",
                ef_search=ef_search,
                recall=recall_against(truth, found),
                latency_ms=measure_latency_ms(hnsw, queries, k),
                build_ms=hnsw_build_ms,
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
        ef_search = "-" if row.ef_search is None else str(row.ef_search)
        lines.append(
            f"| {row.name} | {ef_search} | {row.recall:.3f} "
            f"| {row.latency_ms:.4f} | {row.build_ms:.2f} |"
        )
    return "\n".join(lines)


def save_chart(rows: list[BenchRow], k: int, path: Path) -> None:
    """Plot recall against latency, labelling each HNSW point with its efSearch."""
    hnsw = [row for row in rows if row.ef_search is not None]
    flat = [row for row in rows if row.ef_search is None]

    figure, axes = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.plot(
        [row.latency_ms for row in hnsw],
        [row.recall for row in hnsw],
        color=HNSW_COLOR,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="HNSW",
    )
    axes.plot(
        [row.latency_ms for row in flat],
        [row.recall for row in flat],
        linestyle="none",
        color=FLAT_COLOR,
        marker="s",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="Flat (exact)",
    )
    for row in hnsw:
        axes.annotate(
            f"ef {row.ef_search}",
            (row.latency_ms, row.recall),
            textcoords="offset points",
            xytext=(8, -12),
            ha="left",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    for row in flat:
        axes.annotate(
            "Flat (exact)",
            (row.latency_ms, row.recall),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    axes.set_title(
        "Higher efSearch buys recall with latency",
        loc="left",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    axes.set_xlabel("Query latency (ms, one thread)", fontsize=9, color=TEXT_SECONDARY)
    axes.set_ylabel(f"Recall@{k} against exact search", fontsize=9, color=TEXT_SECONDARY)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.margins(0.12)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
