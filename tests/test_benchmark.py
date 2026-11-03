from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from fine_print.benchmark import (
    BenchRow,
    format_table,
    recall_against,
    run_benchmark,
    save_chart,
)
from fine_print.config import get_settings

PNG_SIGNATURE = b"\x89PNG"


@pytest.fixture(autouse=True)
def few_repeats(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FINE_PRINT_BENCH_LATENCY_REPEATS", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def unit_vectors(count: int, seed: int) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(count, 16)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_recall_counts_the_shared_neighbours() -> None:
    truth = np.array([[1, 2], [3, 4]])
    assert recall_against(truth, np.array([[2, 1], [3, 9]])) == 0.75


def test_benchmark_has_flat_as_truth_and_a_row_per_ef_search() -> None:
    rows, k = run_benchmark(unit_vectors(500, seed=1), unit_vectors(20, seed=2))
    settings = get_settings()

    assert k == settings.bench_k
    assert rows[0].name == "Flat"
    assert rows[0].recall == 1.0
    assert [row.ef_search for row in rows[1:]] == list(settings.bench_ef_search_values)
    assert all(0 < row.recall <= 1 for row in rows)
    assert all(row.latency_ms > 0 and row.build_ms > 0 for row in rows)


def test_raising_ef_search_never_loses_recall_on_this_data() -> None:
    rows, _ = run_benchmark(unit_vectors(2000, seed=1), unit_vectors(50, seed=2))
    recalls = [row.recall for row in rows[1:]]
    assert recalls == sorted(recalls)
    assert recalls[-1] > recalls[0]


def test_k_is_limited_by_the_corpus_size() -> None:
    _, k = run_benchmark(unit_vectors(4, seed=1), unit_vectors(2, seed=2))
    assert k == 4


ROWS = [
    BenchRow(name="Flat", ef_search=None, recall=1.0, latency_ms=0.25, build_ms=1.5),
    BenchRow(name="HNSW", ef_search=10, recall=0.9, latency_ms=0.01, build_ms=200.0),
    BenchRow(name="HNSW", ef_search=100, recall=0.99, latency_ms=0.05, build_ms=200.0),
]


def test_table_shows_every_configuration() -> None:
    table = format_table(ROWS, 10)
    assert "| Index | efSearch | Recall@10 vs flat |" in table
    assert "| Flat | - | 1.000 | 0.2500 | 1.50 |" in table
    assert "| HNSW | 100 | 0.990 | 0.0500 | 200.00 |" in table


def test_chart_is_written_as_a_png_and_creates_its_folder(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "benchmark.png"
    save_chart(ROWS, 10, path)
    assert path.read_bytes().startswith(PNG_SIGNATURE)
