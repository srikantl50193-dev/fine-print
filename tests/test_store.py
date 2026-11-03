from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from fine_print import store
from fine_print.config import get_settings
from fine_print.models import Chunk


@pytest.fixture(autouse=True)
def chroma_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FINE_PRINT_CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_chunk(position: int, company: str = "acme", doc_type: str = "terms") -> Chunk:
    return Chunk(
        text=f"Text {position}.",
        company=company,
        doc_type=doc_type,
        section=f"Section {position}",
        chunk_id=f"{company}_{doc_type}_section_{position:04d}",
        strategy="section",
        start_char=position * 10,
        end_char=position * 10 + 7,
    )


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def built() -> list[Chunk]:
    chunks = [make_chunk(0), make_chunk(1, "beta"), make_chunk(2, "beta", "privacy")]
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    assert store.build_collection(chunks, vectors, "section") == 3
    return chunks


def test_query_ranks_by_similarity_and_restores_every_field(built: list[Chunk]) -> None:
    results = store.query(unit(1, 0.1, 0), "section", top_k=2)

    assert [r.chunk_id for r in results] == [built[0].chunk_id, built[1].chunk_id]
    assert results[0].similarity == pytest.approx(0.995, abs=0.001)
    assert results[1].similarity == pytest.approx(0.0995, abs=0.001)
    assert results[0].model_dump(exclude={"similarity", "label"}) == built[0].model_dump()


@pytest.mark.usefixtures("built")
@pytest.mark.parametrize(
    ("company", "doc_type", "expected"),
    [
        ("beta", None, 2),
        ("BETA", None, 2),
        (None, "privacy", 1),
        ("beta", "terms", 1),
        ("acme", "privacy", 0),
    ],
)
def test_filters(company: str | None, doc_type: str | None, expected: int) -> None:
    results = store.query(unit(1, 1, 1), "section", 10, company=company, doc_type=doc_type)
    assert len(results) == expected


@pytest.mark.usefixtures("built")
def test_top_k_above_the_collection_size_returns_what_exists() -> None:
    assert len(store.query(unit(1, 1, 1), "section", 50)) == 3


@pytest.mark.usefixtures("built")
def test_rebuild_leaves_no_stale_chunks() -> None:
    store.build_collection([make_chunk(9)], np.stack([unit(1, 0, 0)]), "section")
    results = store.query(unit(1, 1, 1), "section", 10)
    assert [r.section for r in results] == ["Section 9"]


@pytest.mark.usefixtures("built")
def test_strategies_are_stored_separately() -> None:
    assert len(store.query(unit(1, 1, 1), "section", 10)) == 3
    with pytest.raises(store.IndexNotBuiltError, match="make index"):
        store.query(unit(1, 1, 1), "fixed", 10)


@pytest.mark.usefixtures("built")
def test_list_companies_is_sorted_and_unique() -> None:
    assert store.list_companies("section") == ["acme", "beta"]


@pytest.mark.usefixtures("built")
def test_get_all_embeddings_returns_every_vector() -> None:
    vectors = store.get_all_embeddings("section")
    assert vectors.shape == (3, 3)
    assert vectors.dtype == np.float32


def test_missing_index_explains_how_to_build_it() -> None:
    with pytest.raises(store.IndexNotBuiltError, match="make index"):
        store.list_companies("section")


def test_mismatched_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        store.build_collection([make_chunk(0)], np.zeros((2, 3)), "section")


def test_hnsw_settings_reach_the_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_PRINT_HNSW_EF_SEARCH", "77")
    get_settings.cache_clear()
    store.build_collection([make_chunk(0)], np.stack([unit(1, 0, 0)]), "section")

    configuration = store._get_collection("section").configuration
    assert configuration is not None
    hnsw = configuration["hnsw"]
    assert hnsw is not None
    assert hnsw["ef_search"] == 77
    assert hnsw["space"] == "cosine"
