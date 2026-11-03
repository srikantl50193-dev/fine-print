import sys
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from fine_print import embed
from fine_print.config import get_settings
from fine_print.models import Chunk


class FakeModel:
    """Records what it is asked to encode, so the tests need no model download."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.kwargs: dict[str, object] = {}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.texts = texts
        self.kwargs = kwargs
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    fake = FakeModel()
    monkeypatch.setattr(embed, "get_model", lambda: fake)
    return fake


def make_chunk() -> Chunk:
    return Chunk(
        text="We keep it.",
        company="acme",
        doc_type="terms",
        section="Retention",
        chunk_id="id",
        strategy="section",
        start_char=0,
        end_char=11,
    )


@pytest.fixture
def clear_model_cache() -> Iterator[None]:
    embed.get_model.cache_clear()
    yield
    embed.get_model.cache_clear()


def fake_library(cached: bool, calls: list[dict[str, object]]) -> SimpleNamespace:
    def load(name: str, **kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only") and not cached:
            raise OSError("not in the cache")
        return name

    return SimpleNamespace(SentenceTransformer=load)


@pytest.mark.usefixtures("clear_model_cache")
def test_downloaded_model_loads_without_contacting_hugging_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_library(True, calls))

    assert embed.get_model() == get_settings().embedding_model
    assert calls == [{"local_files_only": True}]


@pytest.mark.usefixtures("clear_model_cache")
def test_missing_model_is_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_library(False, calls))

    assert embed.get_model() == get_settings().embedding_model
    assert calls == [{"local_files_only": True}, {}]


def test_chunks_are_embedded_with_their_section_title_and_no_prefix(model: FakeModel) -> None:
    vectors = embed.embed_chunks([make_chunk()])
    assert vectors.shape == (1, 4)
    assert model.texts == ["Retention\nWe keep it."]


def test_query_takes_the_bge_prefix_and_returns_one_vector(model: FakeModel) -> None:
    vector = embed.embed_query("Do you keep it?")
    assert vector.shape == (4,)
    assert model.texts == [get_settings().query_prefix + "Do you keep it?"]


def test_vectors_are_normalized_and_batched_from_settings(model: FakeModel) -> None:
    embed.embed_query("Anything?")
    assert model.kwargs["normalize_embeddings"] is True
    assert model.kwargs["batch_size"] == get_settings().embedding_batch_size
