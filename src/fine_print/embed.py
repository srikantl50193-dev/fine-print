"""Turn text into normalized vectors, using a local sentence-transformers model."""

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from fine_print.config import get_settings
from fine_print.models import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once, contacting Hugging Face only if it isn't downloaded yet."""
    # Imported here because it loads torch, which would slow down every CLI command.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    name = get_settings().embedding_model
    try:
        return SentenceTransformer(name, local_files_only=True)
    except OSError:
        return SentenceTransformer(name)


def _encode(texts: Sequence[str]) -> np.ndarray:
    """Embed texts as an array of shape (n, dim), with every vector at length 1.

    Length-1 vectors make a dot product equal to cosine similarity.
    """
    return np.asarray(
        get_model().encode(
            list(texts),
            batch_size=get_settings().embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )


def embed_chunks(chunks: Sequence[Chunk]) -> np.ndarray:
    """Embed chunks for indexing. Passages take no prefix, per the BGE model card."""
    return _encode([chunk.embedding_text() for chunk in chunks])


def embed_query(question: str) -> np.ndarray:
    """Embed one question, adding the query prefix that BGE models expect."""
    return _encode([get_settings().query_prefix + question])[0]
