"""Turn text into normalized vectors, using a local sentence-transformers model."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from fine_print import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once. The first run downloads it from Hugging Face."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Embed document chunks in batches, returning an array of shape (n, dim).

    Vectors are normalized to length 1, which makes a dot product equal to
    cosine similarity.
    """
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_query(question: str) -> np.ndarray:
    """Embed one question, adding the query prefix that BGE models expect."""
    return embed_texts([config.BGE_QUERY_PREFIX + question])[0]
