"""Build and query the Chroma vector store. Each chunking strategy gets a collection.

Chroma searches with an HNSW graph, which links each vector to its near neighbours. Three
settings control the graph, and `Settings` holds their values:

- `hnsw_max_neighbors` (often called M) is the links per vector. Raising it improves recall
  and uses more memory.
- `hnsw_ef_construction` is the candidates considered when adding a vector. Raising it makes
  a better graph and a slower build, with no effect on query speed.
- `hnsw_ef_search` is the candidates considered when answering a query. Raising it improves
  recall and slows queries.
"""

import contextlib
from collections.abc import Sequence
from typing import Any

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from fine_print.config import Strategy, get_settings
from fine_print.models import Chunk, RetrievedChunk

COLLECTION_PREFIX = "fine_print"
WRITE_BATCH_SIZE = 1000


class IndexNotBuiltError(RuntimeError):
    """Raised when a query needs a collection that has not been built yet."""


def _client() -> ClientAPI:
    """Open the on-disk database, with Chroma's usage telemetry switched off."""
    return chromadb.PersistentClient(
        path=str(get_settings().chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def collection_name(strategy: Strategy) -> str:
    """Return the name of a chunking strategy's collection."""
    return f"{COLLECTION_PREFIX}_{strategy}"


def build_collection(chunks: Sequence[Chunk], embeddings: np.ndarray, strategy: Strategy) -> int:
    """Replace a strategy's collection, so no stale chunks survive a rebuild.

    Args:
        chunks: The chunks to store.
        embeddings: Their vectors, in the same order.
        strategy: The chunking strategy that produced them.

    Returns:
        The number of chunks now in the collection.

    Raises:
        ValueError: If there is not exactly one vector per chunk.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} vectors")

    settings = get_settings()
    client = _client()
    name = collection_name(strategy)
    with contextlib.suppress(NotFoundError):  # A first build has nothing to replace.
        client.delete_collection(name)

    # embedding_function=None: we always pass our own vectors. Without it, Chroma would
    # set up its default model, which is not the one we use for queries.
    collection = client.create_collection(
        name=name,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine",
                "max_neighbors": settings.hnsw_max_neighbors,
                "ef_construction": settings.hnsw_ef_construction,
                "ef_search": settings.hnsw_ef_search,
            }
        },
    )
    for start in range(0, len(chunks), WRITE_BATCH_SIZE):
        batch = chunks[start : start + WRITE_BATCH_SIZE]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata() for chunk in batch],
            embeddings=embeddings[start : start + WRITE_BATCH_SIZE],
        )
    return collection.count()


def _get_collection(strategy: Strategy) -> Collection:
    """Open a built collection, or explain how to build it."""
    try:
        return _client().get_collection(collection_name(strategy), embedding_function=None)
    except NotFoundError as error:
        raise IndexNotBuiltError(
            f"No index found for the '{strategy}' strategy. Run: make index"
        ) from error


def _where(company: str | None, doc_type: str | None) -> dict[str, Any] | None:
    """Build a metadata filter from the optional company and document type."""
    conditions: list[dict[str, Any]] = []
    if company:
        conditions.append({"company": company.lower()})
    if doc_type:
        conditions.append({"doc_type": doc_type.lower()})
    if len(conditions) > 1:
        return {"$and": conditions}
    return conditions[0] if conditions else None


def query(
    query_embedding: np.ndarray,
    strategy: Strategy,
    top_k: int,
    company: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top_k nearest chunks, optionally limited by company and document type.

    Raises:
        IndexNotBuiltError: If the strategy's collection does not exist.
    """
    results = _get_collection(strategy).query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=_where(company, doc_type),
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]
    if documents is None or metadatas is None or distances is None:
        return []
    return [
        # Chroma's cosine distance is 1 - similarity.
        RetrievedChunk.model_validate({**metadata, "text": text, "similarity": 1 - distance})
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0], strict=True)
    ]


def list_companies(strategy: Strategy) -> list[str]:
    """Return the sorted company names in an index, for filters in the UI.

    Raises:
        IndexNotBuiltError: If the strategy's collection does not exist.
    """
    metadatas = _get_collection(strategy).get(include=["metadatas"])["metadatas"] or []
    return sorted({str(metadata["company"]) for metadata in metadatas})


def get_all_embeddings(strategy: Strategy) -> np.ndarray:
    """Return every stored vector, so the benchmark does not have to re-embed.

    Raises:
        IndexNotBuiltError: If the strategy's collection does not exist.
    """
    embeddings = _get_collection(strategy).get(include=["embeddings"])["embeddings"]
    return np.asarray(embeddings, dtype=np.float32)
