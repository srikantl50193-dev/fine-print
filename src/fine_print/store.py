"""Build and query the Chroma vector store. Each chunking strategy gets a collection."""

from dataclasses import dataclass

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from fine_print import config
from fine_print.chunking import Chunk

WRITE_BATCH_SIZE = 1000


class IndexNotBuiltError(RuntimeError):
    """Raised when a query needs a collection that has not been built yet."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    company: str
    doc_type: str
    section: str
    chunk_id: str
    similarity: float

    @property
    def label(self) -> str:
        """The citation form that the prompt asks the model to use."""
        return f"[{self.company} - {self.section}]"


def get_client() -> ClientAPI:
    """Open the on-disk database, with Chroma's usage telemetry switched off."""
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def collection_name(strategy: str) -> str:
    return f"{config.COLLECTION_PREFIX}_{strategy}"


def build_collection(chunks: list[Chunk], embeddings: np.ndarray, strategy: str) -> Collection:
    """Replace a strategy's collection, so no stale chunks survive a rebuild."""
    client = get_client()
    name = collection_name(strategy)
    if name in [collection.name for collection in client.list_collections()]:
        client.delete_collection(name)

    # embedding_function=None: we always pass our own vectors. Without it, Chroma
    # would set up its default model, which is not the one we use for queries.
    collection = client.create_collection(
        name=name,
        metadata=config.CHROMA_HNSW_METADATA,
        embedding_function=None,
    )
    for start in range(0, len(chunks), WRITE_BATCH_SIZE):
        batch = chunks[start : start + WRITE_BATCH_SIZE]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata() for chunk in batch],
            embeddings=embeddings[start : start + WRITE_BATCH_SIZE],
        )
    return collection


def get_collection(strategy: str) -> Collection:
    """Open a built collection, or explain how to build it."""
    try:
        return get_client().get_collection(collection_name(strategy), embedding_function=None)
    except (NotFoundError, ValueError) as error:
        raise IndexNotBuiltError(
            f"No index found for the '{strategy}' strategy. "
            f"Run: uv run python scripts/build_index.py --strategy {strategy}"
        ) from error


def query(
    query_embedding: np.ndarray,
    strategy: str,
    top_k: int = config.TOP_K,
    company: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top_k nearest chunks, optionally limited to a single company."""
    results = get_collection(strategy).query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"company": company} if company else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            text=text,
            company=str(metadata["company"]),
            doc_type=str(metadata["doc_type"]),
            section=str(metadata["section"]),
            chunk_id=str(metadata["chunk_id"]),
            similarity=1 - distance,  # Chroma's cosine distance is 1 - similarity.
        )
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def list_companies(strategy: str) -> list[str]:
    """Return the sorted company names in an index, for the UI filter."""
    metadatas = get_collection(strategy).get(include=["metadatas"])["metadatas"]
    return sorted({str(metadata["company"]) for metadata in metadatas})


def get_all_embeddings(strategy: str) -> np.ndarray:
    """Return every stored vector, so the benchmark does not have to re-embed."""
    return np.asarray(get_collection(strategy).get(include=["embeddings"])["embeddings"])
