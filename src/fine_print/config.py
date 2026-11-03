"""All project settings live here, so experiments only ever touch one file."""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "samples"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"

# --- Chunking ----------------------------------------------------------------
# Sizes are counted in words. 150 words is roughly 200 tokens, comfortably below
# the 512 tokens the embedding model reads before it silently truncates.
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30
STRATEGIES = ("fixed", "section")
DEFAULT_STRATEGY = "section"

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32
# BGE models were trained with this prefix on queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Vector store ------------------------------------------------------------
COLLECTION_PREFIX = "fine_print"
CHROMA_HNSW_METADATA = {
    # How "closeness" is measured. On normalized vectors, cosine is the angle
    # between them, and Chroma reports distance = 1 - cosine similarity.
    "hnsw:space": "cosine",
    # Links per node in the graph. More links improve recall but use more memory.
    "hnsw:M": 16,
    # Candidates considered while inserting a node. Higher builds a better graph,
    # but builds more slowly. It has no effect on query time.
    "hnsw:construction_ef": 200,
    # Candidates considered while answering a query. Higher improves recall, but
    # each query is slower. This is the main speed/accuracy dial at query time.
    "hnsw:search_ef": 100,
}

# --- Retrieval and generation ------------------------------------------------
TOP_K = 5
OLLAMA_URL = os.getenv("FINE_PRINT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("FINE_PRINT_OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = 180
# Ollama silently drops prompt text past this many tokens, so it must fit TOP_K chunks.
OLLAMA_NUM_CTX = 4096
NOT_FOUND_MESSAGE = "Not found in the documents"

# --- Evaluation --------------------------------------------------------------
EVAL_K_VALUES = (1, 3, 5, 10)

# --- FAISS benchmark ---------------------------------------------------------
BENCH_K = 10
BENCH_HNSW_M = 16
BENCH_EF_CONSTRUCTION = 200
BENCH_EF_SEARCH_VALUES = (10, 20, 50, 100, 200)
BENCH_LATENCY_REPEATS = 50
