"""All settings in one place, overridable with FINE_PRINT_* environment variables or .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type Strategy = Literal["fixed", "section"]

STRATEGIES: tuple[Strategy, ...] = ("fixed", "section")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Defaults work out of the box on a local machine."""

    model_config = SettingsConfigDict(env_prefix="FINE_PRINT_", env_file=".env", extra="ignore")

    # Paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    external_dir: Path = PROJECT_ROOT / "data" / "external"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    questions_path: Path = PROJECT_ROOT / "eval" / "questions.jsonl"
    out_of_scope_path: Path = PROJECT_ROOT / "eval" / "out_of_scope.jsonl"
    # Held-out set: for reporting on a frozen configuration, and never for tuning.
    holdout_questions_path: Path = PROJECT_ROOT / "eval" / "holdout" / "questions.jsonl"
    holdout_out_of_scope_path: Path = PROJECT_ROOT / "eval" / "holdout" / "out_of_scope.jsonl"
    docs_dir: Path = PROJECT_ROOT / "docs"

    # Chunking. Sizes are in characters: 1000 characters is about 250 tokens, well below
    # the 512 tokens the embedding model reads before it silently truncates.
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    default_strategy: Strategy = "section"

    # Embeddings. The prefix comes from the BGE model card; passages take no prefix.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = Field(default=32, gt=0)
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # HNSW index parameters; store.py explains what each one does.
    hnsw_max_neighbors: int = Field(default=16, gt=1)
    hnsw_ef_construction: int = Field(default=200, gt=0)
    hnsw_ef_search: int = Field(default=100, gt=0)

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    # Set from `make eval`: every answerable question scores above 0.69, and questions on
    # unrelated topics stay below 0.55.
    min_similarity: float = Field(default=0.6, ge=-1, le=1)

    # Evaluation
    eval_k_values: tuple[int, ...] = (1, 3, 5, 10)

    # Benchmark
    bench_k: int = Field(default=10, gt=0)
    bench_ef_search_values: tuple[int, ...] = (10, 20, 50, 100, 200)
    bench_latency_repeats: int = Field(default=20, gt=0)

    # LLM
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_connect_timeout: float = Field(default=5.0, gt=0)
    ollama_read_timeout: float = Field(default=180.0, gt=0)
    ollama_max_attempts: int = Field(default=3, gt=0)
    # Ollama silently drops prompt text beyond this many tokens.
    ollama_num_ctx: int = Field(default=4096, gt=0)
    # Small models occasionally repeat themselves until they time out, so replies are capped.
    # Normal answers on this corpus stay under 400 tokens.
    ollama_num_predict: int = Field(default=512, gt=0)

    # Services
    api_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment only once."""
    return Settings()
