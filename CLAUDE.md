# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

fine-print is a local, citation-first RAG system over the Terms of Service and privacy policies of popular apps. It runs entirely on the user's machine (sentence-transformers, Chroma, Ollama) with no API keys, and it is built from individual parts rather than LangChain or LlamaIndex so that every step stays visible.

## Status: v2 rebuild in progress

The `v2-rebuild` branch replaces the v1 prototype with a fresh scaffold.

- **Done:** settings, domain models, logging, the document download, ingestion (`ingest.py`), chunking (`chunking.py`), and the CLI with a working `fine-print ingest`.
- **Left:** embedding, the vector store, retrieval and the Ollama answer step, evaluation, the index benchmark, the Streamlit app and FastAPI service. `app/`, `docs/` and `eval/` are empty.
- `fine-print index`, `ask`, `eval` and `bench` exit with code 1 and a "not implemented yet" message. Replace each stub in `cli.py` as its stage is built.
- `config.py` comments and paths refer to modules and files that are still to come (`store.py`, `eval/questions.jsonl`, `eval/privacyqa_*`).
- The v1 implementation is the reference for the remaining pieces. Read it with `git show v1-simple:src/fine_print/<module>.py`. v1's config and layout differ from v2 (its chunk sizes are in words, for example), so port ideas rather than copying files.

## Commands

Uses uv, with Python pinned to 3.14. There is no Makefile in v2.

```bash
uv sync                                  # install dependencies
uv run ruff check .                      # lint
uv run ruff format .                     # format
uv run ty check                          # type check (src, tests, scripts, app)
uv run pytest                            # all tests
uv run pytest tests/test_x.py::test_y    # a single test
uv run pytest --cov --cov-report=term-missing   # tests with coverage, as CI runs them
uv run python scripts/download_data.py   # fetch real policies into data/raw/
uv run fine-print ingest                 # parse and chunk data/raw/, and report the counts
uv run pre-commit install                # enable the git hooks
```

CI runs `uv sync --locked`, `ruff check`, `ruff format --check`, `ty check` and the coverage test command, so update `uv.lock` along with any dependency change.

## Architecture

The pipeline: policy files → `Document` (text normalized to Markdown with `#` headings) → `Chunk` (`fixed` or `section` strategy) → BGE embeddings → Chroma HNSW index → retrieval → Ollama prompt → `Answer` with `Citation`s. The stages up to `Chunk` exist.

- **Headings.** The raw files mix ATX, setext, bold-only lines and escaped numbering (`1\. Title`). `normalize_headings` rewrites them all as `#` lines, and `chunking.py` relies only on those. Title Case detection applies to plain text and PDFs only, because in Markdown it picks up navigation links.
- **Chunk windows.** Spans break at whitespace, so consecutive chunks share between `overlap` and `2 * overlap` characters rather than exactly `overlap`. `section` chunks hold body text only and never cross a section; `fixed` chunks can, and take the title of the section contributing the most characters.

Contracts fixed by `models.py` and `config.py`, which later layers must respect:

- **Domain models are immutable.** They inherit from `FrozenModel`. `Chunk.start_char:end_char` must locate the chunk in `Document.text`.
- **Embedding asymmetry.** Queries get `settings.query_prefix` and passages get none, per the BGE model card. Passages are embedded through `Chunk.embedding_text()`, which leads with the section title.
- **Citations.** `RetrievedChunk.label` is the exact form the prompt tells the model to cite (`[Company – Section]`). `Citation` holds only the retrieved chunks the answer actually cites, while `Answer.retrieved` holds everything retrieved.
- **Refusals.** The answer text is `NOT_FOUND_MESSAGE`, with `refused=True` and a `refusal_reason` of `low_similarity` (below `min_similarity`) or `model_declined`.
- **Silent truncation limits.** `chunk_size` is in characters and is kept well under the embedding model's 512-token limit. Ollama drops prompt text beyond `ollama_num_ctx`. Check both limits when changing chunk size, `top_k` or the prompt.

## Configuration

All settings live in `Settings` in `src/fine_print/config.py`, overridable with `FINE_PRINT_<FIELD>` environment variables or `.env`. Add new tunables there rather than as module constants.

- `get_settings()` is `lru_cache`d, so tests that change the environment must call `get_settings.cache_clear()`.
- `.env` is resolved relative to the working directory, so run commands from the repo root.

## Data

- **`data/raw/`** holds real policies named `<company>_<doctype>.md`, with provenance in `MANIFEST.json`. It is gitignored: these are third-party documents, and `scripts/download_data.py` reproduces them.
- **`data/samples/`** holds fictional companies (Tunely, Ridely), copied to `data/raw/` only when every download fails. Never present results from them as findings about real services. The HTML sample deliberately includes nav, script and style content that ingestion must strip.
- **`data/external/`** holds gitignored clones of the source repos. `download_data.py` reuses an existing clone without pulling, so delete the clone to get newer documents.
- **Third-party text stays byte-for-byte as downloaded.** Pre-commit's whitespace fixers exclude `data/raw/`, `data/samples/` and `eval/privacyqa_*`; keep those exclusions when adding hooks.
- **Document list.** `DOCUMENTS` in `download_data.py` names each document's exact source, folder and file: 20 documents across 10 companies. Folder names are case-sensitive on Linux but not on macOS, so check them with `ls`. The script overwrites but never deletes, so remove a dropped company's files from `data/raw/` by hand.
- **Absent services.** Neither archive has Uber or Duolingo, and Airbnb's only privacy document is a captured login page. Lyft, Coursera and PayPal replace them. A file's size doesn't prove it is a real policy, so read a new document before adding it.
- **Known quality issues.** Strava comes from the tosdr snapshot of 2024-07-15, so it is older than the rest. Strava and Zoom terms begin with site navigation text, which ingestion doesn't remove from Markdown files.

## Conventions

- Ruff enforces Google-style docstrings on public APIs in `src/` and a maximum of 7 function arguments (`PL` rules are on). Tests and scripts have relaxed docstring rules; see `per-file-ignores`.
- Logs go to stderr through structlog (`configure_logging`), which keeps stdout clean for piped CLI output.
- Use `respx` to mock httpx calls such as Ollama in tests.
- Linux resolves torch from the CPU-only PyTorch index (`tool.uv.sources`), so CI and Docker images avoid the CUDA libraries.
- v1 ran Streamlit with `--server.fileWatcherType none` because the file watcher trips over torch internals on reload.
