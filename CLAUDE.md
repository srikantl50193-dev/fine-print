# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- `.venv` is managed by uv. Run things with `uv run ...` or `.venv/bin/...`, without relying on an activated shell.
- Add dependencies with `uv add <package>` or `uv add --dev <package>`, which also update `uv.lock`.
- Never use `pip` or `pip3`. The venv has no pip, and on this machine `pip3` resolves to the global Python, so it would install outside the project.
- Recreate the environment with `rm -rf .venv && make setup`. `make setup` also creates it on a fresh clone. It stops with the official install command if uv is missing, and doesn't install uv itself.

## Project

fine-print is a local, citation-first RAG system over the Terms of Service and privacy policies of popular apps. It runs entirely on the user's machine (sentence-transformers, Chroma, Ollama) with no API keys, and it is built from individual parts rather than LangChain or LlamaIndex so that every step stays visible.

## Status: v2 rebuild in progress

The `v2-rebuild` branch replaces the v1 prototype, which is kept at the `v1-simple` tag.

- **Done:** every stage: download, ingest, chunk, index, RAG, eval, benchmark, API and UI.
- **Undecided:** `default_strategy` stays `section` while the fair chunking comparison is reviewed (see the README's results).

## Evaluation data

**Never tune parameters against `eval/holdout/`.** That means thresholds, chunk sizes, the default strategy, prompts and the model, and it applies to editing questions after seeing their scores too. Tune on `eval/questions.jsonl` and `eval/out_of_scope.jsonl`. Run `make eval-holdout` only to report on a frozen configuration, and don't change anything because of what it shows. Once a held-out result has influenced a change, the set is spent and needs replacing.

## Commands

Python is pinned to 3.14. Every Makefile target goes through uv.

```bash
make setup                               # create or repair .venv from uv.lock
make lint                                # ruff check and ruff format --check
make format                              # apply ruff fixes and formatting
make typecheck                           # ty check (src, tests, scripts, app)
make test                                # all tests with coverage, as CI runs them
uv run pytest tests/test_x.py::test_y    # a single test
make data                                # fetch real policies into data/raw/
uv run fine-print ingest                 # parse and chunk data/raw/, and report the counts
uv run pre-commit install                # enable the git hooks
```

`make index`, `eval`, `bench`, `api` and `app` run the later stages. `make app` needs `make api` running, and answering needs Ollama. `uv run fine-print eval --answers` also measures the real model, which takes about a minute.

Tests need no model, Ollama or `data/raw/`: Ollama and the API are mocked with `respx`, the embedding model is faked, and Chroma runs for real in a temporary folder. Because mocks can't show how the real model behaves, also run real questions after changing `rag.py`.

CI runs `uv sync --locked`, `ruff check`, `ruff format --check`, `ty check` and the coverage test command, so update `uv.lock` along with any dependency change.

## Architecture

The pipeline: policy files → `Document` (text normalized to Markdown with `#` headings) → `Chunk` (`fixed` or `section` strategy) → BGE embeddings → Chroma HNSW index → retrieval → Ollama prompt → `Answer` with `Citation`s. The Streamlit app is a client of the FastAPI service, and doesn't import the pipeline.

- **Lazy imports.** `cli.py` imports the embedding, store and benchmark modules inside its commands, because torch makes them slow to load. Don't move them to the top of the file.
- **Tuned from `make eval`.** `min_similarity` is 0.6 because answerable questions score above 0.69 and off-topic ones below 0.55. Questions that name a company score as high as real ones, so only the model can refuse those. Re-measure if the corpus or embedding model changes.
- **Reply cap.** `ollama_num_predict` exists because `llama3.2` can repeat itself until the read timeout. Keep it below `RESERVED_TOKENS` in `rag.py`.

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
- **Third-party text stays byte-for-byte as downloaded.** Pre-commit's whitespace fixers exclude `data/raw/` and `data/samples/`; keep those exclusions when adding hooks.
- **Document list.** `DOCUMENTS` in `download_data.py` names each document's exact source, folder and file: 20 documents across 10 companies. Folder names are case-sensitive on Linux but not on macOS, so check them with `ls`. The script overwrites but never deletes, so remove a dropped company's files from `data/raw/` by hand.
- **Absent services.** Neither archive has Uber or Duolingo, and Airbnb's only privacy document is a captured login page. Lyft, Coursera and PayPal replace them. A file's size doesn't prove it is a real policy, so read a new document before adding it.
- **Known quality issues.** Strava comes from the tosdr snapshot of 2024-07-15, so it is older than the rest. Strava and Zoom terms begin with site navigation text, which ingestion doesn't remove from Markdown files.

## Conventions

- Ruff enforces Google-style docstrings on public APIs in `src/` and a maximum of 7 function arguments (`PL` rules are on). Tests and scripts have relaxed docstring rules; see `per-file-ignores`.
- Logs go to stderr through structlog (`configure_logging`), which keeps stdout clean for piped CLI output.
- Use `respx` to mock httpx calls such as Ollama in tests.
- Linux resolves torch from the CPU-only PyTorch index (`tool.uv.sources`), so CI and Docker images avoid the CUDA libraries.
- v1 ran Streamlit with `--server.fileWatcherType none` because the file watcher trips over torch internals on reload.
