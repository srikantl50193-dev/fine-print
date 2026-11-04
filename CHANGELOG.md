# Changelog

## v3.0: answer quality (2026-11-04)

Generation was the bottleneck, so v3 compared models and prompts while leaving retrieval alone.

### Changed

- The default model is now `gemma4:e4b` and the default prompt is `numbered`. Run `ollama pull gemma4:e4b` to upgrade. It needs 6-7 GB of memory, up from about 2.3 GB for `llama3.2`.

### Added

- Prompts as version-controlled files, and a per-run choice of model and prompt.
- A comparison of 4 models and 5 prompts on the tuning set, picked by a rule fixed before any results.
- A classification of each failure: not retrieved, retrieved but ignored, cited wrong passage, or format error.
- Tools for reviewing replies side by side, and a README report of a reply review done by Claude, not an independent human.
- A screenshot, this changelog, data attribution and a v4 backlog.

### Results on the held-out set

- Correct citation rose from 45% to 85%, with Recall@5 unchanged at 85%.
- Wrong refusals fell from 6 of 20 to 2 of 20. All unanswerable questions were still refused.
- p95 latency was 10.5s.
- The set helped prompt this work and has now been scored twice, so it needs replacing.

## v2: rebuild on real policies (2026-11-03)

v2 replaced the prototype with a tested pipeline over real documents.

### Added

- A corpus of 20 real documents from 10 companies, with recorded provenance.
- A Typer CLI, a FastAPI service, and a Streamlit UI that talks to the API.
- A held-out question set, kept separate from the tuning set, and a chunking comparison at matched chunk length.
- A refusal threshold of 0.6 and a 512-token reply cap, both based on measured behaviour.
- A FAISS benchmark with a synthetic scaling test.
- CI with Ruff, ty and pytest, with the environment managed by uv.

### Results on the held-out set

- Recall@5 was 85% and correct citation 45%.
- All unanswerable questions were refused, with 6 of 20 answerable questions wrongly refused.

## v1: prototype (tag `v1-simple`)

A single-script RAG prototype that showed the pipeline and its measurements work.

- Ingestion, `fixed` and `section` chunking, BGE embeddings, a Chroma index, and answers from `llama3.2` through Ollama.
- A Streamlit UI, recall@k and MRR evaluation, and a FAISS benchmark.
- It ran only on two fictional documents and 10 questions by the same author, so its scores say little about real policies.
