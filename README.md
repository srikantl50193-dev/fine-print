# fine-print

Ask plain-English questions about the Terms of Service and privacy policies of popular apps, and get answers drawn only from the policy text, with a citation for each claim. It runs entirely on your own machine, with no API keys and no cost.

It's built from the individual parts, without LangChain or LlamaIndex, so every step is visible: parsing, chunking, embedding, vector search and prompting. It also measures itself, including where it falls short.

The corpus is 20 documents from 10 companies: Coursera, Discord, Lyft, Netflix, PayPal, Pinterest, Reddit, Spotify, Strava and Zoom. Answers can be wrong and aren't legal advice, so check the cited text.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), `make` and [Ollama](https://ollama.com/). uv installs the right Python version for you.

```bash
ollama pull llama3.2   # the local model, about 2 GB
make setup             # create .venv from uv.lock
make data              # download the policy documents into data/raw/
make index             # build the vector index; the first run downloads the embedding model
```

Then ask from the command line:

```bash
uv run fine-print ask "Can I get a refund?" --company spotify
uv run fine-print ask "Can I get a refund?" --company spotify --retrieve-only   # skips the model
```

Or use the web UI, which talks to the API, so it needs both running in separate terminals:

```bash
make api   # http://127.0.0.1:8000, with interactive docs at /docs
make app   # http://localhost:8501
```

## Commands

| Command | What it does |
| --- | --- |
| `make setup` | Create or repair `.venv` from `uv.lock` |
| `make data` | Download the policy documents |
| `make index` | Build the vector index for both chunking strategies |
| `make eval` | Measure retrieval quality and the refusal threshold |
| `make bench` | Compare vector index types, and save `docs/benchmark.png` |
| `make api` | Start the API on `127.0.0.1:8000` |
| `make app` | Start the web UI on `localhost:8501` |
| `make lint` | Check linting and formatting |
| `make format` | Apply lint fixes and formatting |
| `make typecheck` | Run the type checker |
| `make test` | Run the tests with coverage; they need no model, index or Ollama |

`uv run fine-print eval --answers` also asks the model every evaluation question. Every setting can be overridden with a `FINE_PRINT_*` environment variable or a `.env` file; see [.env.example](.env.example).

## Environment

`.venv` is managed by uv, so use `uv run` or `.venv/bin/...` to run things. Add dependencies with `uv add` or `uv add --dev`, not `pip`. To start again from scratch, run `rm -rf .venv && make setup`.

## How it works

1. **Ingest** normalizes each policy to Markdown with `#` headings.
2. **Chunk** cuts it into passages of up to 1,000 characters, in one of two ways. `fixed` slides a window over the whole document. `section` never crosses a heading.
3. **Embed** turns each passage into a vector with `BAAI/bge-small-en-v1.5`.
4. **Store** keeps the vectors in a Chroma HNSW index on disk.
5. **Retrieve** finds the closest passages to a question, optionally filtered by company and document type.
6. **Refuse or answer.** If no passage scores at least 0.6, the answer is "Not found in the documents" and the model isn't called. Otherwise `llama3.2` answers from those passages only, and must cite each claim.
7. **Check citations.** Only labels of passages the model was actually shown count as citations. An answer with none is flagged as unverified.

Passages are third-party text, so the prompt tells the model to ignore any instructions inside them.

## Results

> **Read this first.** These come from 31 answerable and 15 unanswerable questions written for this project, so they're a small sample, and the 0.6 threshold was chosen on the same questions. A passage counts as relevant if it comes from the right document and contains an expected phrase. That test favours longer chunks and can't tell whether a passage truly answers the question. Measured on an Apple M1 Pro.

### Retrieval (`make eval`)

| Strategy | Chunks | Mean chars | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 1367 | 989 | 0.52 | 0.71 | 0.74 | 0.81 | 0.614 |
| section | 1581 | 792 | 0.58 | 0.65 | 0.65 | 0.68 | 0.618 |

`section` is the default. It ranks slightly better at first place, but `fixed` finds more of the answers lower down, including at the top 5 passages the model actually sees. Fixed chunks are about 25% longer, which this test favours, so this doesn't settle which is better.

### Refusals and answers (`fine-print eval --answers`)

| Measure | Result |
| --- | --- |
| Answerable questions that pass the 0.6 threshold | 100% |
| Unanswerable questions the threshold refuses | 53% |
| Unanswerable questions refused once the model also has a say | 100% |
| Answerable questions the model answers | 81% |
| ...with a valid citation | 68% |
| ...citing a passage that contains the expected text | 48% |

The threshold catches off-topic questions, but not ones that name a company, such as "How many vacation days do Zoom employees get?". Those score as high as real questions, so the model has to decline them, and on this set it did.

The weak number is the last one. Part of the gap is retrieval, because `section` has the expected passage in its top 5 only 65% of the time. The rest is the model: `llama3.2` wrongly refused 6 answerable questions, and it can answer from the wrong sentence of a passage. A citation also only shows that the model named a passage, not that the passage supports the claim.

### Vector index (`make bench`)

Run on the 1,581 `section` vectors, using the 31 answerable questions as queries, on one thread.

| Index | efSearch | Recall@10 vs flat | Query latency (ms) | Build time (ms) |
| --- | --- | --- | --- | --- |
| Flat | - | 1.000 | 0.0515 | 3.27 |
| HNSW | 10 | 0.939 | 0.0093 | 144.96 |
| HNSW | 20 | 0.977 | 0.0136 | 144.96 |
| HNSW | 50 | 0.997 | 0.0270 | 144.96 |
| HNSW | 100 | 1.000 | 0.0514 | 144.96 |
| HNSW | 200 | 1.000 | 0.1045 | 144.96 |

![Recall against query latency for each efSearch value, with flat search reaching full recall at about the same latency as HNSW at efSearch 100](docs/benchmark.png)

At this size, exact search is perfectly usable: every option is far below a millisecond, while the model takes seconds. The Chroma index uses `efSearch` 100, which here matches exact search on both recall and speed. HNSW's advantage should grow with a much larger corpus.

## Limitations

- **Small, local model.** `llama3.2` sometimes refuses when the answer is in front of it, and once repeated itself until the 512-token reply cap cut it off.
- **Keyword relevance.** The evaluation can't see whether a passage really answers a question.
- **Unverified citations.** Nothing checks that a cited passage supports the claim.
- **Replaced services.** The source archives have no usable documents for Uber and Duolingo, and no real privacy policy for Airbnb, so Lyft, Coursera and PayPal are in the corpus instead.
- **Older Strava text.** Strava comes from a 2024-07-15 snapshot, while the others are current to the download.
- **Page furniture.** Strava's documents and Zoom's terms begin with site navigation text, which ingestion doesn't remove from Markdown files.
- **Repeated text.** Reddit's terms contain two regional versions of the agreement, so results can show near-identical passages.

## Privacy

Everything runs locally. Chroma's telemetry is off, `make app` turns off Streamlit's, and both services listen only on localhost. The embedding model is downloaded from Hugging Face on first use; once it's cached, it loads without contacting them.

## Stack

Python 3.14, uv, BeautifulSoup, pypdf, sentence-transformers, Chroma, FAISS (benchmark only), Ollama, FastAPI, Streamlit, Typer, structlog, pytest, Ruff and ty.
