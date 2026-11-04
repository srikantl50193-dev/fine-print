# fine-print

Ask plain-English questions about the Terms of Service and privacy policies of popular apps, and get answers drawn only from the policy text, with a citation for each claim. It runs entirely on your own machine, with no API keys and no cost.

It's built from the individual parts, without LangChain or LlamaIndex, so every step is visible: parsing, chunking, embedding, vector search and prompting. It also measures itself, including where it falls short.

The corpus is 20 documents from 10 companies: Coursera, Discord, Lyft, Netflix, PayPal, Pinterest, Reddit, Spotify, Strava and Zoom. Answers can be wrong and aren't legal advice, so check the cited text.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), `make` and [Ollama](https://ollama.com/). uv installs the right Python version for you.

```bash
ollama pull gemma4:e4b   # the local model, a 6.6 GB download that needs 6-7 GB of memory
make setup               # create .venv from uv.lock
make data                # download the policy documents into data/raw/
make index               # build the vector index; the first run downloads the embedding model
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
| `make eval` | Measure retrieval quality and the refusal threshold on the tuning set |
| `make eval-holdout` | Report on the held-out set, including the model. Never tune from this |
| `make compare` | Compare chunking strategies at matched chunk length, on the tuning set |
| `make compare-answers` | Compare models and prompts on the tuning set; needs each model pulled |
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
6. **Refuse or answer.** If no passage scores at least 0.6, the answer is "Not found in the documents" and the model isn't called. Otherwise the model (`gemma4:e4b` by default) answers from those passages only, and must cite each claim.
7. **Check citations.** Only labels of passages the model was actually shown count as citations. An answer with none is flagged as unverified.

Passages are third-party text, so the prompt tells the model to ignore any instructions inside them.

## Results

> **Read this first.** There are two question sets, reported separately:
>
> - **Tuning set** (31 answerable, 15 unanswerable): the 0.6 refusal threshold was tuned on these questions **only**, so its scores here are optimistic.
> - **Held-out set** (20 answerable, 10 unanswerable, in `eval/holdout/`): written after the v2 settings were fixed, and covers only the default strategy. It has been scored twice, once for each frozen configuration, v2 and v3. No setting was tuned on its scores, but its weak v2 answer score helped prompt the v3 work, so **the set is no longer clean** and should be replaced before it's used again.
>
> Both are small and were written for this project by the same author, so one answerable question moves a score by 3–5 points. A passage counts as relevant if it comes from the right document and contains an expected phrase. That test favours longer chunks and can't tell whether a passage truly answers the question. Measured on an Apple M1 Pro.

### Retrieval

| Question set | Strategy | Chunks | Mean chars | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuning | section | 1581 | 792 | 0.58 | 0.65 | 0.65 | 0.68 | 0.618 |
| Held-out | section | 1581 | 792 | 0.60 | 0.75 | 0.85 | 0.90 | 0.692 |

Held-out retrieval scores higher than tuning. That isn't evidence the system improved, since no setting was tuned for retrieval and the two sets ask about different facts. It shows how much these scores vary with the questions chosen.

### Refusals and answers

v2 is `llama3.2` with the `baseline` prompt. v3, the current default, is `gemma4:e4b` with the `numbered` prompt.

| Measure | Tuning, v2 | Held-out, v2 | Tuning, v3 | Held-out, v3 |
| --- | --- | --- | --- | --- |
| Answerable questions that pass the 0.6 threshold | 100% | 100% | 100% | 100% |
| Unanswerable questions the threshold refuses | 53% | 50% | 53% | 50% |
| Unanswerable questions refused once the model also has a say | 100% | 100% | 100% | 100% |
| Answerable questions the model answers | 81% | 70% | 97% | 90% |
| ...with a valid citation | 68% | 55% | 90% | 85% |
| ...citing a passage that contains the expected text | 48% | 45% | 65% | 85% |
| Answerable questions wrongly refused | 6 of 31 | 6 of 20 | 1 of 31 | 2 of 20 |
| p95 latency when the model is called | 10.3s | not measured | 8.4s | 10.5s |

The threshold, the one setting tuned on these questions, holds up on the held-out set. It catches off-topic questions, but not ones that name a company, such as "How many vacation days do Zoom employees get?". Those score as high as real questions, so the model has to decline them, and on both sets it did.

In v3, every failure on both sets is classed as `not_retrieved`: the correct-citation scores of 65% and 85% equal Recall@5 on each set, so the model cited the expected section every time the keyword test found it in the passages shown. That makes retrieval and the keyword test the limit, though the review below shows the test is sometimes wrong.

Read the held-out jump from 45% to 85% with care. It's one set of 20 questions, where each question moves the score by 5 points, and it isn't a clean estimate because the v2 score on this set helped motivate the change. Scoring higher than tuning doesn't show the system generalizes well either, since Recall@5 is higher on this set too. "Answered" is overstated too: a paraphrased refusal such as "The provided documents do not specify..." counts as an answer, which affects 2 of the 31 tuning questions in v3 and 1 of the 20 held-out ones.

### Choosing the model and prompt (`make compare-answers`, tuning set only)

The rule was fixed before any results: the highest correct-citation rate, among runs that refuse at least 90% of unanswerable questions with p95 latency under 30 seconds, with ties going to the lower latency. Only `llama3.2` with `quote_first` was ineligible. Three `gemma4:e4b` runs tied at 65%, so latency decided it, and the 0.3s lead over `read_all` could be noise. Prompts live in [src/fine_print/prompts/](src/fine_print/prompts/).

| Model | Prompt | Answered | Correct citation | Wrong refusals | Unanswerable refused | p95 (s) | Not retrieved | Retrieved, ignored | Cited wrong passage | Format error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **gemma4:e4b** | **numbered** | 97% | 65% | 1 | 100% | 8.4 | 11 | 0 | 0 | 0 |
| gemma4:e4b | read_all | 97% | 65% | 1 | 100% | 8.7 | 11 | 0 | 0 | 0 |
| gemma4:e4b | numbered_quote | 90% | 65% | 3 | 93% | 9.9 | 11 | 0 | 0 | 0 |
| gemma4:e4b | baseline | 97% | 61% | 1 | 100% | 8.4 | 11 | 0 | 1 | 0 |
| qwen3:8b | numbered | 90% | 61% | 3 | 100% | 9.2 | 11 | 0 | 1 | 0 |
| qwen3:8b | baseline | 90% | 61% | 3 | 100% | 9.8 | 11 | 0 | 1 | 0 |
| qwen3:8b | read_all | 94% | 61% | 2 | 100% | 10.7 | 11 | 0 | 1 | 0 |
| qwen3:8b | numbered_quote | 97% | 61% | 1 | 93% | 11.7 | 11 | 0 | 1 | 0 |
| gemma4:e4b | quote_first | 90% | 58% | 3 | 93% | 7.5 | 11 | 0 | 1 | 1 |
| llama3.1:8b | numbered | 87% | 58% | 4 | 100% | 9.3 | 11 | 0 | 1 | 1 |
| llama3.1:8b | read_all | 87% | 58% | 4 | 93% | 13.8 | 11 | 0 | 1 | 1 |
| llama3.2 | read_all | 87% | 55% | 4 | 100% | 4.7 | 11 | 0 | 3 | 0 |
| llama3.1:8b | baseline | 77% | 55% | 7 | 100% | 7.8 | 11 | 0 | 2 | 1 |
| llama3.1:8b | quote_first | 97% | 55% | 1 | 93% | 10.9 | 11 | 0 | 2 | 1 |
| qwen3:8b | quote_first | 97% | 55% | 1 | 93% | 13.1 | 11 | 0 | 2 | 1 |
| llama3.2 | numbered_quote | 97% | 52% | 1 | 100% | 5.3 | 11 | 0 | 3 | 1 |
| llama3.1:8b | numbered_quote | 100% | 52% | 0 | 93% | 9.4 | 11 | 0 | 3 | 1 |
| llama3.2 | baseline | 81% | 48% | 6 | 100% | 10.3 | 11 | 1 | 2 | 2 |
| llama3.2 | numbered | 77% | 45% | 7 | 100% | 3.3 | 11 | 1 | 3 | 2 |
| llama3.2 | quote_first | 100% | 45% | 0 | 87% | 5.2 | 11 | 0 | 1 | 5 |

The last four columns classify each answerable question that didn't get a correct citation: the expected text wasn't in the passages shown, the model refused despite seeing it, it cited a different passage, or it answered without a usable citation.

All 20 runs have the same 11 `not_retrieved` failures, because retrieval doesn't change between them. The differences between models are in the other three columns, where `gemma4:e4b` with `numbered` has none. At least 4 of those 11 are errors in the keyword test rather than retrieval misses, as the review below shows.

| Model | Download | Memory when loaded | Load time | Speed (tokens/s) |
| --- | --- | --- | --- | --- |
| `llama3.2` | 2.0 GB | 2.3 GB | 3s | 51 |
| `llama3.1:8b` | 4.9 GB | 5.0-5.2 GB | 8s | 23 |
| `qwen3:8b` | 5.2 GB | 5.1-5.9 GB | 8s | 22 |
| `gemma4:e4b` | 6.6 GB | 5.8-7.3 GB | 8-13s | 23 |

Memory ranges cover Ollama's process memory and macOS wired memory, on a 16 GB machine. `/api/ps` reports 0.26 GB for `gemma4:e4b`, so it isn't used. The new default needs about 3 times the memory of `llama3.2` and generates at under half the speed. Its p95 beats `llama3.2` with `baseline`, but that isn't a speed advantage for the model, because `llama3.2` reaches 3.3s with `numbered`. On a machine with 8 GB, set `FINE_PRINT_OLLAMA_MODEL=llama3.2` and `FINE_PRINT_PROMPT_NAME=read_all`, its best prompt here at 55%.

### Reply review

**The review of replies was done by Claude, not an independent human.** Claude also wrote the question sets and prompts, so this is a useful check on the automated score, but not independent evidence. A human review was planned and then skipped, so [eval/review/v3_finalists.md](eval/review/v3_finalists.md) has no marks. Claude read the 46 tuning-set replies from both finalists against the cited passages, after the pick was made, and the pick didn't change. Held-out replies weren't reviewed.

The automated verdict agrees with Claude's reading on **41 of 46** replies for each finalist. These are the wrong and arguable verdicts, which apply to both:

| Question | Topic | Automated | Claude's reading | Why |
| --- | --- | --- | --- | --- |
| Q3 | Spotify group lawsuit | Fail | Good answer | "class action" appears only in the section title, which isn't scored |
| Q12 | Discord selling data | Fail | Good answer | The policy says "selling", but the expected phrase is "sell" |
| Q29 | Strava account deletion | Fail | Good answer | The policy says "account deletion request", but the expected phrase is "request account deletion" |
| Q31 | Pinterest ads from sites you visit | Fail | Good answer | The expected phrase "electric guitars" isn't in the passages shown, but the reply is supported by those that are |
| Q17 | Zoom automatic billing | Pass | Bad answer | The reply says only that terms "apply" in Sections 12 and 14, and never answers. It passes by citing a passage containing "automatic renewals" |
| Q10 | Lyft selling data | Pass | Arguable, counted as good | Accurate, but leaves out that some sharing may count as a "sale" for targeted advertising under some US state laws |
| Q19 | Zoom shutting down a service | Fail | Arguable, counted as bad | Termination for any reason is correct, but the clause that answers this directly, "discontinue the Services", wasn't retrieved |

| `gemma4:e4b` with `numbered` | Automated | Claude's reading |
| --- | --- | --- |
| Answerable questions correct | 20 of 31 (65%) | 23 of 31 (74%), or 22-24 depending on Q10 and Q19 |
| All 46 questions | 35 of 46 (76%) | 38 of 46 (83%) |

The automated score fails in both directions, and it can't be fixed by adding phrases alone. Citing a passage with the expected text doesn't show that the reply answers the question. `read_all` scores the same on this reading. No `numbered` reply mentions an external page the terms point to, including Lyft's Help Center for cancellation fees on Q8.

### Chunking comparison (`make compare`, tuning set only)

The keyword test favours longer chunks, and at the default size limit fixed chunks are about 25% longer than section chunks. So this also scores fixed chunks with their size limit set to match the section mean, keeping the same 150-character overlap.

| Run | Size limit | Chunks | Mean chars | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| section | 1000 | 1581 | 792 | 0.58 | 0.65 | 0.65 | 0.68 | 0.618 |
| fixed, default size | 1000 | 1367 | 989 | 0.52 | 0.71 | 0.74 | 0.81 | 0.614 |
| fixed, matched length | 802 | 1789 | 792 | 0.42 | 0.61 | 0.61 | 0.71 | 0.518 |

`fixed` leads at the top 5 only at its longer default size. At matched length it falls behind `section` everywhere except Recall@10, so most of that lead appears to come from chunk length. `section` remains the default, and the choice is still open. The difference between the matched runs at the top 5 is about one question, so this leans towards `section` without proving it.

### Vector index (`make bench`)

Run on the 1,581 `section` vectors, using the 31 tuning questions as queries, on one thread.

| Index | efSearch | Recall@10 vs flat | Query latency (ms) | Build time (ms) |
| --- | --- | --- | --- | --- |
| Flat | - | 1.000 | 0.0515 | 3.27 |
| HNSW | 10 | 0.939 | 0.0093 | 144.96 |
| HNSW | 20 | 0.977 | 0.0136 | 144.96 |
| HNSW | 50 | 0.997 | 0.0270 | 144.96 |
| HNSW | 100 | 1.000 | 0.0514 | 144.96 |
| HNSW | 200 | 1.000 | 0.1045 | 144.96 |

![Recall against query latency for each efSearch value, with flat search reaching full recall at about the same latency as HNSW at efSearch 100](docs/benchmark.png)

## Design decisions

### Keeping HNSW at `efSearch` 100

At about 1,600 vectors, HNSW gives no speed advantage over exact search. At `efSearch` 100 both take about 0.05 ms per query and both reach full recall, and exact search builds far faster. We keep HNSW at 100 anyway: Chroma's index is HNSW, full recall is the safe setting, and the search time is tiny next to the model's seconds. Lowering `efSearch` to 50 would be twice as fast, but saving 0.02 ms isn't worth any lost recall here.

HNSW starts to matter as the corpus grows, because exact search slows in proportion to the number of vectors while HNSW barely slows:

| Vectors | Flat (ms) | HNSW, `efSearch` 100 (ms) | HNSW recall@10 | HNSW build (s) |
| --- | --- | --- | --- | --- |
| 1,581 (real) | 0.051 | 0.056 | 0.999 | 0.1 |
| 10,000 | 0.343 | 0.100 | 1.000 | 0.3 |
| 100,000 | 3.296 | 0.149 | 1.000 | 5.2 |
| 500,000 | 16.320 | 0.141 | 0.998 | 23.9 |

Only the first row is the real corpus. The larger ones are synthetic, made by resampling our vectors with a little noise, with 200 more made the same way as queries, on one search thread. That data is clustered around real vectors and may be easier than a genuinely larger corpus, so read them as the trend and not as a promise. HNSW overtakes exact search by about 10,000 vectors, roughly 6 times this corpus, or around 60 companies at the current size. Even at 500,000 vectors exact search takes just 16 ms, so for one local user the speed only clearly matters at millions of vectors, or under many simultaneous queries.

## Limitations

- **Local model.** `gemma4:e4b` needs 6-7 GB of memory. Local models can also give incomplete answers: on Q10 above it leaves out a qualifier that matters.
- **Paraphrased refusals.** A reply such as "The provided documents do not specify..." isn't recognized as a refusal, so it counts as answered and isn't marked as a refusal in the UI or API.
- **Keyword relevance.** The evaluation can't see whether a passage really answers a question, and it misses inflections and section titles.
- **Passages that stop short.** Fixed size limits can cut a clause, as on Reddit's licence terms (Q15), and some clauses aren't retrieved at all (Q19).
- **Unverified citations.** Nothing checks that a cited passage supports the claim.
- **Replaced services.** The source archives have no usable documents for Uber and Duolingo, and no real privacy policy for Airbnb, so Lyft, Coursera and PayPal are in the corpus instead.
- **Older Strava text.** Strava comes from a 2024-07-15 snapshot, while the others are current to the download.
- **Page furniture.** Strava's documents and Zoom's terms begin with site navigation text, which ingestion doesn't remove from Markdown files.
- **Repeated text.** Reddit's terms contain two regional versions of the agreement, so results can show near-identical passages.

## Privacy

Everything runs locally. Chroma's telemetry is off, `make app` turns off Streamlit's, and both services listen only on localhost. The embedding model is downloaded from Hugging Face on first use; once it's cached, it loads without contacting them.

## Stack

Python 3.14, uv, BeautifulSoup, pypdf, sentence-transformers, Chroma, FAISS (benchmark only), Ollama, FastAPI, Streamlit, Typer, structlog, pytest, Ruff and ty.
