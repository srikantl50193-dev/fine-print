# fine-print

Ask plain-English questions about the Terms of Service and privacy policies of popular apps, and get answers drawn only from the policy text, with a citation for each claim. It runs entirely on your own machine, with no API keys.

> **Status:** v2 is being rebuilt. Downloading and chunking documents work today; `make index`, `eval`, `bench`, `api` and `app` depend on stages that are still being built.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and `make`. uv installs the right Python version for you.

```bash
make setup                 # create .venv from uv.lock
make data                  # download the policy documents into data/raw/
uv run fine-print ingest   # parse and chunk them, and report the counts
```

## Commands

| Command | What it does |
| --- | --- |
| `make setup` | Create or repair `.venv` from `uv.lock` |
| `make data` | Download the policy documents |
| `make lint` | Check linting and formatting |
| `make format` | Apply lint fixes and formatting |
| `make typecheck` | Run the type checker |
| `make test` | Run the tests with coverage |
| `make index` | Build the vector index |
| `make eval` | Measure retrieval quality |
| `make bench` | Compare vector index types |
| `make api` | Start the API on `127.0.0.1:8000` |
| `make app` | Start the web UI on `localhost:8501` |

## Environment

`.venv` is managed by uv, so use `uv run` or `.venv/bin/...` to run things. Add dependencies with `uv add` or `uv add --dev`, not `pip`. To start again from scratch, run `rm -rf .venv && make setup`.
