.PHONY: setup lint format typecheck test data index eval eval-holdout compare compare-answers bench api app

setup:
	bash scripts/setup.sh

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

test:
	uv run pytest --cov --cov-report=term-missing

data:
	uv run python scripts/download_data.py

index:
	uv run fine-print index

eval:
	uv run fine-print eval

# Reports on a frozen configuration. Never tune anything from its results.
eval-holdout:
	uv run fine-print eval --holdout --answers

compare:
	uv run python scripts/compare_chunking.py

# Runs every model and prompt on the tuning questions, which takes hours. Resumes if stopped.
compare-answers:
	uv run python scripts/compare_answers.py

bench:
	uv run fine-print bench

# Bound to localhost so the services are not visible to others on your network.
api:
	uv run uvicorn fine_print.api:app --host 127.0.0.1 --port 8000

# The file watcher is off because it trips over torch's internals on every reload.
app:
	uv run streamlit run app/streamlit_app.py --server.address localhost --server.fileWatcherType none --browser.gatherUsageStats false
