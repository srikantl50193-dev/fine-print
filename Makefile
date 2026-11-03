.PHONY: install index eval bench app test

install:
	uv sync

index:
	uv run python scripts/build_index.py --strategy fixed
	uv run python scripts/build_index.py --strategy section

eval:
	uv run python -m fine_print.evaluate

bench:
	uv run python scripts/benchmark_index.py --strategy section

# Bound to localhost so the app is not visible to others on your network.
# The file watcher is off because it trips over torch's internals on every reload.
app:
	uv run streamlit run app.py --server.address localhost --server.fileWatcherType none --browser.gatherUsageStats false

test:
	uv run pytest -q
