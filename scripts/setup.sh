#!/usr/bin/env bash
# Create or repair .venv from uv.lock. Run through `make setup`.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with the official installer, then run this again:" >&2
    echo >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo >&2
    echo "Other install options: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv sync --locked
echo "Environment ready in .venv. Next: make data"
