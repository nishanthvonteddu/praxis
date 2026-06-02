#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync
exec uv run uvicorn praxis.main:app --host 0.0.0.0 --port "${PRAXIS_PORT:-8099}" --reload
