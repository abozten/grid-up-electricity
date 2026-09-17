#!/bin/bash
# ==============================================================================
# Grid-Up Electricity Transformer Web Explorer
# Launch Script
# ==============================================================================

set -e

PORT=${1:-8000}
HOST="0.0.0.0"

echo "⚡ Starting Grid-Up Transformer Web Explorer on http://localhost:${PORT}"

export PYTHONPATH=.

# 1. Build database if it does not exist
if [ ! -f "data/transformers.db" ]; then
    echo "📦 Initializing SQLite database (data/transformers.db)..."
    .venv/bin/python src/web/build_db.py
fi

# 2. Start Uvicorn Server
echo "🚀 Launching FastAPI server..."
.venv/bin/python -m uvicorn src.web.app:app --host "${HOST}" --port "${PORT}" --reload
