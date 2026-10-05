#!/bin/bash
# Shell launcher for the Wearable Modality Synergy Streamlit Dashboard

set -eo pipefail

PORT=${1:-8501}

echo "Launching Streamlit Dashboard on port $PORT..."
echo "If running on Turing HPC, forward the port via SSH:"
echo "  ssh -L $PORT:localhost:$PORT khickey@turing.wpi.edu"
echo ""

exec uv run streamlit run scripts/dashboard_app.py --server.port "$PORT" --server.headless true
