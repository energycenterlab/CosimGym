#!/bin/bash
# Streamlit dashboard startup script for local JSON results.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Starting Streamlit dashboard for simulation results..."
echo "Dashboard URL: http://localhost:8052"
echo "Results path: $REPO_ROOT/results"
echo "Press Ctrl+C to stop"
echo ""

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "cosim_gym" ]; then
    echo "Please activate the conda environment first:"
    echo "  conda activate cosim_gym"
    echo ""
    exit 1
fi

cd "$REPO_ROOT"
streamlit run src/dashboard/streamlit_dashboard.py --server.port=8052 --server.address=localhost
