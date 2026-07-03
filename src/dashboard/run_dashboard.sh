#!/bin/bash
# Streamlit dashboard startup script: Results page (JSON/Parquet) + Live page (MQTT).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Starting Streamlit dashboard for simulation results..."
echo "Dashboard URL: http://localhost:8052"
echo "Results path: $REPO_ROOT/results"
echo "Press Ctrl+C to stop"
echo ""

cd "$REPO_ROOT"

if [ "$CONDA_DEFAULT_ENV" != "cosim_gym" ]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate cosim_gym
fi

streamlit run src/dashboard/streamlit_dashboard.py --server.port=8052 --server.address=localhost --server.headless true
