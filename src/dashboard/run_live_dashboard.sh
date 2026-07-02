#!/bin/bash
# Streamlit live-view startup script — subscribes to the Mosquitto broker (cosim/#).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Starting Streamlit live dashboard (MQTT: cosim/#)..."
echo "Dashboard URL: http://localhost:8053"
echo "Requires: docker compose -f src/docker-compose.yaml up -d  (mosquitto running)"
echo "Press Ctrl+C to stop"
echo ""

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "cosim_gym" ]; then
    echo "Please activate the conda environment first:"
    echo "  conda activate cosim_gym"
    echo ""
    exit 1
fi

cd "$REPO_ROOT"
streamlit run src/dashboard/live_dashboard.py --server.port=8053 --server.address=localhost
