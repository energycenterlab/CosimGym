"""
Streamlit application entrypoint for visualizing simulation results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

st.set_page_config(
    page_title="Simulation Results Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.dashboard_app import run_dashboard


run_dashboard()
