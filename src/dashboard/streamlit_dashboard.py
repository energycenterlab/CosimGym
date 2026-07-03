"""
Streamlit application entrypoint. Combines the post-run results explorer
(`dashboard_app.py`) and the live MQTT view (`live_dashboard.py`) into a single
app via `st.navigation`, so both are served by one `streamlit run` process instead
of two separate scripts/ports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# st.set_page_config must be called exactly once, before pg.run() — Streamlit
# forbids calling it from within a page.
st.set_page_config(
    page_title="Simulation Results Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.dashboard_app import run_dashboard
from src.dashboard.live_dashboard import render_live_dashboard

pg = st.navigation(
    [
        st.Page(run_dashboard, title="Results", default=True),
        st.Page(render_live_dashboard, title="Live"),
    ]
)
pg.run()
