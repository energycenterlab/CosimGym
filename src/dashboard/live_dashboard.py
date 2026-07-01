"""
live_dashboard.py — Streamlit live view for streamed cosim data (digitaltwin_interfaces M5).

Subscribes to `cosim/#` on the Mosquitto broker and shows the latest value per topic plus a
rolling time-series chart, refreshed on a timer. This is the first *live* dashboard path — it
watches whatever any federate is currently mirroring via `flags.stream: true` (BaseFederate.
_stream_outbound) or bridging via an interface federate (InterfaceFederate), while
`dashboard_app.py` remains the historical, post-run explorer reading `results/` files.

Message payloads (both stream mirror and interface-federate bridges) are JSON:
`{sim_id, key, value, sim_time, wall_time}`. Interface-federate topics are user-declared in
`interface_config` and may not carry `sim_id`/`sim_time` (external actuator/sensor payloads) —
handled defensively below.

Run: `streamlit run src/dashboard/live_dashboard.py`
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import paho.mqtt.client as mqtt

MAX_HISTORY_PER_TOPIC = 500
DEFAULT_TOPIC_FILTER = "cosim/#"


@st.cache_resource(show_spinner=False)
def _get_subscriber(host: str, port: int, topic_filter: str) -> "_LiveSubscriber":
    subscriber = _LiveSubscriber(host, port, topic_filter)
    subscriber.start()
    return subscriber


class _LiveSubscriber:
    """Background paho-mqtt client feeding a lock-guarded per-topic history buffer.

    One instance per (host, port, topic_filter) combination, cached for the Streamlit session
    via st.cache_resource so the connection and buffers survive page reruns.
    """

    def __init__(self, host: str, port: int, topic_filter: str):
        self._host = host
        self._port = port
        self._topic_filter = topic_filter
        self._lock = threading.Lock()
        self._history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PER_TOPIC))
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"cosim_live_dashboard_{id(self)}")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def start(self) -> None:
        self._client.connect(self._host, self._port)
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self._topic_filter)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        payload.setdefault("wall_time", datetime.now().isoformat())
        payload["_received_at"] = time.time()
        with self._lock:
            self._history[msg.topic].append(payload)

    def snapshot(self) -> Dict[str, list]:
        with self._lock:
            return {topic: list(records) for topic, records in self._history.items()}


def render_live_dashboard() -> None:
    st.set_page_config(page_title="CosimGym Live View", layout="wide", initial_sidebar_state="expanded")
    st.title("CosimGym Live View")
    st.caption(
        "Live MQTT feed — shows values as they are streamed/bridged *during* a running "
        "simulation. For post-run analysis, use the main dashboard (`dashboard_app.py`)."
    )

    with st.sidebar:
        st.header("Broker")
        host = st.text_input("MQTT host", value="localhost")
        port = st.number_input("MQTT port", value=11883, min_value=1, max_value=65535, step=1)
        topic_filter = st.text_input("Topic filter", value=DEFAULT_TOPIC_FILTER)
        refresh_secs = st.slider("Refresh interval (s)", min_value=1, max_value=10, value=2)
        sim_id_filter = st.text_input("Filter by sim_id (optional)", value="")

    subscriber = _get_subscriber(host, int(port), topic_filter)
    snapshot = subscriber.snapshot()

    if sim_id_filter:
        snapshot = {
            topic: records for topic, records in snapshot.items()
            if any(r.get("sim_id") == sim_id_filter for r in records)
        }

    if not snapshot:
        st.info(
            f"No messages received yet on `{topic_filter}` from `{host}:{port}`. "
            "Start a scenario with `flags.stream: true` or an interface federate to see live data."
        )
        time.sleep(refresh_secs)
        st.rerun()
        return

    st.subheader(f"Latest values ({len(snapshot)} topics)")
    latest_rows = []
    for topic, records in sorted(snapshot.items()):
        last = records[-1]
        latest_rows.append({
            "topic": topic,
            "key": last.get("key", ""),
            "value": last.get("value"),
            "sim_id": last.get("sim_id", ""),
            "sim_time": last.get("sim_time"),
            "wall_time": last.get("wall_time"),
        })
    st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

    st.subheader("Time series")
    selected_topics = st.multiselect(
        "Topics to chart", options=sorted(snapshot.keys()),
        default=sorted(snapshot.keys())[:min(4, len(snapshot))],
    )
    if selected_topics:
        fig = go.Figure()
        for topic in selected_topics:
            records = snapshot[topic]
            x = [r.get("sim_time") if r.get("sim_time") is not None else r["_received_at"] for r in records]
            y = [r.get("value") for r in records]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=topic))
        fig.update_layout(height=450, xaxis_title="sim_time (falls back to wall clock)", yaxis_title="value")
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(refresh_secs)
    st.rerun()


render_live_dashboard()
