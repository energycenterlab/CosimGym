"""
mqtt_adapter.py

MQTT implementation of InterfaceAdapter, backed by Mosquitto. `connect`/`close`
never block the sim thread: the paho-mqtt client runs its network I/O on its
own background thread (`loop_start`), and outbound publishes go through a
second, bounded drop-oldest queue drained by our own background thread — so a
slow/unreachable broker can never make `publish()` block or grow unbounded.
Inbound: `subscribe()` registers topics with the broker; the paho network
thread's `on_message` callback writes into a lock-guarded latest-value dict,
read (cheaply, no I/O) by `latest()` on the sim thread.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import json
import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import paho.mqtt.client as mqtt

from adapters.base_adapter import InterfaceAdapter


class MqttAdapter(InterfaceAdapter):
    def __init__(
        self,
        client_id: str = "cosim_dt",
        host: str = "localhost",
        port: int = 1883,
        qos: int = 0,
        outbound_maxsize: int = 1000,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client_id = client_id
        self.host = host
        self.port = port
        self.qos = qos
        self.logger = logger or logging.getLogger(__name__)

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._connected = threading.Event()

        self._outbound: "queue.Queue[tuple[str, Dict[str, Any]]]" = queue.Queue(maxsize=outbound_maxsize)
        self._stop_drain = threading.Event()
        self._drain_thread: Optional[threading.Thread] = None

        self._inbound: Dict[str, Dict[str, Any]] = {}
        self._inbound_lock = threading.Lock()

    def connect(self) -> None:
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        if not self._connected.wait(timeout=10):
            self.logger.warning(
                f"MqttAdapter '{self.client_id}': no CONNACK from {self.host}:{self.port} within 10s"
            )
        self._stop_drain.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._outbound.full():
            try:
                self._outbound.get_nowait()  # drop-oldest: dashboards want the latest value
            except queue.Empty:
                pass
        try:
            self._outbound.put_nowait((topic, payload))
        except queue.Full:
            pass  # lost the race with another producer thread — fine, next publish will land

    def subscribe(self, topics: List[str]) -> None:
        for topic in topics:
            self._client.subscribe(topic, qos=self.qos)

    def latest(self, topic: str) -> Optional[Dict[str, Any]]:
        with self._inbound_lock:
            return self._inbound.get(topic)

    def close(self) -> None:
        self._stop_drain.set()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=2)
        self._client.loop_stop()
        self._client.disconnect()

    def _drain_loop(self) -> None:
        while not self._stop_drain.is_set():
            try:
                topic, payload = self._outbound.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._client.publish(topic, json.dumps(payload, default=str), qos=self.qos)
            except Exception:
                self.logger.exception(f"MqttAdapter '{self.client_id}': failed to publish on '{topic}'")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._connected.set()
        self.logger.info(f"MqttAdapter '{self.client_id}': connected to {self.host}:{self.port}")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {'value': msg.payload}
        with self._inbound_lock:
            self._inbound[msg.topic] = payload
