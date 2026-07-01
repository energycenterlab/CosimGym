"""
mqtt_adapter.py

MQTT implementation of InterfaceAdapter, backed by Mosquitto. The paho-mqtt
client runs its network loop on its own background thread (`loop_start`), so
`connect`/`close` never block the sim thread. Outbound publish (M1) and
inbound subscribe/latest (M3) are implemented in later milestones.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import logging
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
        self._connected = threading.Event()

    def connect(self) -> None:
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        if not self._connected.wait(timeout=10):
            self.logger.warning(
                f"MqttAdapter '{self.client_id}': no CONNACK from {self.host}:{self.port} within 10s"
            )

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError("MqttAdapter outbound publish lands in M1")

    def subscribe(self, topics: List[str]) -> None:
        raise NotImplementedError("MqttAdapter inbound subscribe lands in M3")

    def latest(self, topic: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("MqttAdapter inbound latest() lands in M3")

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._connected.set()
        self.logger.info(f"MqttAdapter '{self.client_id}': connected to {self.host}:{self.port}")
