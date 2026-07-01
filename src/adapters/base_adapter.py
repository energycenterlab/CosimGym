"""
base_adapter.py

Abstract transport adapter used by the `stream` outbound mirror (any federate)
and by InterfaceFederate (bidirectional digital-twin bridge). Concrete
adapters (MQTT now; Redis/Kafka/Modbus/OPC-UA later) implement this ABC so
core federate code never depends on a specific transport.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class InterfaceAdapter(ABC):
    """Transport adapter contract shared by the `stream` mirror and InterfaceFederate."""

    @abstractmethod
    def connect(self) -> None:
        """Open the transport connection. Must not block the sim thread once returned."""

    @abstractmethod
    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Enqueue *payload* for outbound delivery on *topic*. Non-blocking."""

    @abstractmethod
    def subscribe(self, topics: List[str]) -> None:
        """Subscribe to *topics* so their values become readable via `latest()`."""

    @abstractmethod
    def latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Return the most recently received payload for *topic*, or None."""

    @abstractmethod
    def close(self) -> None:
        """Close the transport connection and stop any background threads."""
