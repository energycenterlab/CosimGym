"""
override_registry.py

Shared Redis-backed channel for the digital-twin interface federate's OUTPUT
and PARAMETER overrides (M4). Unlike the M3 INPUT-injection path (a normal
HELICS pub the target subscribes to), an output/param override has no HELICS
representation — the target already computes that value itself — so the
interface federate writes the bounds-clipped external value here, and the
target federate/model reads it each step.

Key scheme: cosim:override:<scope>:<sim_id>:<federation>:<federate>:<entity>:<var>
`scope` is "output" or "param".

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import os
from typing import Any, Optional, Tuple

from utils.redis_client import RedisClient

_REGISTRY_PREFIX = "cosim:override"


def _key(scope: str, sim_id: str, federation: str, federate: str, entity: str, var: str) -> str:
    return f"{_REGISTRY_PREFIX}:{scope}:{sim_id}:{federation}:{federate}:{entity}:{var}"


def parse_target(target: str, default_federation: str) -> Tuple[str, str, str, str]:
    """Parse a bridge's `helics_key` (an override target, not a HELICS name here)
    into (federation, federate, entity, var). Same-federation form:
    '<federate>.<instance>/<var>'. Cross-federation form:
    '<federation>.<federate>.<instance>/<var>'.

    `entity` is reconstructed as '<federate>.<instance>' — that combined string
    is what BaseFederate actually uses as an entity id (e.g. `entity['id']`,
    `pub['entity_name']`), NOT the bare instance number.
    """
    path, var = target.split('/', 1)
    parts = path.split('.')
    if len(parts) == 2:
        federate, instance = parts
        return default_federation, federate, f"{federate}.{instance}", var
    if len(parts) == 3:
        federation, federate, instance = parts
        return federation, federate, f"{federate}.{instance}", var
    raise ValueError(f"Malformed override target '{target}'")


class OverrideRegistry:
    """Redis-backed store for output/param override values set by interface federates (see module docstring)."""

    def __init__(self, logger=None):
        """Connect to Redis using REDIS_HOST/REDIS_PORT env vars (defaults: localhost:6379)."""
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', '6379'))
        self._client = RedisClient(host=host, port=port, logger=logger)

    def set_override(self, scope: str, sim_id: str, federation: str, federate: str,
                      entity: str, var: str, value: Any) -> None:
        """Write an override value for one (scope, entity, var), expiring after 1 hour."""
        self._client.set_json(_key(scope, sim_id, federation, federate, entity, var),
                               {'value': value}, expire_seconds=3600)

    def clear_override(self, scope: str, sim_id: str, federation: str, federate: str,
                        entity: str, var: str) -> None:
        """Remove an override, restoring the target federate's computed value next step."""
        self._client.delete(_key(scope, sim_id, federation, federate, entity, var))

    def get_override(self, scope: str, sim_id: str, federation: str, federate: str,
                      entity: str, var: str) -> Optional[Any]:
        """Return the current override value for (scope, entity, var), or None if unset."""
        data = self._client.get_json(_key(scope, sim_id, federation, federate, entity, var))
        return data.get('value') if data else None
