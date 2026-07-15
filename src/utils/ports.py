"""Centralized default ports for CosimGym infra services.

Single source of truth = ``src/.env`` (also read natively by docker-compose, so
the containers and the Python processes stay in sync). Every port *default* in
the Python codebase resolves through this module, so a shared-machine user who
needs to dodge an occupied port changes it in ONE place (``src/.env``) instead
of hunting scattered literals.

Resolution order per port (first hit wins):
  1. the process environment (e.g. an explicit ``export REDIS_PORT=...``) — this
     keeps the historical env-var overrides working, and lets a per-run value
     (e.g. the one ``federate_launcher`` derives from ``--redis-url``) win.
  2. ``src/.env`` — loaded into ``os.environ`` at import via ``setdefault`` so it
     never clobbers an explicit export from (1).
  3. the hardcoded default below — identical to the historical value, so a repo
     with no ``.env`` behaves exactly as before.

Only *infra* ports live here (Redis, Mosquitto/MQTT, MinIO, and the HELICS
broker auto-assign range). Per-scenario ``broker_config.port`` values stay in
their scenario YAML — they are user co-simulation config, not global defaults.
"""

import os
from pathlib import Path
from typing import Tuple

# src/.env — sits next to this package's parent (src/), the same directory
# docker-compose treats as its project dir when run as `-f src/docker-compose.yaml`.
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _load_env_file() -> None:
    """Load ``src/.env`` into ``os.environ`` without overriding existing exports."""
    if not _ENV_FILE.exists():
        return
    for raw in _ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            # setdefault → an explicit environment export always wins over .env.
            os.environ.setdefault(key, val)


_load_env_file()


def _int(name: str, default: int, *aliases: str) -> int:
    """First non-empty of ``name`` then ``aliases`` in the environment, else ``default``."""
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value not in (None, ""):
            return int(value)
    return default


def _str(name: str, default: str, *aliases: str) -> str:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value not in (None, ""):
            return value
    return default


def redis_port() -> int:
    """Manager Redis port. Env: COSIM_REDIS_PORT (or legacy REDIS_PORT). Default 6379."""
    return _int("COSIM_REDIS_PORT", 6379, "REDIS_PORT")


def mqtt_port() -> int:
    """Host Mosquitto/MQTT port. Env: COSIM_MQTT_PORT (or legacy MQTT_PORT). Default 11883."""
    return _int("COSIM_MQTT_PORT", 11883, "MQTT_PORT")


def minio_port() -> int:
    """MinIO S3 API port. Env: COSIM_MINIO_PORT. Default 9000."""
    return _int("COSIM_MINIO_PORT", 9000, "MINIO_PORT")


def minio_console_port() -> int:
    """MinIO web console port. Env: COSIM_MINIO_CONSOLE_PORT. Default 9101."""
    return _int("COSIM_MINIO_CONSOLE_PORT", 9101)


def minio_endpoint() -> str:
    """Full MinIO S3 endpoint URL, e.g. ``http://localhost:9000``.

    Host from COSIM_MINIO_HOST (or legacy MINIO_HOST), default ``localhost``;
    port from :func:`minio_port`.
    """
    host = _str("COSIM_MINIO_HOST", "localhost", "MINIO_HOST")
    return f"http://{host}:{minio_port()}"


def helics_port_range() -> Tuple[int, int]:
    """(min, max) for the HELICS broker TCP auto-assign scan. Default (20000, 30000).

    Env: COSIM_HELICS_PORT_MIN / COSIM_HELICS_PORT_MAX.
    """
    return _int("COSIM_HELICS_PORT_MIN", 20000), _int("COSIM_HELICS_PORT_MAX", 30000)
