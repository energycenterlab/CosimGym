#!/usr/bin/env python3
"""
reward_functions.py

Library of reusable reward functions for Reinforcement Learning agents.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

"""
catalog_loader.py

Standalone script that reads catalog.yaml and uploads all model metadata
to Redis using RedisJSON.  Designed to run as a one-shot docker-compose
service after the redis container is healthy, but also runnable manually:

    python src/models/model_catalog/catalog_loader.py

Redis key scheme
----------------
catalog:physical_models:<model_name>  – full model JSON
catalog:rl_agents:<model_name>        – full model JSON
catalog:index                         – {"physical_models": [...], "rl_agents": [...]}

Environment variables (with defaults)
--------------------------------------
REDIS_HOST  localhost
REDIS_PORT  6379
"""

import os
import sys
import time
import logging
from pathlib import Path

import yaml
import redis

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# catalog.yaml lives in the same directory as this script
CATALOG_YAML = Path(__file__).parent / "catalog.yaml"

# Map catalog 'category' field value -> Redis key prefix
CATEGORY_MAP: dict = {
    "physical_model": "physical_models",
    "rl_agent":       "rl_agents",
}
DEFAULT_CATEGORY = "other"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """Load and return the contents of a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def connect_redis(host: str, port: int, max_retries: int = 10, retry_delay: float = 2.0) -> redis.Redis:
    """Connect to Redis, retrying on failure (useful when called right after container start)."""
    for attempt in range(1, max_retries + 1):
        try:
            client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            return client
        except redis.ConnectionError as exc:
            logger.warning(
                f"Attempt {attempt}/{max_retries} – Redis not ready: {exc}. "
                f"Retrying in {retry_delay}s…"
            )
            time.sleep(retry_delay)
    raise RuntimeError(f"Could not connect to Redis at {host}:{port} after {max_retries} attempts.")


def upload_catalog(client: redis.Redis, catalog_data: dict) -> None:
    """Push every model in catalog_data to Redis and update the index key."""
    models: dict = catalog_data.get("models", {})
    index: dict = {}  # {category_key: [model_name, ...]}

    for model_name, model_data in models.items():
        category_raw = model_data.get("category", DEFAULT_CATEGORY)
        category_key = CATEGORY_MAP.get(category_raw, DEFAULT_CATEGORY)

        redis_key = f"catalog:{category_key}:{model_name}"

        # Store the full model payload (include the name for convenience)
        payload = {"name": model_name, **model_data}
        client.json().set(redis_key, ".", payload)
        logger.info(f"  Uploaded: {redis_key}")

        index.setdefault(category_key, [])
        if model_name not in index[category_key]:
            index[category_key].append(model_name)

    # Write the index so clients can discover available models without scanning keys
    client.json().set("catalog:index", ".", index)
    logger.info(f"Catalog index written: {index}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))

    logger.info(f"Loading catalog from {CATALOG_YAML}")
    catalog_data = load_yaml(CATALOG_YAML)
    model_count = len(catalog_data.get("models", {}))
    logger.info(f"Found {model_count} model(s) in catalog.yaml")

    client = connect_redis(host, port)
    upload_catalog(client, catalog_data)
    logger.info("Catalog upload complete.")


if __name__ == "__main__":
    main()
