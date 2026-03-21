"""
RedisCatalog.py

Redis-backed model catalog – drop-in replacement for ModelCatalog.

Queries the Redis instance populated by catalog_loader.py instead of reading YAML files from disk.  The public interface mirrors ModelCatalog so switching is a one-line change in BaseFederate._register_entities:
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
"""
RedisCatalog.py

Redis-backed model catalog – drop-in replacement for ModelCatalog.

Queries the Redis instance populated by catalog_loader.py instead of
reading YAML files from disk.  The public interface mirrors ModelCatalog
so switching is a one-line change in BaseFederate._register_entities:

    # before
    catalog = ModelCatalog()
    # after
    catalog = RedisCatalog()

Falls back gracefully (logs a warning and returns None) when Redis is
unreachable, so the class can be instantiated safely at import time.

Environment variables (with defaults)
--------------------------------------
REDIS_HOST  localhost
REDIS_PORT  6379
"""

import os
import logging
from typing import Any, Dict, List, Optional, Set

import redis

from models.model_catalog.ModelCatalog import (
    InterfaceType,
    ModelMetadata,
    ParameterSpec,
    ParameterType,
)


class RedisCatalog:
    """
    Redis-backed model catalog.

    Fetches model metadata from keys populated by catalog_loader.py:

        catalog:physical_models:<model_name>
        catalog:rl_agents:<model_name>
        catalog:index  →  {"physical_models": [...], "rl_agents": [...]}

    Parameters
    ----------
    host : str, optional
        Redis host.  Defaults to REDIS_HOST env var or 'localhost'.
    port : int, optional
        Redis port.  Defaults to REDIS_PORT env var or 6379.
    logger : logging.Logger, optional
        Custom logger.  Defaults to module-level logger.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        self._client: Optional[redis.Redis] = None

        try:
            client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            client.ping()
            self._client = client
            self.logger.info(f"RedisCatalog connected to Redis at {host}:{port}")
        except redis.ConnectionError as exc:
            self.logger.warning(
                f"RedisCatalog: cannot connect to Redis at {host}:{port} – {exc}. "
                "All queries will return None / empty list."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self) -> Dict[str, List[str]]:
        """Return the catalog index dict, or {} on failure."""
        if not self._client:
            return {}
        try:
            return self._client.json().get("catalog:index", ".") or {}
        except Exception as exc:
            self.logger.error(f"RedisCatalog: failed to read catalog:index – {exc}")
            return {}

    def _category_for(self, model_name: str) -> Optional[str]:
        """Look up which category bucket contains *model_name*."""
        for category_key, names in self._index().items():
            if model_name in names:
                return category_key
        return None

    @staticmethod
    def _parse(model_name: str, data: Dict[str, Any]) -> ModelMetadata:
        """Reconstruct a ModelMetadata dataclass from a raw dict (as stored in Redis)."""

        def _specs(raw: Dict[str, Any]) -> Dict[str, ParameterSpec]:
            specs: Dict[str, ParameterSpec] = {}
            for name, sd in (raw or {}).items():
                specs[name] = ParameterSpec(
                    name=name,
                    type=ParameterType(sd.get("type", "float")),
                    default_value=sd.get("default_value"),
                    description=sd.get("description", ""),
                    unit=sd.get("unit", ""),
                    min_value=sd.get("min_value"),
                    max_value=sd.get("max_value"),
                    required=sd.get("required", False),
                    tags=sd.get("tags", []),
                )
            return specs

        return ModelMetadata(
            name=model_name,
            class_name=data["class_name"],
            module_path=data["module_path"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            domain=data.get("domain", ""),
            category=data.get("category", ""),
            tags=data.get("tags", []),
            time_step=data.get("time_step", 1.0),
            min_time_step=data.get("min_time_step", 0.0),
            max_time_step=data.get("max_time_step", float("inf")),
            parameters=_specs(data.get("parameters", {})),
            inputs=_specs(data.get("inputs", {})),
            outputs=_specs(data.get("outputs", {})),
            states=_specs(data.get("states", {})),
            dependencies=data.get("dependencies", []),
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors ModelCatalog)
    # ------------------------------------------------------------------

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Fetch and return ModelMetadata for *model_name* from Redis."""
        if not self._client:
            self.logger.error("RedisCatalog: no Redis connection.")
            return None

        category_key = self._category_for(model_name)
        if not category_key:
            self.logger.warning(f"RedisCatalog: model '{model_name}' not found in catalog:index.")
            return None

        redis_key = f"catalog:{category_key}:{model_name}"
        try:
            data = self._client.json().get(redis_key, ".")
        except Exception as exc:
            self.logger.error(f"RedisCatalog: failed to fetch {redis_key} – {exc}")
            return None

        if not data:
            self.logger.warning(f"RedisCatalog: no data at key {redis_key}.")
            return None

        # 'name' was added by catalog_loader for convenience; strip before parsing
        data.pop("name", None)
        return self._parse(model_name, data)

    def get_inputs_outputs(
        self,
        model_name: str,
        simulation_id: Optional[str] = None,
        instance_ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Dict[str, ParameterSpec]]]:
        """
        Return only the inputs and outputs sections for *model_name*.

        Returns a dict with two keys::

            {
                "inputs":  {signal_name: ParameterSpec, ...},
                "outputs": {signal_name: ParameterSpec, ...},
            }

        Returns None if the model is not found or Redis is unavailable.
        """
        if not self._client:
            self.logger.error("RedisCatalog: no Redis connection.")
            return None

        category_key = self._category_for(model_name)
        if not category_key:
            self.logger.warning(
                f"RedisCatalog: model '{model_name}' not found in catalog:index."
            )
            return None

        redis_key = f"catalog:{category_key}:{model_name}"
        try:
            # Fetch only the two sub-documents we need – avoids pulling the full model
            raw_inputs  = self._client.json().get(redis_key, ".inputs")  or {}
            raw_outputs = self._client.json().get(redis_key, ".outputs") or {}

            # Scenario-scoped overrides for dynamic model metadata.
            if simulation_id and instance_ctx:
                fed = instance_ctx.get("federation")
                federate = instance_ctx.get("federate")
                instance = instance_ctx.get("instance")
                if fed is not None and federate is not None and instance is not None:
                    override_key = f"cosim:catalog_override:{simulation_id}:{fed}:{federate}:{instance}"
                    override_payload = self._client.json().get(override_key, ".") or {}
                    raw_inputs = {**raw_inputs, **(override_payload.get("inputs") or {})}
                    raw_outputs = {**raw_outputs, **(override_payload.get("outputs") or {})}
        except Exception as exc:
            self.logger.error(
                f"RedisCatalog: failed to fetch inputs/outputs for '{model_name}' – {exc}"
            )
            return None

        def _specs(raw: Dict[str, Any]) -> Dict[str, ParameterSpec]:
            specs: Dict[str, ParameterSpec] = {}
            for name, sd in raw.items():
                specs[name] = ParameterSpec(
                    name=name,
                    type=ParameterType(sd.get("type", "float")),
                    default_value=sd.get("default_value"),
                    description=sd.get("description", ""),
                    unit=sd.get("unit", ""),
                    min_value=sd.get("min_value"),
                    max_value=sd.get("max_value"),
                    required=sd.get("required", False),
                    tags=sd.get("tags", []),
                )
            return specs

        return {"inputs": _specs(raw_inputs), "outputs": _specs(raw_outputs)}

    def query(self, path: str) -> Any:
        """
        Generic leaf accessor using a dot-separated path.

        The first segment must always be the model name; the rest maps
        directly to the JSON structure stored in Redis.

        Examples
        --------
        c.query("spring_mass_damper.parameters.mass.unit")      # → "kg"
        c.query("spring_mass_damper.outputs.position.min_value")# → -100.0
        c.query("spring_mass_damper.version")                   # → "1.0.0"
        c.query("spring_mass_damper.tags")                      # → ["example"]
        c.query("spring_mass_damper")                           # → full model dict

        Notes
        -----
        - Returns None (with a warning) if the path does not resolve.
        - Array index access is not supported in path syntax; retrieve the
          list with e.g. query("model.tags") and index in Python.
        """
        if not self._client:
            self.logger.error("RedisCatalog: no Redis connection.")
            return None

        # Split into model name and optional field path
        parts = path.split(".", maxsplit=1)
        model_name = parts[0]
        field_path = parts[1] if len(parts) == 2 else None

        category_key = self._category_for(model_name)
        if not category_key:
            self.logger.warning(
                f"RedisCatalog: model '{model_name}' not found in catalog:index."
            )
            return None

        redis_key = f"catalog:{category_key}:{model_name}"
        json_path = f".{field_path}" if field_path else "."

        try:
            result = self._client.json().get(redis_key, json_path)
        except redis.exceptions.ResponseError:
            # RedisJSON raises ResponseError when the path does not exist in the document
            self.logger.warning(
                f"RedisCatalog: path '{path}' does not exist in the model document."
            )
            return None
        except Exception as exc:
            self.logger.error(
                f"RedisCatalog: failed to query '{redis_key}' at path '{json_path}' – {exc}"
            )
            return None

        if result is None:
            self.logger.warning(
                f"RedisCatalog: path '{path}' resolved to nothing."
            )
        return result

    def search_models(
        self,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """Return all models matching the given filters (all optional)."""
        results: List[ModelMetadata] = []
        for _cat_key, model_names in self._index().items():
            for model_name in model_names:
                meta = self.get_model_metadata(model_name)
                if meta is None:
                    continue
                if domain and meta.domain != domain:
                    continue
                if category and meta.category != category:
                    continue
                if tags and not any(t in meta.tags for t in tags):
                    continue
                results.append(meta)
        return results

    
