"""
Advanced caching layer for dashboard performance optimization.
Handles persistent metadata caching and lazy record iteration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cosim_dashboard_cache"
METADATA_INDEX_FILE = CACHE_DIR / "metadata_index.json"
INDEX_TTL_SECONDS = 3600  # 1 hour


class MetadataIndex:
    """
    Persistent cache for scenario/sim_id/federation metadata.
    Avoids repeated directory scans by caching the inventory.
    """

    def __init__(self):
        self.index: dict[str, Any] = {}
        self.last_update = 0.0

    def load(self) -> None:
        """Load cached metadata index from disk."""
        if not METADATA_INDEX_FILE.exists():
            return

        try:
            with open(METADATA_INDEX_FILE, encoding="utf-8") as f:
                data = json.load(f)
                self.index = data.get("index", {})
                # Use stored timestamp to know when cache was last updated
                self.last_update = data.get("timestamp", 0.0)
        except (json.JSONDecodeError, OSError):
            self.index = {}
            self.last_update = 0.0

    def save(self) -> None:
        """Persist metadata index to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(METADATA_INDEX_FILE, encoding="utf-8", mode="w") as f:
                json.dump(
                    {"index": self.index, "timestamp": self.last_update},
                    f,
                )
        except OSError as e:
            logger.warning(f"Failed to save metadata index: {e}")

    def get_scenarios(self, results_path: Path) -> dict[str, Any]:
        """
        Get scenario index with caching.
        Returns: {scenario_name: {sim_ids: [...]}}
        
        Smart mode: Uses cache if it exists and is not stale,
        only scans if necessary.
        """
        cache_key = str(results_path)
        cached = self.index.get(cache_key)

        # If cache exists in memory, use it (fastest)
        if cached is not None:
            # Check if persisted cache is still valid (compare mtimes)
            if self._is_cache_fresh(results_path):
                # Cache exists on disk and is fresh, re-use it
                return cached
        
        # Cache is stale or doesn't exist - scan directory
        cached = self._scan_scenarios(results_path)
        self.index[cache_key] = cached
        self.save()

        return cached

    def _is_cache_fresh(self, results_path: Path) -> bool:
        """Check if cached metadata is still valid (compare directory mtime)."""
        if not results_path.exists() or not self.last_update:
            return False
        
        try:
            # If results directory hasn't changed since our last index, cache is fresh
            results_mtime = results_path.stat().st_mtime
            return results_mtime <= self.last_update
        except OSError:
            return False

    def _scan_scenarios(self, results_path: Path) -> dict[str, Any]:
        """Scan results directory and build scenario index."""
        if not results_path.exists():
            return {}

        scenarios = {}
        try:
            for scenario_dir in sorted(results_path.iterdir()):
                if not scenario_dir.is_dir():
                    continue

                scenario_name = scenario_dir.name
                sim_ids = sorted(
                    (d.name for d in scenario_dir.iterdir() if d.is_dir()),
                    reverse=True,
                )
                scenarios[scenario_name] = {"sim_ids": sim_ids}
        except OSError as e:
            logger.warning(f"Failed to scan scenarios: {e}")
        
        # Update timestamp to mark when scan was completed
        self.last_update = time.time()
        return scenarios

    def invalidate(self) -> None:
        """Clear the cache."""
        self.index = {}


class LazyRecordIterator:
    """
    Lazy iterator for records that yields batches without loading entire dataset.
    Useful for large simulations to reduce memory footprint.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]], batch_size: int = 1000):
        """
        Initialize lazy iterator.

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            The records to iterate over.
        batch_size : int
            Number of records per batch.
        """
        self.records = records
        self.batch_size = batch_size
        self.position = 0

    def __iter__(self) -> Iterator[list[Mapping[str, Any]]]:
        """Yield batches of records."""
        self.position = 0
        while self.position < len(self.records):
            batch = self.records[self.position : self.position + self.batch_size]
            self.position += self.batch_size
            yield batch

    def __len__(self) -> int:
        """Return total record count."""
        return len(self.records)


# Global metadata index instance
_metadata_index = MetadataIndex()
_metadata_index.load()


def get_global_metadata_index() -> MetadataIndex:
    """Get the global metadata index instance."""
    return _metadata_index
