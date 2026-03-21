"""
Parquet-based caching for simulation result records.
Provides 5-10x speedup for repeat loads of the same simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_CACHE_DIR = Path.home() / ".cosim_dashboard_cache" / "parquet"


def get_parquet_cache_path(scenario: str, sim_id: str, record_type: str = "timeseries") -> Path:
    """
    Get the path for cached Parquet file.

    Parameters
    ----------
    scenario : str
        Scenario name
    sim_id : str
        Simulation ID
    record_type : str
        Type of records: 'timeseries' or 'episodes'

    Returns
    -------
    Path
        Path to Parquet file
    """
    return PARQUET_CACHE_DIR / scenario / sim_id / f"{record_type}.parquet"


def _check_cache_valid(parquet_path: Path, results_path: Path) -> bool:
    """
    Check if Parquet cache is still valid by comparing mtimes.

    Parameters
    ----------
    parquet_path : Path
        Path to Parquet cache file
    results_path : Path
        Path to source results directory

    Returns
    -------
    bool
        True if cache is valid (not stale)
    """
    if not parquet_path.exists():
        return False

    try:
        parquet_mtime = parquet_path.stat().st_mtime
        results_mtime = results_path.stat().st_mtime
        # Cache is valid if it's newer than the results directory
        return parquet_mtime > results_mtime
    except OSError:
        return False


def save_records_to_parquet(
    records: list[dict[str, Any]],
    scenario: str,
    sim_id: str,
    record_type: str = "timeseries",
) -> bool:
    """
    Save records to Parquet cache.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records to cache
    scenario : str
        Scenario name
    sim_id : str
        Simulation ID
    record_type : str
        Type of records: 'timeseries' or 'episodes'

    Returns
    -------
    bool
        True if save was successful
    """
    if not records:
        return False

    try:
        parquet_path = get_parquet_cache_path(scenario, sim_id, record_type)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(records)
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"Saved Parquet cache: {parquet_path} ({len(records)} records)")
        return True
    except Exception as e:
        logger.warning(f"Failed to save Parquet cache: {e}")
        return False


def load_records_from_parquet(
    scenario: str,
    sim_id: str,
    results_path: Path,
    record_type: str = "timeseries",
) -> list[dict[str, Any]] | None:
    """
    Load records from Parquet cache if valid.

    Parameters
    ----------
    scenario : str
        Scenario name
    sim_id : str
        Simulation ID
    results_path : Path
        Path to source results
    record_type : str
        Type of records: 'timeseries' or 'episodes'

    Returns
    -------
    list[dict[str, Any]] | None
        Records if cache is valid and found, None otherwise
    """
    parquet_path = get_parquet_cache_path(scenario, sim_id, record_type)
    results_dir = results_path / scenario / sim_id

    # Check if cache exists and is valid
    if not _check_cache_valid(parquet_path, results_dir):
        return None

    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        records = df.to_dict("records")
        logger.debug(f"Loaded from Parquet cache: {parquet_path} ({len(records)} records)")
        return records
    except Exception as e:
        logger.warning(f"Failed to load Parquet cache: {e}")
        return None


def clear_parquet_cache() -> None:
    """Clear all Parquet cache files."""
    try:
        if PARQUET_CACHE_DIR.exists():
            import shutil

            shutil.rmtree(PARQUET_CACHE_DIR)
            logger.info("Cleared Parquet cache")
    except Exception as e:
        logger.warning(f"Failed to clear Parquet cache: {e}")
