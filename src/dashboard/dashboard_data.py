"""Pure data-loading helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ijson
import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results"

TIME_SERIES_COLUMNS = [
    "time",
    "federation",
    "federate",
    "model_instance",
    "attribute",
    "type",
    "mode",
    "value",
]

EPISODE_COLUMNS = [
    "federation",
    "federate",
    "mode",
    "episode",
    "episode_reward",
    "episode_length",
]

FILTERABLE_TAGS = {"federation", "federate", "model_instance", "attribute", "type", "mode"}

_FILE_PATTERNS = [
    (re.compile(r"^(.+)_(train|test)_rl_storage\.json$"), True),
    (re.compile(r"^(.+)_(train|test)_storage\.json$"), False),
    (re.compile(r"^(.+)_storage\.json$"), False),
]


def parse_storage_filename(filename: str) -> tuple[str, str, bool] | None:
    """
    Return ``(federate_name, mode, is_rl)`` for a storage JSON filename.

    ``mode`` is ``train``, ``test``, or ``-`` for legacy files without an
    explicit mode.
    """
    for pattern, is_rl in _FILE_PATTERNS:
        match = pattern.match(filename)
        if not match:
            continue

        groups = match.groups()
        if len(groups) == 2:
            return groups[0], groups[1], is_rl
        return groups[0], "-", is_rl

    return None


def list_scenarios(results_path: Path = RESULTS_PATH) -> list[str]:
    """List scenario directories inside the results folder."""
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(directory.name for directory in results_path.iterdir() if directory.is_dir())


def list_simulation_ids(
    scenario: str,
    results_path: Path = RESULTS_PATH,
) -> list[str]:
    """List simulation run IDs (newest first) for a scenario."""
    path = results_path / scenario
    if not path.exists():
        return []
    return sorted((directory.name for directory in path.iterdir() if directory.is_dir()), reverse=True)


def load_simulation_metadata(
    scenario: str,
    sim_id: str,
    results_path: Path = RESULTS_PATH,
) -> dict[str, Any] | None:
    """Read and return ``metadata.json`` for a simulation run."""
    metadata_path = results_path / scenario / sim_id / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def list_federations(
    scenario: str,
    sim_id: str,
    results_path: Path = RESULTS_PATH,
) -> list[str]:
    """List federation subdirectory names for a simulation run."""
    path = results_path / scenario / sim_id
    if not path.exists():
        return []
    return sorted(directory.name for directory in path.iterdir() if directory.is_dir())


def load_all_records(
    scenario: str,
    sim_id: str,
    results_path: Path = RESULTS_PATH,
) -> list[dict[str, Any]]:
    """
    Parse federate storage JSON files under a simulation run into flat records.

    Record schema:
    ``federation, federate, model_instance, attribute, type, mode, time, value``
    """
    base_path = results_path / scenario / sim_id
    if not base_path.exists():
        return []

    records: list[dict[str, Any]] = []
    section_type = {"inputs": "input", "outputs": "output", "params": "param"}

    for federation_dir in sorted(base_path.iterdir()):
        if not federation_dir.is_dir():
            continue

        federation = federation_dir.name
        for json_file in sorted(federation_dir.glob("*.json")):
            parsed = parse_storage_filename(json_file.name)
            if parsed is None:
                continue

            federate_name, mode, is_rl = parsed
            payload = _read_json_dict(json_file)
            if payload is None:
                continue

            time_index = payload.get("time", [])
            if not isinstance(time_index, list) or not time_index:
                continue

            if is_rl:
                _append_rl_records(records, federation, federate_name, mode, time_index, payload)
                continue

            _append_standard_records(records, federation, mode, time_index, payload, section_type)

    return records


def load_rl_episode_records(
    scenario: str,
    sim_id: str,
    results_path: Path = RESULTS_PATH,
) -> list[dict[str, Any]]:
    """Load episode-level RL metrics from RL storage files."""
    base_path = results_path / scenario / sim_id
    if not base_path.exists():
        return []

    records: list[dict[str, Any]] = []

    for federation_dir in sorted(base_path.iterdir()):
        if not federation_dir.is_dir():
            continue

        federation = federation_dir.name
        for json_file in sorted(federation_dir.glob("*_rl_storage.json")):
            parsed = parse_storage_filename(json_file.name)
            if parsed is None:
                continue

            federate_name, mode, is_rl = parsed
            if not is_rl:
                continue

            payload = _read_json_dict(json_file)
            if payload is None:
                continue

            episode_rewards = payload.get("episode_rewards", [])
            episode_lengths = payload.get("episode_lengths", [])
            if not isinstance(episode_rewards, list) or not isinstance(episode_lengths, list):
                continue

            for episode_index, (reward, episode_length) in enumerate(
                zip(episode_rewards, episode_lengths)
            ):
                records.append(
                    {
                        "federation": federation,
                        "federate": federate_name,
                        "mode": mode,
                        "episode": episode_index,
                        "episode_reward": reward,
                        "episode_length": episode_length,
                    }
                )

    return records


def get_available_tags(
    records: Sequence[Mapping[str, Any]],
    tag: str,
    federations: Sequence[str] | None = None,
    federates: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    attributes: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> list[str]:
    """Return sorted unique values for ``tag`` after applying upstream filters."""
    if tag not in FILTERABLE_TAGS:
        raise ValueError(f"Unsupported tag: {tag}")

    filtered = filter_records(
        records,
        federations=federations,
        federates=federates,
        models=models,
        attributes=attributes,
        types=types,
        modes=modes,
    )
    return sorted({record[tag] for record in filtered})


def filter_records(
    records: Iterable[Mapping[str, Any]],
    federations: Sequence[str] | None = None,
    federates: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    attributes: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Filter records with the same federation/federate/model cascade used by the UI."""
    return [
        record
        for record in records
        if _matches_filters(
            record,
            federations=federations,
            federates=federates,
            models=models,
            attributes=attributes,
            types=types,
            modes=modes,
        )
    ]


def build_dataframe(
    records: Sequence[Mapping[str, Any]],
    federations: Sequence[str] | None = None,
    federates: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    attributes: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a filtered time-series dataframe from flat storage records."""
    filtered_records = filter_records(
        records,
        federations=federations,
        federates=federates,
        models=models,
        attributes=attributes,
        types=types,
        modes=modes,
    )
    if not filtered_records:
        return pd.DataFrame(columns=TIME_SERIES_COLUMNS)

    dataframe = pd.DataFrame(filtered_records, columns=TIME_SERIES_COLUMNS)
    dataframe["time"] = pd.to_datetime(dataframe["time"], errors="coerce")
    dataframe = dataframe.dropna(subset=["time"])
    return dataframe.sort_values(
        ["time", "federation", "federate", "model_instance", "attribute", "type", "mode"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_episode_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build an RL episode-level dataframe from flat records."""
    if not records:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    return pd.DataFrame(records, columns=EPISODE_COLUMNS)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    """
    Read JSON file with optimized buffering for faster I/O.
    """
    try:
        # Use larger buffer for faster disk I/O (default is 8KB, we use 256KB)
        with open(path, encoding="utf-8", buffering=256 * 1024) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def _append_rl_records(
    records: list[dict[str, Any]],
    federation: str,
    federate_name: str,
    mode: str,
    time_index: Sequence[Any],
    payload: Mapping[str, Any],
) -> None:
    record_specs = (
        ("observations", "observation"),
        ("actions", "action"),
    )

    for section_key, record_type in record_specs:
        section = payload.get(section_key, {})
        if not isinstance(section, dict):
            continue

        for attribute, values in section.items():
            if not _is_time_aligned(values, time_index):
                continue
            for time_value, record_value in zip(time_index, values):
                records.append(
                    {
                        "federation": federation,
                        "federate": federate_name,
                        "model_instance": "agent",
                        "attribute": attribute,
                        "type": record_type,
                        "mode": mode,
                        "time": time_value,
                        "value": record_value,
                    }
                )

    rewards = payload.get("rewards", [])
    if not _is_time_aligned(rewards, time_index):
        return

    for time_value, record_value in zip(time_index, rewards):
        records.append(
            {
                "federation": federation,
                "federate": federate_name,
                "model_instance": "agent",
                "attribute": "reward",
                "type": "reward",
                "mode": mode,
                "time": time_value,
                "value": record_value,
            }
        )


def _append_standard_records(
    records: list[dict[str, Any]],
    federation: str,
    mode: str,
    time_index: Sequence[Any],
    payload: Mapping[str, Any],
    section_type: Mapping[str, str],
) -> None:
    for section_key, type_label in section_type.items():
        section = payload.get(section_key, {})
        if not isinstance(section, dict):
            continue

        for instance_key, signals in section.items():
            if not isinstance(signals, dict) or not signals:
                continue

            parts = instance_key.split(".", 1)
            federate_name = parts[0]
            model_instance = parts[1] if len(parts) > 1 else instance_key

            for attribute, values in signals.items():
                if not _is_time_aligned(values, time_index):
                    continue
                for time_value, record_value in zip(time_index, values):
                    records.append(
                        {
                            "federation": federation,
                            "federate": federate_name,
                            "model_instance": model_instance,
                            "attribute": attribute,
                            "type": type_label,
                            "mode": mode,
                            "time": time_value,
                            "value": record_value,
                        }
                    )


def _is_time_aligned(values: Any, time_index: Sequence[Any]) -> bool:
    return isinstance(values, list) and len(values) == len(time_index)


def _matches_filters(
    record: Mapping[str, Any],
    federations: Sequence[str] | None = None,
    federates: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    attributes: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> bool:
    filters = (
        ("federation", federations),
        ("federate", federates),
        ("model_instance", models),
        ("attribute", attributes),
        ("type", types),
        ("mode", modes),
    )

    for field_name, selected_values in filters:
        if selected_values and record[field_name] not in selected_values:
            return False

    return True
