from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dashboard.dashboard_data import (
    build_dataframe,
    build_episode_dataframe,
    get_available_tags,
    load_all_records,
    load_rl_episode_records,
    parse_storage_filename,
)


class ParseStorageFilenameTests(unittest.TestCase):
    def test_supports_rl_partitioned_files(self) -> None:
        self.assertEqual(
            parse_storage_filename("controller_train_rl_storage.json"),
            ("controller", "train", True),
        )

    def test_supports_standard_partitioned_files(self) -> None:
        self.assertEqual(
            parse_storage_filename("meter_test_storage.json"),
            ("meter", "test", False),
        )

    def test_supports_legacy_files(self) -> None:
        self.assertEqual(
            parse_storage_filename("building_storage.json"),
            ("building", "-", False),
        )

    def test_rejects_non_storage_files(self) -> None:
        self.assertIsNone(parse_storage_filename("metadata.json"))


class DashboardDataLoadingTests(unittest.TestCase):
    def test_load_all_records_flattens_standard_legacy_and_rl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            self._write_json(
                results_root / "scenario_a" / "run_001" / "fed_alpha" / "hvac_train_storage.json",
                {
                    "time": ["2026-01-01T00:05:00", "2026-01-01T00:00:00"],
                    "inputs": {"hvac.zone_1": {"temperature": [20.0, 19.0]}},
                    "outputs": {"hvac.zone_1": {"power": [1.0, 1.5]}},
                    "params": {"hvac.zone_1": {"setpoint": [21.0, 21.0]}},
                },
            )
            self._write_json(
                results_root / "scenario_a" / "run_001" / "fed_alpha" / "grid_storage.json",
                {
                    "time": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
                    "outputs": {"grid": {"voltage": [230.0, 231.0]}},
                },
            )
            self._write_json(
                results_root / "scenario_a" / "run_001" / "fed_beta" / "agent_test_rl_storage.json",
                {
                    "time": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
                    "observations": {"state_of_charge": [0.4, 0.5]},
                    "actions": {"charge_power": [1.0, 0.0]},
                    "rewards": [0.2, 0.3],
                    "episode_rewards": [1.2, 1.6],
                    "episode_lengths": [12, 10],
                },
            )

            records = load_all_records("scenario_a", "run_001", results_path=results_root)

            self.assertEqual(len(records), 14)
            self.assertIn(
                {
                    "federation": "fed_alpha",
                    "federate": "hvac",
                    "model_instance": "zone_1",
                    "attribute": "temperature",
                    "type": "input",
                    "mode": "train",
                    "time": "2026-01-01T00:05:00",
                    "value": 20.0,
                },
                records,
            )
            self.assertIn(
                {
                    "federation": "fed_beta",
                    "federate": "agent",
                    "model_instance": "agent",
                    "attribute": "reward",
                    "type": "reward",
                    "mode": "test",
                    "time": "2026-01-01T00:05:00",
                    "value": 0.3,
                },
                records,
            )

            dataframe = build_dataframe(records, federations=("fed_alpha",), types=("output",))
            self.assertEqual(list(dataframe["attribute"]), ["voltage", "power", "voltage", "power"])
            self.assertTrue(dataframe["time"].is_monotonic_increasing)

            available_attributes = get_available_tags(
                records,
                "attribute",
                federations=("fed_beta",),
                types=("observation", "reward"),
            )
            self.assertEqual(available_attributes, ["reward", "state_of_charge"])

    def test_load_rl_episode_records_builds_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            self._write_json(
                results_root / "scenario_b" / "run_010" / "fed_rl" / "agent_train_rl_storage.json",
                {
                    "time": ["2026-01-01T00:00:00"],
                    "observations": {"obs": [0.1]},
                    "actions": {"act": [1.0]},
                    "rewards": [0.5],
                    "episode_rewards": [1.0, 1.5, 2.0],
                    "episode_lengths": [10, 9, 8],
                },
            )

            records = load_rl_episode_records("scenario_b", "run_010", results_path=results_root)
            dataframe = build_episode_dataframe(records)

            self.assertEqual(len(records), 3)
            self.assertEqual(list(dataframe["episode"]), [0, 1, 2])
            self.assertEqual(list(dataframe["episode_reward"]), [1.0, 1.5, 2.0])
            self.assertEqual(list(dataframe["episode_length"]), [10, 9, 8])

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
