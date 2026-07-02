"""
test_parquet_storage.py — unit tests for ParquetStorageWriter (nonblocking_storage plan, S2).

Run: pytest tests/test_parquet_storage.py -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.parquet_storage import ParquetStorageWriter


def _row(ts, mode="test", inputs=None, outputs=None, params=None):
    return {
        "ts": ts, "time": f"2024-01-01T00:00:{ts:02d}", "mode": mode,
        "inputs": inputs or {}, "outputs": outputs or {}, "params": params or {},
    }


class TestParquetStorageWriter:

    def test_writes_valid_parquet_with_expected_schema(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="spring_federate",
                                       federation_name="federation_1")
        batch = [
            _row(0, outputs={"spring_federate.0": {"position": 0.0, "velocity": 0.0}}),
            _row(1, outputs={"spring_federate.0": {"position": 0.01, "velocity": 0.19}}),
        ]
        writer.on_batch(batch)
        writer.close()

        path = tmp_path / "spring_federate_test_storage.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert set(df.columns) == {"time", "federation", "federate", "model_instance",
                                    "attribute", "type", "mode", "value"}
        assert len(df) == 4  # 2 ticks x 2 attributes
        assert set(df["attribute"]) == {"position", "velocity"}
        assert set(df["type"]) == {"output"}
        assert (df["federate"] == "spring_federate").all()
        assert (df["model_instance"] == "0").all()

    def test_separate_files_per_mode(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="fed",
                                       federation_name="fedn")
        writer.on_batch([
            _row(0, mode="train", outputs={"fed.0": {"x": 1.0}}),
            _row(1, mode="test", outputs={"fed.0": {"x": 2.0}}),
        ])
        writer.close()

        assert (tmp_path / "fed_train_storage.parquet").exists()
        assert (tmp_path / "fed_test_storage.parquet").exists()
        train_df = pd.read_parquet(tmp_path / "fed_train_storage.parquet")
        test_df = pd.read_parquet(tmp_path / "fed_test_storage.parquet")
        assert len(train_df) == 1
        assert len(test_df) == 1

    def test_multiple_batches_accumulate_as_row_groups(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="fed",
                                       federation_name="fedn")
        for i in range(5):
            writer.on_batch([_row(i, outputs={"fed.0": {"x": float(i)}})])
        writer.close()

        df = pd.read_parquet(tmp_path / "fed_test_storage.parquet")
        assert len(df) == 5
        assert sorted(df["value"].tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_non_numeric_value_coerced_to_none(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="fed",
                                       federation_name="fedn")
        writer.on_batch([_row(0, params={"fed.0": {"solver": "rk4"}})])
        writer.close()

        df = pd.read_parquet(tmp_path / "fed_test_storage.parquet")
        assert len(df) == 1
        assert df["value"].isna().all()
        assert df["attribute"].iloc[0] == "solver"

    def test_empty_batch_is_a_no_op(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="fed",
                                       federation_name="fedn")
        writer.on_batch([_row(0)])  # no inputs/outputs/params -> nothing to flatten
        writer.close()
        # no rows ever produced -> no file should have been created
        assert not (tmp_path / "fed_test_storage.parquet").exists()

    def test_close_without_any_batch_does_not_raise(self, tmp_path):
        writer = ParquetStorageWriter(base_dir=str(tmp_path), federate_name="fed",
                                       federation_name="fedn")
        writer.close()  # must not raise even though nothing was ever written
