"""
parquet_storage.py — pyarrow-backed batch writer for `memory_config.sink='parquet'`
(nonblocking_storage plan, S2).

Consumes the row batches produced by `utils.async_storage.AsyncStorageWriter` (S1)
— one row per tick, nested by entity:
`{time, mode, inputs: {entity: {var: value}}, outputs: {...}, params: {...}}`
— and flattens each into the SAME long/tidy schema
`dashboard_data.load_all_records` already produces from the JSON sink
(`TIME_SERIES_COLUMNS`: time, federation, federate, model_instance, attribute,
type, mode, value). Matching that schema exactly is what lets the dashboard
consume Parquet results with a minimal addition rather than a new format.

Writes to the same `results/<scenario>/<sim_id>/<federation>/` layout as the
JSON sink, one file per mode: `<federate>_<mode>_storage.parquet`. A
`pyarrow.parquet.ParquetWriter` is opened lazily per mode on first batch and
kept open — one row group appended per `on_batch` call (the "batched" write
the plan asks for) — until `close()`.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import os
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

_SECTION_TYPE = {'inputs': 'input', 'outputs': 'output', 'params': 'param'}

SCHEMA = pa.schema([
    ('time', pa.string()),
    ('federation', pa.string()),
    ('federate', pa.string()),
    ('model_instance', pa.string()),
    ('attribute', pa.string()),
    ('type', pa.string()),
    ('mode', pa.string()),
    ('value', pa.float64()),
])


def _coerce_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ParquetStorageWriter:
    """One instance per federate. Routes each batch's rows to a per-mode
    `ParquetWriter` (lazily opened), so `train`/`test` land in separate files
    exactly like the JSON sink's `<federate>_<mode>_storage.json` files."""

    def __init__(self, base_dir: str, federate_name: str, federation_name: str, logger=None):
        self._base_dir = base_dir
        self._federate_name = federate_name
        self._federation_name = federation_name
        self._logger = logger
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._paths: Dict[str, str] = {}
        self._row_counts: Dict[str, int] = {}

    def on_batch(self, batch: List[dict]) -> None:
        by_mode: Dict[str, List[dict]] = {}
        for row in batch:
            mode = row.get('mode') or 'test'
            time_value = str(row.get('time'))
            for section, kind in _SECTION_TYPE.items():
                for entity_id, variables in row.get(section, {}).items():
                    federate_name, _, model_instance = entity_id.partition('.')
                    for attribute, value in variables.items():
                        by_mode.setdefault(mode, []).append({
                            'time': time_value,
                            'federation': self._federation_name,
                            'federate': federate_name,
                            'model_instance': model_instance,
                            'attribute': attribute,
                            'type': kind,
                            'mode': mode,
                            'value': _coerce_value(value),
                        })

        for mode, records in by_mode.items():
            if not records:
                continue
            table = pa.Table.from_pylist(records, schema=SCHEMA)
            writer = self._writers.get(mode)
            if writer is None:
                os.makedirs(self._base_dir, exist_ok=True)
                path = os.path.join(self._base_dir, f"{self._federate_name}_{mode}_storage.parquet")
                writer = pq.ParquetWriter(path, SCHEMA)
                self._writers[mode] = writer
                self._paths[mode] = path
                self._row_counts[mode] = 0
            writer.write_table(table)
            self._row_counts[mode] += len(records)

    def close(self) -> None:
        for mode, writer in self._writers.items():
            writer.close()
            if self._logger:
                self._logger.info(
                    f"Parquet storage ({mode}) saved to {self._paths[mode]} "
                    f"({self._row_counts[mode]} rows)"
                )
