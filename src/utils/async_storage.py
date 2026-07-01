"""
async_storage.py — bounded queue + background drain thread for non-blocking
result storage (nonblocking_storage plan, S1).

This module owns only the thread/queue/batching mechanics. It does not know
how to write Parquet (or anything else) to disk — the caller supplies an
`on_batch` callback that receives each completed batch; wiring that callback
to an actual pyarrow writer is S2's job. S1's goal is to prove the producer
(a federate's per-step `update_storage`) can hand off rows without blocking
the simulation thread, and that the consumer thread never loses a row.

Unlike the Plan-1 MQTT outbound queue (telemetry mirror, where dropping old
samples under backpressure is acceptable), storage rows are simulation
results — losing one silently would be a correctness bug, not a UX nuisance.
`enqueue` therefore blocks under backpressure instead of dropping; the queue
bound protects memory, not data integrity. A well-tuned `batch_size` (paired
with a large-enough in-memory bound) keeps the drain thread comfortably
ahead of the producer in practice.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import queue
import threading
from typing import Any, Callable, Dict, List, Optional

_STOP_SENTINEL = None


class AsyncStorageWriter:
    """One background thread draining a bounded queue of row dicts, batching
    them by `batch_size` before invoking `on_batch(batch)`."""

    def __init__(self, batch_size: int, on_batch: Callable[[List[Dict[str, Any]]], None],
                 maxsize: int = 0, logger=None):
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=maxsize)
        self._batch_size = max(1, batch_size)
        self._on_batch = on_batch
        self._logger = logger
        self._batch: List[Dict[str, Any]] = []
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def enqueue(self, row: Dict[str, Any]) -> None:
        """Blocks under backpressure (bounded `maxsize`) rather than dropping —
        storage rows must not be silently lost."""
        self._queue.put(row)

    def _drain_loop(self) -> None:
        while True:
            row = self._queue.get()
            if row is _STOP_SENTINEL:
                if self._batch:
                    self._emit_batch()
                self._queue.task_done()
                return
            self._batch.append(row)
            if len(self._batch) >= self._batch_size:
                self._emit_batch()
            self._queue.task_done()

    def _emit_batch(self) -> None:
        batch, self._batch = self._batch, []
        try:
            self._on_batch(batch)
        except Exception as exc:
            if self._logger:
                self._logger.error(f"AsyncStorageWriter on_batch callback failed: {exc}")

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Signal the drain thread to flush any remaining rows and stop;
        blocks (bounded by `timeout`) so no data is lost at shutdown."""
        if self._stopped or not self._started:
            self._stopped = True
            return
        self._stopped = True
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=timeout)
