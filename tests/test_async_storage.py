"""
test_async_storage.py — unit tests for AsyncStorageWriter (nonblocking_storage plan, S1).

Run: pytest tests/test_async_storage.py -v
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.async_storage import AsyncStorageWriter


def _make_writer(batch_size=10, maxsize=0):
    batches = []
    lock = threading.Lock()

    def on_batch(batch):
        with lock:
            batches.append(list(batch))

    writer = AsyncStorageWriter(batch_size=batch_size, on_batch=on_batch, maxsize=maxsize)
    writer.start()
    return writer, batches


class TestAsyncStorageWriter:

    def test_batches_flush_at_batch_size(self):
        writer, batches = _make_writer(batch_size=5)
        for i in range(12):
            writer.enqueue({"i": i})
        writer.close(timeout=5)
        # 12 rows, batch_size=5 -> two full batches of 5 flushed during the run,
        # plus one trailing partial batch of 2 flushed by close().
        assert sum(len(b) for b in batches) == 12
        assert len(batches) == 3
        assert [len(b) for b in batches] == [5, 5, 2]

    def test_no_data_loss_under_many_rows(self):
        writer, batches = _make_writer(batch_size=7)
        n = 1000
        for i in range(n):
            writer.enqueue({"i": i})
        writer.close(timeout=10)
        all_rows = [row["i"] for batch in batches for row in batch]
        assert sorted(all_rows) == list(range(n))

    def test_close_flushes_trailing_partial_batch(self):
        writer, batches = _make_writer(batch_size=100)
        writer.enqueue({"i": 1})
        writer.enqueue({"i": 2})
        # far short of batch_size=100 -- only close() should flush these
        writer.close(timeout=5)
        assert sum(len(b) for b in batches) == 2

    def test_close_is_idempotent(self):
        writer, batches = _make_writer(batch_size=10)
        writer.enqueue({"i": 1})
        writer.close(timeout=5)
        writer.close(timeout=5)  # must not hang or raise
        assert sum(len(b) for b in batches) == 1

    def test_enqueue_does_not_block_producer_for_reasonable_batch_size(self):
        writer, batches = _make_writer(batch_size=50)
        start = time.time()
        for i in range(500):
            writer.enqueue({"i": i})
        elapsed = time.time() - start
        writer.close(timeout=10)
        assert elapsed < 2.0  # producer should never meaningfully wait on the drain thread here
        assert sum(len(b) for b in batches) == 500

    def test_on_batch_exception_does_not_kill_drain_thread(self):
        seen = []

        def flaky_on_batch(batch):
            seen.append(batch)
            if len(seen) == 1:
                raise RuntimeError("boom")

        writer = AsyncStorageWriter(batch_size=2, on_batch=flaky_on_batch)
        writer.start()
        writer.enqueue({"i": 1})
        writer.enqueue({"i": 2})  # triggers the raising batch
        writer.enqueue({"i": 3})
        writer.enqueue({"i": 4})  # should still be processed despite the prior exception
        writer.close(timeout=5)
        assert len(seen) == 2
        assert sum(len(b) for b in seen) == 4
