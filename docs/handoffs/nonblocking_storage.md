# Handoff — `nonblocking_storage` (Plan 2)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `nonblocking_storage` (created off `main`, independent of `digitaltwin_interfaces`)

> **Process (same as Plan 1):** the agent never runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last committed milestone

**S0 — `sink` field on `MemoryConfig`** ✅ ticked. Commit: `9f63cad`.

## Staged, awaiting your commit: S1 — background writer thread + bounded queue

- **New `src/utils/async_storage.py`** — `AsyncStorageWriter`: one background
  drain thread + a `queue.Queue`, batching rows by `batch_size` before
  calling a pluggable `on_batch(batch)` callback. **Deliberately does not
  write Parquet** — that callback is a placeholder until S2 wires pyarrow;
  this class only owns the thread/queue/batching mechanics (S1's scope).
  Key design choice, called out in the module docstring: unlike Plan 1's MQTT
  outbound queue (telemetry, drop-oldest is fine), `enqueue()` **blocks**
  under backpressure instead of dropping — losing a result row would be a
  correctness bug, not a UX nuisance. `close()` flushes any trailing partial
  batch and joins the thread, so nothing is lost at shutdown.
- **`src/core/BaseFederate.py`**:
  - `__init__`: `self._async_storage_writer = None`, `self._async_rows_flushed = 0`.
  - `update_storage()`: when `self.config.memory_config.sink == 'parquet'`,
    builds a per-tick row snapshot (`{ts, time, mode, inputs, outputs, params}`,
    nested by entity — a simple placeholder shape; S2 will decide the actual
    flat/columnar schema when it designs the pyarrow write) **alongside** the
    existing in-memory `self.storage` append (unchanged for `sink: json`),
    and hands it to `_enqueue_async_storage_row()`, which lazily creates and
    starts the writer on first use.
  - `_on_storage_batch(batch)`: S1's placeholder consumer — just counts and
    logs (`AsyncStorageWriter: flushed batch of N rows (total M) — pyarrow
    write not implemented yet (S2)`). No file I/O yet.
  - `run()`: closes the async writer (flushing any remainder) **before**
    calling `store_local_file()` — so by the time `store_local_file` raises
    its `NotImplementedError` for `sink: parquet` (unchanged from S0), every
    row from the run has already been drained through the queue with zero
    loss. The external contract (parquet still hard-fails at run's end) is
    intentionally **unchanged** — S1 only proves the plumbing underneath it.
  - **`RL_Federate.update_storage`/`run` were NOT wired in this milestone** —
    its storage partition schema (observations/actions/rewards/episodes) is
    structurally different from `BaseFederate`'s (inputs/outputs/params), so
    it needs its own row-builder design rather than reusing this one as-is.
    Deferred to when S2 designs the actual Parquet schema (a natural point to
    decide both federate types' row shapes together) — flagged here so it
    isn't forgotten, not because it's out of scope for Plan 2 overall.
- **New `tests/test_async_storage.py`** (6 tests, isolated — no HELICS):
  batching triggers at `batch_size`; zero data loss across 1000 rapid
  enqueues; `close()` flushes a trailing partial batch; `close()` is
  idempotent; producer isn't meaningfully slowed by the drain thread; an
  exception inside `on_batch` is logged but doesn't kill the drain thread
  (subsequent batches still process).

**Verified:**
- `pytest tests/test_async_storage.py -v` → 6 passed.
- `pytest tests/test_rl_config.py -q` → 58 passed, 1 skipped (unaffected —
  no config schema changes this milestone).
- Regression: `python src/test_script.py` green (`sink: json` default path,
  byte-for-byte unaffected by the new `parquet`-only code path).
- **Runtime integration check**: scratch scenario (`spring_mass_damper`,
  single federate, 30 ticks, `memory_config: {sink: parquet, batch_size: 7}`).
  Federate log shows 4 full batches of 7 + one trailing batch of 2 via
  `close()` = **30 rows total, exactly matching the 30 ticks — zero loss**.
  The run still ends with the same `NotImplementedError` from `store_local_file`
  as it did before this milestone (S0's contract, unchanged), confirmed via
  the federate's `.log` traceback.

**Your action:** review the staged diff, commit (e.g. `feat(storage): S1
background writer thread + bounded queue fed from update_storage`), then say
continue — I'll tick S1 and move to S2.

## Next step (after you commit S1)

**S2 — Parquet sink via pyarrow (batched), same `results/` layout.** Wire a
real pyarrow writer as the `on_batch` callback passed into
`AsyncStorageWriter` (replacing/extending `_on_storage_batch`'s placeholder
in `BaseFederate`), writing to
`results/<scenario>/<sim_id>/<federation>/<federate>_<mode>_storage.parquet`
(or `.parquet`-per-batch files merged, or an Arrow dataset — this is the
actual design decision S2 needs to make; the plan says "same `results/`
layout used today" and notes `dashboard_parquet_cache.py` is "already
Parquet-based," worth reading first to align schemas). Concretely:
1. Decide the on-disk row/column schema (the current S1 row shape is nested
   by entity — flatten it however's most convenient for pyarrow + the
   dashboard reader).
2. Replace `store_local_file`'s `NotImplementedError` for `sink == 'parquet'`
   with the real write path (probably: nothing to do there anymore, since
   the async writer already wrote everything incrementally via `on_batch` —
   `store_local_file` for `sink: parquet` should become close to a no-op,
   maybe just closing out any final metadata).
3. Also wire `RL_Federate` (see the note above — deferred from S1 for this
   exact reason).
4. Verify the dashboard renders `sink: parquet` results identically to
   `sink: json` ones for the same scenario.

First concrete action: read `src/dashboard/dashboard_parquet_cache.py` to see
the Parquet schema/layout it already expects, so S2's writer produces
directly-compatible files rather than needing a translation step.

## Files touched so far

**S0 (committed `9f63cad`):** `src/utils/config_dataclasses.py`,
`src/core/BaseFederate.py`, `src/core/RL_Federate.py`, `tests/test_rl_config.py`.
**S1 (staged):** `src/core/BaseFederate.py` (further changes), new
`src/utils/async_storage.py`, new `tests/test_async_storage.py`.

## State of the tree

On `nonblocking_storage`, 1 commit ahead of `main` (`9f63cad`). S1's changes
are `git add`ed but **uncommitted**, waiting on you.

## Blockers / deviations from the plan

1. **`RL_Federate` async wiring deferred to S2** (see above) — its storage
   schema differs enough from `BaseFederate`'s that reusing S1's row-builder
   as-is would be the wrong shape; better decided alongside S2's actual
   Parquet schema design.
2. **`memory_config.sink='parquet'` still hard-fails** at the very end of a
   run (`store_local_file`'s `NotImplementedError`, unchanged from S0) even
   though S1 now silently drains and counts every row underneath that. This
   is intentional — the external contract (no working Parquet output yet)
   hasn't changed, only the internal plumbing feeding the eventual writer.
3. **S1's row shape is a placeholder** (`{ts, time, mode, inputs: {entity:
   {var: value}}, outputs: {...}, params: {...}}`) — nested by entity, not
   flattened to columns. S2 should feel free to redesign this entirely when
   it builds the real pyarrow schema; nothing downstream depends on this
   shape yet (the `on_batch` callback is the only consumer, and it's a
   throwaway counter/logger in S1).
4. **This branch does not include Plan 1's `digitaltwin_interfaces` work**
   (not merged into `main`) — the two plans are intentionally decoupled.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # nonblocking_storage; S1 staged, not committed
git log --oneline -3                       # 9f63cad (S0) on top of 38948b3 (main)
git diff --staged --stat                   # S1's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio (no mosquitto on this branch)

# Unit tests for the new plumbing (fast, no HELICS):
python -m pytest tests/test_async_storage.py -v   # 6 passed

# Config gate + regression — must be unaffected (sink defaults to json):
python -m pytest tests/test_rl_config.py -q       # 58 passed, 1 skipped
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# Runtime integration check (sink: parquet) — build a small scratch scenario
# (spring_mass_damper, single federate) with memory_config: {sink: parquet,
# batch_size: <small>}, then:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('/absolute/path/to/scratch.yaml')"
# check the federate's .log: batches should sum to exactly the tick count
# (zero loss), and the run should still end with the same NotImplementedError
# as S0 (external contract unchanged).
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> (Plan 2 section) and `docs/handoffs/nonblocking_storage.md`. We're on branch
> `nonblocking_storage`. S0 and S1 are implemented; S1 (background writer
> thread + bounded queue, `src/utils/async_storage.py`) is staged, not
> committed — review and commit it yourself first, then tell the agent to
> tick S1 and continue to S2 (real Parquet writer via pyarrow, replacing S1's
> placeholder `on_batch` counter). Follow the per-milestone loop: implement,
> run the milestone's check + regression tests, stage (don't commit), update
> the handoff doc, then stop and wait for the user's commit + 'continue'
> signal before ticking the box and starting S3."
