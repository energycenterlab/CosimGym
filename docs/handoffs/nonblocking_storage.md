# Handoff — `nonblocking_storage` (Plan 2)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `nonblocking_storage` (created off `main`, independent of `digitaltwin_interfaces`)

> **Process (same as Plan 1):** the agent never runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last committed milestone

**S1 — background writer thread + bounded queue** ✅ ticked. Commit: `5e31568`.
(Earlier: S0 `9f63cad`.)

## Staged, awaiting your commit: S2 — Parquet sink via pyarrow

- **New `src/utils/parquet_storage.py`** — `ParquetStorageWriter`: consumes
  the row batches S1's `AsyncStorageWriter` produces (nested by entity) and
  flattens each into the **same long/tidy schema**
  `dashboard_data.load_all_records()` already builds from JSON
  (`time, federation, federate, model_instance, attribute, type, mode, value`
  — `dashboard_data.py:15-24`'s `TIME_SERIES_COLUMNS`). Matching that schema
  exactly (not inventing a new one) is deliberate — it's what would let the
  dashboard read these files with a minimal addition later (see "Deferred"
  below). Writes one `pyarrow.parquet.ParquetWriter` per **mode**, opened
  lazily on first batch and kept open — one row group per `on_batch` call —
  to `results/<scenario>/<sim_id>/<federation>/<federate>_<mode>_storage.parquet`,
  mirroring the JSON sink's per-mode file split. `close()` finalizes every
  open writer (required — an unclosed `ParquetWriter` produces a corrupt/
  unreadable file, no footer written). Non-numeric values (there aren't any
  in practice today, but `model.state.parameters` is untyped) are coerced to
  `None` rather than breaking the fixed `float64` schema.
- **`src/core/BaseFederate.py`**:
  - `_enqueue_async_storage_row()`: now creates a real `ParquetStorageWriter`
    alongside the `AsyncStorageWriter` (replacing S1's placeholder
    `_on_storage_batch` counter, which is now deleted) and wires
    `ParquetStorageWriter.on_batch` as the queue's batch callback.
  - New `_results_base_dir()` helper — the `results/<scenario>/<sim_id>/
    <federation>/` path, factored out of `store_local_file` so both it and
    the new Parquet writer compute the identical path from
    `self.simulation_id`/`self.federation_name`.
  - `run()`: after `self._async_storage_writer.close()` (drains remaining
    queued rows), now also calls `self._parquet_storage_writer.close()` to
    finalize the Parquet file(s) — **must** happen before
    `store_local_file()`, which for `sink: parquet` no longer raises
    `NotImplementedError` (that was S0/S1's placeholder contract) — it now
    just logs that the data was already flushed incrementally and returns
    (all the real writing already happened via `on_batch` during the run).
  - **`RL_Federate` is unchanged this milestone** — its `store_local_file`
    still raises the S0 `NotImplementedError` for `sink: parquet`, exactly
    as before. This is correct/safe, not a bug: since `RL_Federate.
    update_storage` was never wired to the async writer (deferred at S1),
    if it silently "succeeded" now it would silently produce *no* file at
    all (neither JSON nor Parquet) — a real data-loss bug. Keeping the hard
    failure there until RL's own wiring is designed is intentional.
  - `environment.yml`: added `pyarrow>=14.0.0` as a direct conda dependency
    (previously only pulled in transitively for the dashboard's own use of
    `pandas(engine="pyarrow")`; core simulation code now imports it directly
    when `sink: parquet` is used).
- **New `tests/test_parquet_storage.py`** (6 tests, isolated — no HELICS,
  uses `tmp_path`): correct schema/columns; separate files per mode
  (train/test); multiple batches accumulate as row groups in one file;
  non-numeric values coerce to `None` without breaking the schema; an empty
  batch (no inputs/outputs/params for that tick) is a no-op, no file created;
  `close()` without ever writing a batch doesn't raise.

**Verified:**
- `pytest tests/test_parquet_storage.py -v` → 6 passed.
- `pytest tests/test_rl_config.py tests/test_async_storage.py
  tests/test_parquet_storage.py -q` → 70 passed, 1 skipped.
- Regression: `python src/test_script.py` and
  `OMP_NUM_THREADS=1 python src/test_script_rl.py` both green — `sink: json`
  (the default, used by every existing scenario) is byte-for-byte unaffected.
- **Runtime integration check, the important one**: ran the same scratch
  scenario (`spring_mass_damper`, single federate, 30 ticks, `batch_size: 7`)
  twice — once with `sink: json`, once with `sink: parquet` — and directly
  compared the two outputs:
  - Parquet file exists, is valid, has the expected schema, **60 rows**
    (30 ticks × 2 attributes: position, velocity) — logged as
    `"Parquet storage (test) saved to .../spring_federate_test_storage.parquet
    (60 rows)"`.
  - `position`/`velocity` timeseries read back from the Parquet file via
    `pd.read_parquet` are **identical, value-for-value**, to the same
    timeseries read from the JSON file from the `sink: json` run.
  - No `NotImplementedError` — `store_local_file` logged
    `"memory_config.sink='parquet' — storage already flushed via
    ParquetStorageWriter"` and returned cleanly.

**Your action:** review the staged diff, commit (e.g. `feat(storage): S2
Parquet sink via pyarrow, matching the dashboard's tidy schema`), then say
continue — I'll tick S2 and move to S3.

## Next step (after you commit S2)

**S3 — Remove/supersede dead `flush_storage`; perf check.** Per the plan:
`BaseFederate.flush_storage` (the disabled, "too slow" InfluxDB path,
`BaseFederate.py` — search `flush_storage`) is dead code now fully
superseded by the S1/S2 async Parquet path; remove it (it's never called —
confirm with a grep for callers before deleting) or explicitly deprecate it
if something still references it. Then measure per-step wall-time
with/without the async writer (`sink: json` vs `sink: parquet`, same
scenario) to confirm negligible sim-thread impact, per the plan's
"Verification" section — the design rationale (in-process queue, GIL
released during pyarrow I/O) predicts this, but S3 should actually measure it
rather than assume it.

**Deferred work worth flagging (not S3's job, but adjacent):**
- **Dashboard read-support for `.parquet` result files** was NOT added this
  milestone. `dashboard_data.load_all_records()` (`src/dashboard/
  dashboard_data.py:112`) currently only globs `*.json`. Because S2's schema
  was deliberately designed to match `TIME_SERIES_COLUMNS` exactly, adding
  parquet read support later should be a small, low-risk change (glob
  `*_storage.parquet` too, `pd.read_parquet(...).to_dict("records")`,
  `extend()` into the same `records` list) — but it's a dashboard-code change
  outside Plan 2's explicit milestone list, and the user has separately
  mentioned wanting to redesign/unify the dashboards soon (see commit
  `a135355` on `digitaltwin_interfaces`), so deliberately left this for a
  dedicated pass rather than touching dashboard code speculatively here.
  Flag to the user before doing this, since it might best happen alongside
  that broader dashboard redesign rather than as a bolt-on now.
- **`RL_Federate` Parquet wiring** — still deferred (see S1's note, repeated
  above): needs its own row-builder for its different storage schema
  (observations/actions/rewards/episodes vs. inputs/outputs/params).

## Files touched so far

**S0 (committed `9f63cad`):** `src/utils/config_dataclasses.py`,
`src/core/BaseFederate.py`, `src/core/RL_Federate.py`, `tests/test_rl_config.py`.
**S1 (committed `5e31568`):** `src/core/BaseFederate.py` (further changes),
new `src/utils/async_storage.py`, new `tests/test_async_storage.py`.
**S2 (staged):** `src/core/BaseFederate.py` (further changes),
`environment.yml` (+pyarrow), new `src/utils/parquet_storage.py`, new
`tests/test_parquet_storage.py`.

## State of the tree

On `nonblocking_storage`, 2 commits ahead of `main` (`9f63cad`, `5e31568`).
S2's changes are `git add`ed but **uncommitted**, waiting on you.

## Blockers / deviations from the plan

1. **`RL_Federate` still raises `NotImplementedError` for `sink: parquet`**
   (unchanged since S0) — deliberately not wired this milestone; see above.
2. **Dashboard has no Parquet read support yet** — deliberately deferred to
   avoid touching dashboard code speculatively; flag to the user first (see
   above) since they've mentioned a broader dashboard redesign.
3. **S1's row shape (nested by entity) was kept as-is** rather than
   flattened earlier in the pipeline — `ParquetStorageWriter.on_batch` does
   the flattening. Fine for now; if profiling in S3 shows the flattening
   itself is a hot path, it could move earlier (into `update_storage`
   directly), but that's a perf-driven decision for S3, not S2.
4. **Value column is a fixed `float64`** — any genuinely non-numeric
   parameter value would silently become `null` in the Parquet output (test-
   covered: `test_non_numeric_value_coerced_to_none`). Not an issue for any
   current model in the catalog (parameters are numeric), but worth knowing
   if a future model stores a string/categorical parameter.
5. **This branch does not include Plan 1's `digitaltwin_interfaces` work**
   (not merged into `main`) — the two plans are intentionally decoupled.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # nonblocking_storage; S2 staged, not committed
git log --oneline -4                       # 5e31568 (S1), 9f63cad (S0), 38948b3 (main)
git diff --staged --stat                   # S2's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio (no mosquitto on this branch)

# Unit tests for the new plumbing (fast, no HELICS):
python -m pytest tests/test_async_storage.py tests/test_parquet_storage.py -v   # 12 passed

# Config gate + regression — must be unaffected (sink defaults to json):
python -m pytest tests/test_rl_config.py -q       # 58 passed, 1 skipped
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# Runtime integration check (sink: parquet vs sink: json equivalence) — build
# a small scratch scenario (spring_mass_damper, single federate), run it once
# with each sink value, then:
python -c "
import json, pandas as pd
j = json.load(open('results/<scenario>/<json_sim_id>/federation_1/spring_federate_test_storage.json'))
df = pd.read_parquet('results/<scenario>/<parquet_sim_id>/federation_1/spring_federate_test_storage.parquet')
pos_pq = df[df['attribute']=='position'].sort_values('time')['value'].tolist()
assert j['outputs']['spring_federate.0']['position'] == pos_pq
"
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> (Plan 2 section) and `docs/handoffs/nonblocking_storage.md`. We're on branch
> `nonblocking_storage`. S0-S2 are implemented; S2 (real Parquet writer via
> pyarrow, `src/utils/parquet_storage.py`) is staged, not committed — review
> and commit it yourself first, then tell the agent to tick S2 and continue
> to S3 (remove dead `flush_storage`, measure sim-thread perf impact with/
> without the async writer). Follow the per-milestone loop: implement, run
> the milestone's check + regression tests, stage (don't commit), update the
> handoff doc, then stop and wait for the user's commit + 'continue' signal
> before ticking the box and starting S4 (docs)."
