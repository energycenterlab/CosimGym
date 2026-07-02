# Handoff — `nonblocking_storage` (Plan 2)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `nonblocking_storage` (created off `main`, independent of `digitaltwin_interfaces`)

> **Process (same as Plan 1):** the agent never runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Plan 2 status: COMPLETE

All milestones done and committed: S0 `9f63cad`, S1 `5e31568`, S2 `a2239bf`,
S3 `2280a0c`, S4 `dd260d8`. All boxes ticked in the plan file. Nothing
staged, nothing pending.

## Summary of what Plan 2 built

`memory_config.sink` (`json` default / `parquet` / `none`) on top of the
existing per-federate results pipeline:
- `json` — unchanged synchronous end-of-run write (today's behavior).
- `parquet` — non-blocking: `update_storage()` enqueues a row per tick to
  `AsyncStorageWriter` (`src/utils/async_storage.py`, bg thread + bounded
  queue that **blocks rather than drops** under backpressure — no data
  loss), which batches rows (`memory_config.batch_size`) and hands each
  batch to `ParquetStorageWriter` (`src/utils/parquet_storage.py`), which
  flattens them into the dashboard's existing tidy schema and writes them
  via `pyarrow.parquet.ParquetWriter`, one row group per batch, finalized at
  run end. Verified value-for-value identical to the JSON sink on a live
  run, and negligible sim-thread cost (~3µs/tick) vs `json`.
- `none` — skip local file storage.
- Dead InfluxDB `flush_storage` path (disabled, unreachable) removed.
- Fully documented in `CLAUDE.md` and `docs/user_guide/` (general.md is the
  canonical `sink` reference; federate.md, dashboard.md, running_scenarios.md
  cross-reference it).

## Benchmark: json vs parquet sink (stress test, pre-merge)

Before merging to `main`, ran a stress comparison: `stress_multi_building_json`
/ `stress_multi_building_parquet` (twins of `multi_building_grid_test`, 50
`rc_building` instances across 5 federates + weather + grid, extended to a
full year hourly = 8760 ticks, `attrs: "all"`). 2 runs each, on this machine
(local SSD, no contention).

**Two real bugs surfaced by the stress test, fixed in this pass:**

1. **`update_storage()` grew the JSON-shaped `self.storage` partitions every
   tick regardless of `sink`** (`BaseFederate.py`, now fixed) — `sink:
   parquet` was building the row dict for the async writer *in addition to*
   the legacy per-instance/per-var Python lists, which were never read
   (`store_local_file()` no-ops for parquet) but never stopped growing
   either. Fixed: JSON partitions are now only populated when
   `sink != 'parquet'`.
2. **`AsyncStorageWriter`'s queue was unbounded in production** — the class
   supports a `maxsize` bound (and its own docstring says storage rows
   "block under backpressure... the queue bound protects memory"), but
   `BaseFederate._enqueue_async_storage_row()` never passed one, so the
   queue was always `maxsize=0` (infinite) — the bound existed in code but
   was never wired up. Fixed: now bounded to `batch_size * 3`.

**Results (after both fixes):**

| Metric | json sink | parquet sink | Notes |
|---|---|---|---|
| `simulation_duration` (8760 ticks) | 246.7s / 248.7s | 252.7s / 246.7s | No meaningful difference — dominated by HELICS/pandapower stepping, not storage I/O, at this scale. |
| Peak RSS — each `building_federate_N` (10 instances) | ~113–115 MB | ~178–185 MB | Parquet is **higher**, not lower. |
| Peak RSS — `grid_federate` / `weather_federate` | ~265–338 MB | ~278–352 MB | Roughly flat between sinks — dominated by pandapower/pandas baseline overhead, not timeseries storage. |
| Result dir size on disk | 30.34 MB | 11.11 MB | Parquet's columnar+compressed long format is ~2.7x smaller than the JSON dump, as expected. |

**Why parquet's per-federate RSS didn't drop after fix #2:** re-ran twice
after bounding the queue (`maxsize=batch_size*3`) — peak RSS barely moved
(182.5→181.5→180.2→183.3 MB across repeats), which rules out an unbounded
Python-side backlog as the dominant cause (that hypothesis predicted a much
larger drop once bounded). The remaining ~65-70MB gap vs. json is most
likely **pyarrow's own C++ memory-pool overhead** (arena allocation that
isn't necessarily released back to the OS between batches) rather than
data actually held live in Python — i.e. a fixed cost of depending on
pyarrow at all, not something that scales with instance/tick count. Fix #2
is still worth keeping (it's a real correctness gap vs. the documented
design — without it, a writer thread that falls badly behind under a much
heavier workload than this one *would* backlog unboundedly), just not the
explanation for what this particular benchmark measured.

**Bottom line for the merge decision:**
- **Benefit of `sink: parquet`, confirmed:** non-blocking incremental writes
  (no single large blocking `json.dump` at run end), ~2.7x smaller result
  files, and (after fix #1) no longer double-storing every tick.
- **Limit, confirmed:** at this scale, parquet sink uses *more* peak memory
  per building-federate process than json (pyarrow's own overhead), and
  wall-clock timing is a wash — it is not a memory-usage win in absolute
  terms, only relative to what it *would* have used pre-fix (double
  storage). Don't oversell it as a memory optimization; the real benefit is
  non-blocking I/O and much smaller result artifacts.
- Both known deferred gaps below (RL federate, dashboard) are unaffected by
  this benchmark and remain out of scope for this merge.

## Deliberately deferred (not part of Plan 2 — need your go-ahead before anyone touches them)

1. **Dashboard Parquet read-support** — `load_all_records()`
   (`src/dashboard/dashboard_data.py`) only globs `*.json`. The `sink:
   parquet` schema was designed to match the dashboard's columns exactly so
   this would be a small addition later, but it wasn't done here since
   you've mentioned wanting to redesign/unify the dashboards — better to
   fold this in there than bolt it on now.
2. **`RL_Federate` Parquet wiring** — `sink: parquet` still raises
   `NotImplementedError` for `type: rl` federates; its storage schema
   (observations/actions/rewards/episodes) differs from `BaseFederate`'s
   and was never wired to the async writer. Needs its own design pass.

## State of the tree

`nonblocking_storage`, 5 commits ahead of `main`: `9f63cad` `5e31568`
`a2239bf` `2280a0c` `dd260d8`. Clean — nothing staged or uncommitted.
Merging into `main` (if/when wanted) is your call, not automatic.

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
**S3 (committed `2280a0c`):** `src/core/BaseFederate.py` (deletion only —
dead `flush_storage`/`infl_client` code removed, no new files).
**S4 (staged):** `src/core/BaseFederate.py` (1-line dead-import removal),
`CLAUDE.md`, `docs/user_guide/scenario_configuration/general.md`,
`docs/user_guide/scenario_configuration/federate.md`,
`docs/user_guide/dashboard.md`, `docs/user_guide/running_scenarios.md`
(all docs-only except the import).

## Blockers / deviations from the plan (kept for reference)

1. **`RL_Federate` still raises `NotImplementedError` for `sink: parquet`**
   — deliberately not wired, documented as a known limitation in both
   `CLAUDE.md` and `general.md`. Left for a future, separately-scoped piece
   of work (RL's storage schema differs from `BaseFederate`'s).
2. **Dashboard has no Parquet read support yet** — deliberately deferred to
   avoid touching dashboard code speculatively; documented explicitly in
   `dashboard.md` so it's not mistaken for a bug. Flag to the user before
   doing this, given their separate stated intent to redesign the
   dashboards.
3. **S1's row shape (nested by entity) was kept as-is** rather than
   flattened earlier in the pipeline — S3's perf check found this isn't a
   hot path, so no change was made.
4. **Value column is a fixed `float64`** — any genuinely non-numeric
   parameter value would silently become `null` in the Parquet output
   (test-covered). Not an issue for any current catalog model.
5. **This branch does not include Plan 1's `digitaltwin_interfaces` work**
   (not merged into `main`) — the two plans are intentionally decoupled.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # nonblocking_storage; clean
git log --oneline -6                       # dd260d8 (S4) 2280a0c (S3) a2239bf (S2) 5e31568 (S1) 9f63cad (S0) 38948b3 (main)

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio (no mosquitto on this branch)

python -m pytest tests/test_async_storage.py tests/test_parquet_storage.py tests/test_rl_config.py -q   # 70 passed, 1 skipped
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py
```

## One-line kickoff prompt for a fresh session

> "Plan 2 (`nonblocking_storage` branch) is complete — see
> `docs/handoffs/nonblocking_storage.md` for what it built and what's
> deliberately deferred (dashboard Parquet read-support, `RL_Federate`
> wiring). If asked to work on either of those, treat them as new,
> separately-scoped tasks, not a continuation of Plan 2's milestone list."
