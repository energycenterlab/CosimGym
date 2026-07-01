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
