# Handoff — `nonblocking_storage` (Plan 2)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `nonblocking_storage` (created off `main`, independent of `digitaltwin_interfaces`)

> **Process (same as Plan 1):** the agent never runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last committed milestone

**S3 — removed dead `flush_storage`/InfluxDB path; perf check** ✅ ticked.
Commit: `2280a0c`. (Earlier: S2 `a2239bf`, S1 `5e31568`, S0 `9f63cad`.)

## Staged, awaiting your commit: S4 — docs (final Plan 2 milestone)

Docs-only, plus one tiny leftover cleanup found while reviewing S3's diff.

- **`src/core/BaseFederate.py`**: removed a stray unused
  `from utils.influxdb_client import InfluxClient` import — dead since S3
  deleted `flush_storage` (its only user); the module itself
  (`src/utils/influxdb_client.py`) is untouched and still documented as an
  optional/legacy path in `dashboard.md`. 1 line removed.
- **`CLAUDE.md`**: rewrote the "Storage & Results" section to document all
  three `sink` values (`json`/`parquet`/`none`), how the async Parquet path
  works (`AsyncStorageWriter` → `ParquetStorageWriter`), the `RL_Federate`
  `sink: parquet` limitation, and that the dashboard only reads JSON today.
  Added `memory_config.sink` and `memory_config.batch_size` to the Config
  Reference bullet list.
- **`docs/user_guide/scenario_configuration/general.md`**: this is the
  canonical per-field `memory_config` reference — added the `sink` field to
  its table and a full "sink options" subsection (json/parquet/none, what
  each does, the RL and dashboard-read caveats). Clarified `batch_size`'s
  meaning now differs by sink (buffer size for json; actual flush
  granularity for parquet).
- **`docs/user_guide/scenario_configuration/federate.md`**: added `sink:
  parquet` to the per-federate `memory_config` override example, with a
  one-line note on why you'd isolate one federate onto it, and a link to
  `general.md` for full semantics.
- **`docs/user_guide/dashboard.md`**: reworked "The Data Pipeline" from a
  2-item to a 4-item list — explicitly separated the dashboard's own
  **read-side** Parquet cache (`dashboard_parquet_cache.py`, unrelated,
  pre-existing) from the new **federate-level** `sink: parquet` write path,
  since these are two different things that both involve "Parquet" and are
  easy to conflate. States plainly that the dashboard doesn't read
  `sink: parquet` output yet.
- **`docs/user_guide/running_scenarios.md`**: updated the "Where output
  goes" bullet to mention both sink formats and link to the full reference.
- **Not touched (correctly, per the plan)**: `README.md` has no
  storage/results content to update; `docs/overview/terminology.md`'s one
  `memory_config` mention is generic enough to stay as-is.

**Verified:**
- `pytest tests/test_rl_config.py tests/test_async_storage.py
  tests/test_parquet_storage.py -q` → 70 passed, 1 skipped (docs-only + one
  dead-import removal — no behavior change expected or seen).
- `python src/test_script.py` → green.

**Your action:** review the staged diff, commit (e.g. `docs(storage):
document memory_config.sink (json/parquet/none) across CLAUDE.md and
user_guide`), then say continue — I'll tick S4. **This is the last
milestone in Plan 2** — once S4 is ticked, Plan 2 is complete.

## Next step (after you commit S4)

Plan 2 (`nonblocking_storage`) is done — S0 through S4 all ticked. No
further milestones. Two items were deliberately deferred throughout (not
part of Plan 2's scope, flagged for a future decision, not a TODO to pick up
automatically):
1. Dashboard read-support for `.parquet` result files (schema already
   matches, low-risk to add later, but deliberately left for you to decide
   given your separate plan to redesign the dashboards).
2. `RL_Federate` wiring to the async/Parquet path (different storage
   schema, needs its own design).
If/when you want to merge this branch into `main`, that's a decision for
you to make explicitly — not something to do automatically at Plan 2's end.

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

## State of the tree

On `nonblocking_storage`, 4 commits ahead of `main` (`9f63cad`, `5e31568`,
`a2239bf`, `2280a0c`). S4's change is `git add`ed but **uncommitted**,
waiting on you. **Once committed, Plan 2 is fully complete (S0-S4 all
ticked).**

## Blockers / deviations from the plan

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
git status && git branch --show-current   # nonblocking_storage; S4 staged, not committed
git log --oneline -5                       # 2280a0c (S3), a2239bf (S2), 5e31568 (S1), 9f63cad (S0), 38948b3 (main)
git diff --staged --stat                   # S4's staged changes (docs + 1-line import removal)

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio (no mosquitto on this branch)

python -m pytest tests/test_async_storage.py tests/test_parquet_storage.py tests/test_rl_config.py -q   # 70 passed, 1 skipped
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> (Plan 2 section) and `docs/handoffs/nonblocking_storage.md`. We're on branch
> `nonblocking_storage`. S0-S4 are implemented; S4 (docs — `memory_config.sink`
> reference across `CLAUDE.md` and `docs/user_guide/`, plus a 1-line dead
> import cleanup) is staged, not committed — review and commit it yourself
> first, then tell the agent to tick S4. **This is the last milestone — Plan
> 2 is then complete.** Two items were deliberately deferred throughout (not
> automatic follow-ups): dashboard Parquet read-support, and `RL_Federate`
> Parquet wiring — both documented, both need your explicit go-ahead before
> anyone picks them up."
