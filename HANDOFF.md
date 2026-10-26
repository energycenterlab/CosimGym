# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session)

1. Realign docs with the author's correct premises on reset/restart and EnergyPlus (done).
2. Plan objective 1 — model reset refactor (done, approved). **Execution plan:**
   `docs/changes_reports/model_reset_redesign_plan.md` — phases F0..F6, one session each,
   each with its own file list and done-criteria. **F0–F4 DONE. Next session: F5, then F6.**

Execution setup chosen by the author: Sonnet executes, Opus as advisor (advisor at start and
before closing each phase). Each phase ends with: plan doc `STATUS` updated, `graphify
update .`, HANDOFF rewritten, commit proposed (commit only on explicit ok).

## Ground truth (stated by the author — do not contradict it)

- `BaseFMUModel` wraps **any** FMU, from any engine, modelling anything. EnergyPlus FMUs
  are only the most used; do not generalise EnergyPlus facts to all FMUs.
- Two **independent** concepts:
  - **RL episode reset** (`full` / `rolling` / `none`) — only matters in RL runs; resets
    the model between episodes.
  - **Model horizon restart** — any model with an intrinsic limit (a CSV covering one year,
    an EnergyPlus run period of a few months) must be restarted when it reaches it, in plain
    co-sim as well as RL. An **edge case**: avoid it by choosing models whose limit covers
    the scenario.
- **EnergyPlus FMU**: can be started at any date within its max time, restarted from
  anywhere. Constraints: total experiment duration is a multiple of **86400 s**; minimum
  step **60 s**. **Verified in F4:** a start in the middle of a day is silently rounded down
  to midnight, so starts must be midnights (author's choice: `ScenarioManager` errors if
  `rolling_window` is not whole days for an E+ federate).

## Current Progress — F1–F4 DONE (uncommitted)

Docs realignment committed (`116a6b4`). Everything else is **uncommitted** (author commits).

F1 (clock + base reset): `BaseModel` has the abs/rel clocks (`rel_ts/rel_time`, never folded at the horizon; backend reads `horizon_pos_s()`; `_epoch` triggers `reset('horizon')`), one `_reset(mode, rel_ts_target, span_ticks)`, concrete no-op `reset(mode)`; `BaseFederate._reset` (none silent, publish on full+rolling, rolling target `reset_count·W`); `pass` resets removed; KNOWN_ISSUES §7 deleted; `tests/test_model_reset.py`.

F2 (units + span + horizon cleanup), details in the plan doc under "Fatto in F2":
- `ScenarioManager.reset_in_federate_ticks` + `_assign_federate_rl_configs()`; launcher just passes `rl_config` (or `{}`).
- `BaseFederate._next_span()`, `n_resets_train`, `initial_span_ticks` in model config → `BaseModel.span_ticks`; no reset on the last tick.
- Removed `simulation_horizon` (key, method, fields, docs, CLAUDE.md), `_apply_simulation_horizon`, `_shortest_declared_horizon`, `_model_catalog`, `_reset_on_horizon`.
- Docs: `rl.md` (units in agent steps, divisibility error, W==P is a restart), `general.md` section removed, `custom_models.md`, KNOWN_ISSUES §5 pruned.
- Verified: pytest 310 pass / 11 fail (same 11 as the baseline: 9 `test_fmu_state_snapshot.py` → F4 deletes, 2 KNOWN_ISSUES §6); 5 scenarios pass; mixed-period run (weather 900 s) resets in lockstep.
F3 (CSV): `BaseCSVReader` serves the row from the clock (`horizon_pos_s`), no cursor, no `reset` override, derived `max_sim_time = rows·real_period`; `WeatherCSVReader(BaseCSVReader)` only picks/renames the column; `skip_rows` = data rows. Detail + known limit (init publish after a rolling reset) in the plan doc under "Fatto in F3".

F4 (FMU), detail in the plan doc under "Fatto in F4": `BaseFMUModel` rewritten (952 → 543 lines): `reset(mode)` = teardown + new slave at `horizon_pos_s()` with stop = span (E+: whole days, capped at the RunPeriod end); snapshot/replay machinery and `supports_rollback` gone; `user_defined.fmu_engine: energyplus` on `bui0_building_fmu` (`--engine` in the register script); per-model reset attrs removed; F1 bug fixed (horizon restart kept dropping `span_ticks` → `_span_end_rel`); `test_fmu_state_snapshot.py` deleted, `tests/test_fmu_reset.py` added. pytest 282 pass (remote file excluded: KNOWN_ISSUES §6); `bui0_fmu_test`, `fmu_horizon_smoketest`, `bui0_setpoint_DQN`, `rc_building_test_base`, `bui_hp_DQN_rollingreset` pass; E+ rolling smoke `.eio` = 01/01, 01/03, 01/05, 01/07.

The six rejected `cs_*` rolling YAMLs were moved to `src/scenarios/to_review_rolling_window/` (README explains why); the author will review them later. `src/test_script.py` still has commented calls to two of them.

**Ask the author (F4 close):** each reset creates `fmu_output/restart_<n>/`; hundreds of episodes → hundreds of directories. Reuse one directory per model instance?

## Environment note (this machine)

`src/docker-compose.yaml` fixed and verified (2026-10-26): Redis now `redis:8.10.1-alpine` (allowed patched versions: 8.10.1, 8.8.2, 8.6.6, 8.4.6, 8.2.9, 7.4.11, 7.2.16, 6.2.24), MinIO/mc from `cgr.dev/chainguard/*` (Docker Hub images gone; MinIO runs as `user: root`, no curl healthcheck, `minio-init` retries). The stack is up (`cosim_redis`, `cosim_minio`, `cosim_mosquitto`, catalog loaded, bucket `fmus` created) and left running. Run scenarios with `export PATH=/media/space/rando/Environments/cosim_gym/bin:$PATH` (see env-gotchas memory), not `conda run`.

## Code state still broken / pending (KNOWN_ISSUES §5)

Parallel doesn't propagate resets (F5); one-tick publish of `init_state` after a reset (CSV row 0, FMU default `TBuilding`); horizon restarts out of phase between models; last E+ slave ends with a socket error (pre-existing).

## Left untouched on purpose

`tests/test_fmu_horizon_detection.py` (still green, untouched); `scripts/fmu_warmstart_validation/` and its results (built on the wrong premise); `tests/regression_suite.py`; scenario YAMLs; `docs/paper` does not exist (paper lives in a separate repo).

## What Worked

Splicing `base_model.py` by line range with a script after asserting anchors; baseline pytest via a throwaway `git worktree` of HEAD to diff failures; advisor on the wrap formula (caught that "origin + k·W mod max" shifts episode 2 when `start_time ≠ origin`); the author then replaced the wrap by a non-folding clock + `horizon_pos_s` (cleaner, no negative `rel_ts`).

## What Didn't Work / Watch Out

- Do not reuse the old replay / run-period-shift / checkpoint-slave reasoning (wrong premise); older FMU experiments are unverified.
- Do not infer "state carries over across a restart" for EnergyPlus; a restart at date X is a fresh start at X.
- Check `.eio` `Environment` line, not what the framework wrote.
- `cmd1 && VAR=x && ( ... ) &` backgrounds the whole chain; `$VAR` is empty in the foreground.

## Next Steps

1. F5 (parallel reset forwarding), then F6 (closure: consolidate tests, 2 regression scenarios incl. an E+ rolling one with whole-day window — ask go-ahead before the full run —, KNOWN_ISSUES §5, changes report `model_reset_redesign_changes.md`, plan → DONE).
2. Objective 2 (EnergyPlus Python API model) only after objective 1: `pyenergyplus_lbnl 26.1.0` in `cosim_gym` (import not verified), EnergyPlus 23.1 at `/opt/energyplus`, `run_energyplus` blocking/callback-driven.
3. Nothing is committed without an explicit ask. Execution: Sonnet + Opus advisor; each phase ends with plan STATUS, `graphify update .`, HANDOFF rewrite.
