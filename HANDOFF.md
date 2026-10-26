# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session)

Realign the documentation with the author's correct premises about model reset/restart
and EnergyPlus, **before** any planning or implementation. Docs only; no `.py` touched,
nothing committed.

Two objectives follow, to be handled **separately, FMU first**:

1. **Refactor the reset functions** — generic in `BaseModel`, then specific in
   `BaseFMUModel`. EnergyPlus FMUs are an exception in how they restart, and others may be.
2. **A new model that runs EnergyPlus through its Python API** (no IDF→FMU export).

The next request starts with **planning objective 1**. No plan exists yet.

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
  step **60 s**.

## Current Progress — done, uncommitted (docs only)

Redis 8.2.10 upgrade is committed (`d6fe250`); only the FMU/reset docs below are modified.

- `docs/user_guide/fmu_models.md`: §3 EnergyPlus-start paragraph corrected; §4 rewritten
  as "Simulation horizon and restart" (the two concepts, `max_sim_time` / `sim_start_date`,
  EnergyPlus constraints, optional state save/restore, register script). Replay, input
  history, `replay_inputs`, quadratic cost, "two states held" and "cannot be restarted
  anywhere else" removed. Mechanics flagged *under redesign*. Anchor is now
  `#4-simulation-horizon-and-restart`.
- `docs/user_guide/scenario_configuration/general.md` (`simulation_horizon`),
  `federate.md` (`fmu_reset` keys), `rl.md` (`reset.mode: rolling`): off-topic content left
  alone; on-topic false/removed-code claims replaced or marked under redesign.
- `docs/KNOWN_ISSUES.md`: §5 rewritten as the list of verified mid-refactor defects (below);
  §7 kept untouched (its fix is a planning decision); `Last reviewed` 2026-10-26.
- `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`: rewritten; kept only items
  valid under the correct premise (no horizon inference, no truncation flag for the agent,
  horizon coherence across federates, `parallel_execution` reset propagation, teardown cost).
  Replay / run-period shift / checkpoint slave / "SETTLED" entries deleted.
- `docs/changes_reports/fmu_horizon_and_reset_changes.md`: dated CORRECTION block at top,
  §6 conclusions marked superseded, wrong inline claims fixed. Kept as history.

## Code state observed (verified in source, 2026-10-26; also in KNOWN_ISSUES §5)

- Snapshot used after `freeInstance` (`reset('full')` reassigns `self.snapshot =
  self.initial_snapshot`, then tears the slave down); `_start_instance` overwrites
  `self.snapshot` every time.
- Restore branch `state.ts > 1` in `_push_initial_state_to_fmu` unreachable
  (`BaseModel._reset` zeroes `ts` via `init_state` deepcopy first).
- `_state_snapshots` / `_snapshot_target_ts` missing → `AttributeError` for rolling +
  state-capable FMU; `tests/test_fmu_state_snapshot.py` fails.
- `base_model.py:360`: `rel_ts = ts - episode_length + rolling_window` right only for the
  first reset or `rolling_window == episode_length`.
- `_stop_time_seconds()` returns `max_sim_time` as an absolute stop even when start > 0 —
  breaks the 86400 s duration; the first `doStep` must also sit at `startTime`.
- `reset('rolling')` always tears down, even when `rolling_window == episode_length`.
- Horizon restart is per model (`BaseModel.local_time`); `BaseFederate._reset_on_horizon`
  commented out; `ScenarioManager._apply_simulation_horizon` still fills
  `timing_configs.max_sim_time`, which nothing acts on; `BaseFederate._report_simulation_horizons`
  only warns on mismatch.
- `BaseModel.reset` abstract in 13 catalog models without it (KNOWN_ISSUES §7) — the gate is
  red (6/32 scenarios start, 25 failed tests) until decided.
- `BaseCSVReader.reset` handles `full`/`horizon_limit`/`rolling`/`soft`; its
  `_reposition_backend` is commented out.

## Left untouched on purpose (stale, belongs to the redesign)

- Docstrings and commented-out blocks in `base_FMU_model.py`, `base_model.py`,
  `base_csv_reader.py`, `BaseFederate.py`.
- `tests/test_fmu_state_snapshot.py`, `tests/test_model_local_clock.py`,
  `tests/test_fmu_horizon_detection.py` (reference the removed API).
- `scripts/fmu_warmstart_validation/` and its `results*.json` (built on the wrong premise).
- `tests/regression_suite.py`; scenario YAMLs (`fmu_horizon_smoketest` is still valid).
- `docs/paper` does not exist in this repo; the paper lives in a separate repo — not checked
  for the false EnergyPlus claim. Memory files contain nothing on this topic.

## What Worked

Grepping docs/scenarios/catalog/memory for FMU-specific terms (not just "rolling", which is
mostly RL), then checking each doc claim against source before keeping it. The doc/code
mismatch list was longer than the doc/doc one.

## What Didn't Work / Watch Out

- Do not reuse the old followups' reasoning (replay is the only way, run-period shifting is
  ignored by EnergyPlus, checkpoint slave). It rested on a wrong premise. The older
  experiments (`startTime != 0` → `status 3`, rewritten `RunPeriod` ignored) were interpreted
  under that premise — treat them as unverified, not as findings.
- Do not infer "state carries over across a restart" for EnergyPlus; the author only said any
  start date works. A restart at date X is a fresh start at X.
- Do not fix §7 or the FMU defects as drive-bys; they belong to the plan.
- Check `.eio` `Environment` line (what the slave actually simulated), never only what the
  framework wrote.

## Next Steps

1. Plan objective 1 (reset refactor): `docs/changes_reports/<feature>_plan.md` (e.g.
   `model_reset_redesign_plan.md`). Decide `BaseModel.reset` default vs abstract (§7),
   clock/`rel_ts` semantics, who owns rolling start-tick arithmetic, per-model vs
   federation-wide horizon restart, parallel/CSV/FMU specialisation, and an EnergyPlus
   start-date/duration strategy honouring 86400 s / 60 s.
2. Optional, non-blocking: confirm the EnergyPlus start-anywhere behaviour on BUI0 through
   the `.eio` `Environment` line.
3. Then plan objective 2 (EnergyPlus Python API model). Notes: `pyenergyplus_lbnl 26.1.0`
   is in `cosim_gym` (import not verified), EnergyPlus 23.1 at `/opt/energyplus`; the
   version mismatch with existing IDFs must be decided; `run_energyplus` is blocking and
   callback-driven, so master stepping needs a thread handshake.
4. Update the stale tests and `regression_suite.py` as part of implementing (not before).
5. Nothing is committed without an explicit ask.
