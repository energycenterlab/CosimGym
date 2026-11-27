# EnergyPlus (Python API) generic model: changes report

Plan: `eplus_model_plan.md`. Branch `eplus_model`, model side only (no change to `src/core/`). Nothing committed.

## What was done
- `src/models/base_eplus_model.py` `EnergyPlusModel(BaseModel)`; `eplus_worker.py` (one E+ process per instance, socketpair JSON-lines IPC);
  `eplus_idf.py` (IDF parse/edit, I/O normalisation); `eplus_registry.py` (installs, Transition-chain upgrade + cache);
  `model_catalog/eplus_discover.py` (lists the IDF's API interface, previews key patterns).
- One generic catalog entry `eplus_model` (no I/O, no IDF). Building and `eplus_io` live in the scenario's `user_defined` (the CSV-reader approach);
  the model builds its own state from `eplus_io`, binds it to the connections, expands `*`/`?` keys at start-up (listing cached), accepts `{schedule: NAME}`
  inputs, derives `max_sim_time`/`sim_start_date` from the IDF RunPeriod and logs runtime errors.
  First design (a catalog entry per building written by a `register` CLI) was dropped as superfluous.
- Scenarios `bui0_eplus_test`, `bui0_eplus_multi_test`, `eplus_horizon_smoketest`; regression suite entries; `tests/test_eplus_idf.py` (4 pass).

## Why these choices
- Worker subprocess: E+ is blocking and non-rewindable, versions differ per IDF, a crash must not take the federate down. Alternative (in-process) rejected.
- Own tick counter + clock check, not `current_sim_time` (not a tick boundary under HVAC sub-stepping).
- Mid-run start / reset = restart the worker at midnight + free-run + optional `burn_in_days` (E+ has no state snapshot in the API).
- `catalog.yaml` appended as text: `yaml.dump` rewrote the whole file and dropped comments (fixed after it happened).

## Verified (real runs)
BUI0 (23.1): outputs follow setpoint; 144-tick clock check; mid-run start; N=3; bad `real_period` rejected; reset; no leaked workers.
EC_IdealLoads (24.1): start-up ~42 s, setpoints obeyed, N=4 `agg: mean`. BUI0 upgraded to 24.1: 8363.566 vs 8363.565 W.
Scenarios through ScenarioManager: single, 2 parallel instances (identical traces), horizon restart (tick 144 == tick 0).
`burn_in_days` 0/1/3/7/14 vs continuous run: error of heating load 23 % / 12 % / 4 % / 0.6 % / 0.04 %.

## Not done / problems
- Numeric comparison with the FMU twin: `bui0_fmu_test` was killed by SIGTERM in SETUP (cause not diagnosed; EnergyPlus-API results only checked for physical consistency and 23.1 vs 24.1 agreement).
- Limits are in `KNOWN_ISSUES.md` (§ EnergyPlus model) and `future_and_TODOs/eplus_model_followups.md`.
- Full `tests/regression_suite.py` not run.

## Pre-merge review follow-ups
- Inputs with a `default` stay active when unconnected (otherwise the default was silently lost); `expand_io` rejects colliding expanded names; the guide states that `init_state` applies from the first tick while restarts burn in with the spec `default`.
- Scenario `eplus_ec_pattern_test` (EC_IdealLoads) covers key patterns, the `{schedule:}` shorthand, unconnected defaults and the RunPeriod horizon through ScenarioManager.
