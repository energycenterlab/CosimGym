# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file live in `CLAUDE.md` → **Working Rules**. Read that first.

## Goal
Branch `eplus_model`: a generic CosimGym model co-simulating any EnergyPlus IDF via `pyenergyplus` (model side only, no `src/core/` changes). Plan: `docs/changes_reports/eplus_model_plan.md`; report: `eplus_model_changes.md`; user doc: `docs/user_guide/eplus_models.md`.

## Current progress (committed and pushed by the user as c3a8706; not merged)
- Generic approach (user request): ONE catalog entry `eplus_model` (no IDF, no I/O). IDF/EPW and `eplus_io` are declared in the scenario `user_defined` (like the CSV reader). The model builds its state from `eplus_io`, binds it to `connections.subscribes/publishes` (unused entries dropped, missing ones = logged error), expands `*`/`?` keys at start-up (API listing cached in `~/.cosimgym/eplus_cache/listings`), supports `{schedule: NAME}` inputs, takes `max_sim_time`/`sim_start_date` from the IDF RunPeriod, logs runtime errors. The per-building `register` CLI and the two per-building catalog entries were removed; `eplus_discover.py` remains (listing + pattern preview).
- Verified: scenarios `bui0_eplus_test`, `bui0_eplus_multi_test`, `eplus_horizon_smoketest` pass with the generic entry (same values as before); EC_IdealLoads (24.1): pattern expansion, binding error logged, schedule shorthand drives a real actuator (0.25/0.75 read back). `tests/test_eplus_idf.py`: 5 pass.
- Pre-merge review follow-ups applied: unconnected inputs with a `default` are kept and driven; `expand_io` raises on name collisions; guide states `init_state` vs `default` timing; new scenario `eplus_ec_pattern_test` (EC_IdealLoads: `*` patterns, `{schedule:}`, defaults, horizon from RunPeriod; verified: 5 inputs/2 outputs, Occ_rb=0.5 over 96 ticks, T 20.0-20.3) added to `SCENARIOS`. The BaseFederate horizon log was checked: it is correct (initialize runs in the BaseModel constructor), nothing to move.
- Regression suite on the final code (c3a8706, E+ 23.1.0 on PATH): 46 passed, 0 failed, 4 xfail (known, unchanged; no UNEXPECTED-PASS), incl. `eplus_ec_pattern_test`. Ready to merge `eplus_model` into `main` (user decides; not merged).
- Regression suite entries Docs rewritten: `docs/user_guide/eplus_models.md`, plus KNOWN_ISSUES, models_reference, developer_guide, CLAUDE.md, changes report, plan note.

## What worked
- Worker subprocess + callbacks; own tick counter with clock check; restart-at-midnight for reset/mid-run start; `burn_in_days` (14 d → 0.04 % error on BUI0).
- 23.1 and 24.1 (Transition upgrade) agree on BUI0 (8363.566 vs 8363.565 W).
- Run scenarios with `export PATH=/opt/conda/envs/cosim_gym/bin:$PATH` (helics_broker), Redis docker up.

## What did not work
- `yaml.dump` of `catalog.yaml` rewrote the whole file (that register CLI no longer exists).
- `bui0_fmu_test` SIGTERM was KNOWN_ISSUES §14 (no `energyplus` on PATH): it passes with `PATH=$HOME/.cosimgym/energyplus/23.1.0:/opt/conda/envs/cosim_gym/bin:$PATH`. Run the suite with that PATH.
- EnergyPlus 26.x IDFs (the four `2_zones_*`) cannot run here (glibc 2.31); deferred by user.

## Next steps
1. FMU vs API compared on BUI0 (1 day, 144 steps, identical IDF and inputs): the FMU embeds its own weather `meteo_frassinetto_TMY.epw`, NOT the San Francisco epw that `bui0_eplus_test` uses, so a like-for-like run needs the FMU's epw (extract it from `BUI0.fmu`, not in the repo). With the same epw: TBuilding agrees to 0.008 K (rmse 0.002 K) from step 12; HeatingLoadTarget within ~1.7 % rmse except at setpoint-change steps (47-49: 2-6 %, step 107: -32 kW FMU vs +6 kW API) and the first 3 steps (FMU reports its initial state at tick 0, API is one zone step ahead). Cause of the load differences at setpoint changes NOT verified (likely when the new setpoint is reflected in the sampled load). Not committed: neither a test nor a scenario change.
2. **Numerical comparisons FMU vs API are still to do, properly**: create two essentially identical scenarios (same IDF, same inputs, same weather file, same dates/steps; only the model differs: FMU vs `eplus_model`), which needs the FMU's `meteo_frassinetto_TMY.epw` in `resources/` (or an FMU rebuilt with the San Francisco epw). Then compare TBuilding/HeatingLoadTarget (the load differs at setpoint-change steps; cause unverified). Not a merge blocker.
3. Merge `eplus_model` into `main` (suite green on the final code); then re-run the suite once on `main`.
4. Decide `burn_in_days` default; exercise `skip_sizing`, PythonPlugin, `To:Variable`.
5. `docs/` changed a lot: graphify graph is stale (not rebuilt; costs tokens).
