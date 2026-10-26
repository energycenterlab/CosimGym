# Model reset & horizon restart — follow-ups

Context. Two independent concepts live in `BaseModel` / `BaseFMUModel`:

- **Model horizon restart** — a model with an intrinsic limit (`max_sim_time`: an
  EnergyPlus run period, a CSV covering one year) is restarted when it reaches it.
  Applies to any model, RL or not. An edge case: the normal answer is to pick models
  whose limit covers the scenario.
- **RL episode reset** — `full` / `rolling` / `none`, only in RL runs.

The mechanics of both are being redesigned (see `docs/KNOWN_ISSUES.md` §5 and §7;
user-facing page: `docs/user_guide/fmu_models.md` §4). This file keeps only the
deferred items that stay valid whatever the redesign decides. The earlier analysis of
restart-by-replay, run-period shifting and checkpoint slaves rested on a wrong premise
about EnergyPlus (an EnergyPlus FMU can start and restart at any date inside its maximum
time) and has been removed; the history is in
`docs/changes_reports/fmu_horizon_and_reset_changes.md`.

---

## 1. No horizon is ever inferred — it must be declared

A model with no `max_sim_time` in its catalog entry is treated as unbounded and is never
restarted; one that does have a limit then fails when stepped past it. Inferring a limit
means a model silently acquiring a limit nobody wrote down, so the explicit field plus a
warning from `fmu_catalog_register.py` when nothing is detected was preferred.

**Possible later:** at registration, if the entry declares no horizon but the FMU is
recognisably an EnergyPlus export, default it from the run period and log it.

## 2. The discontinuity at a restart is not signalled to the RL agent

A restart returns the model to its initial conditions, so the transition spanning it is
not a valid MDP transition. Nothing marks it. Rare when the horizon is long; every
episode when an episode is longer than the horizon.

**Possible later:** raise a discontinuity flag that reaches `RL_Federate` as
`truncated=True` (truncation, not termination), or restart only at an episode boundary,
or refuse such a configuration at parse time.

## 3. Horizon coherence across federates

Restart is currently per model: each model restarts from its own `max_sim_time`. A
schedule model and the building it feeds stay in phase only if they declare the same
horizon (or none); `BaseFederate._report_simulation_horizons` warns about a mismatch
inside one federate but not across federates. `sim_start_date` is stored in the catalog
and read by no code — results are not labelled with it, so a scenario calendar and a
model's own calendar can differ in labels only.

## 4. `parallel_execution` does not propagate resets to worker processes

`src/core/parallel_executor.py` has no reset path: workers rebuild their model shard from
config and step their own copies, so a reset in the main process never reaches them. RL is
already rejected together with `parallel_execution`, but a model's horizon restart is not
RL. Options: reject a horizon-declaring model with `parallel_execution`, or give the
executor a reset entry point that tells every worker to reset. `RL_Federate` has its own
run loop and does no horizon handling; same class of gap.

## 5. Every FMU restart pays EnergyPlus finishing its run period

Measured on BUI0 with EnergyPlus 23.1 (before the redesign): freeing the slave
(`terminate` + `freeInstance`) took ~3.75 s because `fmi2Terminate` hands control back to
EnergyPlus, which runs out the rest of its run period before exiting; instantiating and
warming up took ~0.55 s. The teardown cost scales with the run period still ahead, so
bounding each instance to the span actually needed would reduce it. To be reconciled with
`max_sim_time` and the whole-day (86400 s) constraint. Killing the process is the blunt
alternative and would lose the `.err` / `.eso` files written on exit.
