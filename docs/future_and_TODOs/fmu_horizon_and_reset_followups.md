# FMU simulation horizon & reset — follow-ups

Context: `BaseModel` now keeps a model-local clock and restarts a model that
declares `max_sim_time` when it reaches that horizon; `BaseFMUModel` implements
the restart and the forward replay a rolling reset needs. User guide:
`docs/user_guide/fmu_models.md` §4. Tests: `tests/test_model_local_clock.py`,
`tests/test_fmu_horizon_detection.py`.

These are the deliberate limitations left behind, each with the reasoning for
why it was left rather than solved.

---

## 1. No horizon is ever inferred — it must be declared

**Now:** a model with no `max_sim_time` in its catalog entry is treated as
unbounded. It is stepped for the whole scenario and never restarted. An
EnergyPlus FMU whose entry does not declare its RunPeriod length therefore still
fails when the run passes that length.

**Why:** inferring a horizon means a model silently acquiring a limit nobody
wrote down. The explicit field plus a loud warning from the register script when
nothing is detected was preferred over a heuristic.

**Possible later:** a fallback heuristic at registration — if the entry declares
no horizon but the FMU is recognisably an EnergyPlus export (generation tool,
`modelIdentifier`, IDF-derived description), default to 31536000 s and log it.
Cheap to add in `fmu_catalog_register.parse_fmu`; the question is only whether an
inferred limit is better than a missing one.

---

## 2. Rolling resets are quadratic *only* without state save/restore

**Done:** FMUs that declare `canGetAndSetFMUstate` are now restarted by saving and
restoring their state (`BaseFMUModel._take_snapshot` / `_restore_snapshot`),
which is instant and exact and needs no replay and no input history. The initial
state is saved at startup and reused by every full reset and horizon restart;
under `rolling` the next episode's start point is saved in passing, so a rewind
is a restore. Verified against the FMI 3.0 Feedthrough FMU in
`tests/test_fmu_state_snapshot.py` (restores happen in place - the slave is never
re-instantiated). The capability is read from the FMU's own modelDescription; the
catalog can only disable it, never claim it.

**Still open, for FMUs without the capability** (EnergyPlus reports
`canGetAndSetFMUstate=false`, and its real state lives in the EnergyPlus process
rather than in the 8 declared FMI variables, so it cannot be emulated):

```
restarts     = episodes
replay steps ≈ rolling_window × episodes × (episodes − 1) / 2
```

The model logs this estimate at startup and proceeds; it never blocks.

- **A second warm slave.** The only checkpoint available to a non-snapshotting FMU
  is "another instance held at a known point". That caps replay length at the
  distance between checkpoints, at the price of a second EnergyPlus process.
- **Align `rolling_window` with `episode_length`.** When they are equal the next
  episode starts exactly where the last ended, no rewind is needed, and the cost
  collapses to zero. Worth recommending for EnergyPlus-backed rolling studies.

---

## 3. Replay inputs are recorded, not reconstructed

**Now:** recording only happens for FMUs that *cannot* save their state - a saved
state carries everything, so the history is skipped entirely when snapshots are
available. On the replay path the FMU is fed the input values recorded at those
ticks of the current epoch. When the recorded history does not cover the requested
prefix, the initial inputs are held constant and a warning says the replayed span
is approximate. Recording is only active when the federate runs rolling resets
(`reset_mode` is injected into the model config), so other runs pay nothing.

**Limits:** memory is `n_ticks × n_inputs` floats per instance (a year at 15-min
steps is a few MB; at 1-min steps, tens of MB), and the history is discarded when
the slave restarts at local time 0, since everything after that point belongs to
a span it no longer has.

**Possible later:** cap the history with a ring buffer sized to the largest
rewind the reset policy can ask for (`rolling_window × episodes`), or persist it
alongside the results so a replay can outlive a restart.

---

## 4. The physical discontinuity at a restart is not signalled to the RL agent

**Now:** a restart returns the model to its initial conditions, so the transition
spanning it is not a valid MDP transition — the building's state jumps. Nothing
marks it: the agent sees it as an ordinary step.

**Possible later, from the original design discussion:**

- `policy: preemptive` — restart at the last episode boundary before the horizon,
  so the discontinuity always coincides with an episode reset where observations
  are re-read anyway. Costs at most one episode of the cycle. Only defined when
  `max_sim_time ≥ episode_length × real_period`.
- `policy: immediate` + truncation flag — keep restarting exactly at the horizon,
  but have the model raise a discontinuity flag that reaches `RL_Federate` as
  `truncated=True` (truncation, not termination: the value bootstrap must be
  kept). This becomes important, not optional, when an episode is longer than the
  horizon, because then *every* episode contains one.
- `policy: error` — refuse at parse time for studies that must be restart-free.

---

## 5. Clock coherence across federates — SOLVED, with one caveat

**Resolved.** The horizon restart is a federate-wide reset, not a model-private
one: `ScenarioManager._apply_simulation_horizon` resolves the scenario's horizon
(from the models' catalog entries or the scenario's `simulation_horizon` key) and
gives it to every federate, and `BaseFederate._reset_on_horizon` restarts every
model it owns at that tick. Federates do not have to coordinate — they all count
the same ticks — so a feeder and the building it feeds go back to their first
step together.

`state.time` follows the model clock rather than raw federation ticks, so a model
that decides anything from the date rewinds with the restart. Verified in
`src/scenarios/fmu_horizon_smoketest.yaml`: with a 1-day horizon in a 3-day run,
both the feeder's schedule and the FMU's `TBuilding` replay day 1 exactly on day
2. This also fixed a pre-existing bug where a plain `full` RL reset rewound the
FMU but not the schedule model feeding it.

**Caveat:** models whose cursor is not derived from the model clock still need a
`_reposition_backend`. `BaseCSVReader` has one; a custom model holding its own
counter will keep counting through a restart unless it implements the hook or
derives its position from `state.ts` / `local_ts()`.

**Also unresolved:** the FMU's *calendar* still restarts at its own RunPeriod
begin date, which need not match the scenario's start date. `sim_start_date`
records what that date is but nothing uses it to label results — a run whose
scenario starts in March while the FMU's RunPeriod starts in January will have
the FMU simulating January while the scenario calendar says March. Only the
labels disagree; every model is at the same *elapsed* simulated time.

---

## 6. `parallel_execution` does not propagate resets to worker processes — TO ADDRESS

**Status:** open, needs design. Nothing in the repo currently hits it, but it is a
correctness gap rather than a limitation, so it should be thought through and
closed rather than left indefinitely.

**What is wrong:** `src/core/parallel_executor.py` has **no reset path at all**.
Workers rebuild their shard of model instances from the config and step their own
copies; `BaseFederate._reset()` and `BaseFederate._reset_on_horizon()` both
iterate `self.entities`, which are the *main process's* copies. A reset therefore
never reaches the models that are actually being stepped.

**Why it has not bitten yet:** `parallel_execution` is already rejected with
`type: rl` (`NotImplementedError`), so an RL episode reset can never co-occur with
it. The horizon restart is the new case — **it is not an RL feature**. It fires in
plain co-simulation, so a model declaring `max_sim_time` together with
`parallel_execution: true` would restart only the main-process copies while the
workers keep stepping past the limit. For an EnergyPlus FMU that means the worker
slaves fail at their run period, or worse, silently produce results from a state
nobody restarted.

**Options considered:**

- **Reject the combination.** Raise `NotImplementedError` when a model declares
  `max_sim_time` and `parallel_execution` is true, exactly as the repo already
  rejects `parallel_execution` with `override_enabled` and `type: rl`. Small and
  loud, but closes off a combination that is otherwise attractive (many FMU
  instances stepped in parallel is precisely the scaling case).
- **Wire resets through the executor.** Give `ParallelModelExecutor` a reset entry
  point that sends the mode and target tick to every worker, each of which
  repositions its own shard. This is the real fix, and the one worth designing:
  workers hold live FMU slaves, so each performs its own restart or replay, and
  the replay cost multiplies by worker count unless the FMU supports state
  save/restore. Interaction with the escalating `close()` path (sentinel → join →
  terminate → kill) needs checking, as does what happens to a worker that fails to
  restart.

**Related, smaller:** `RL_Federate` has its own run loop and never calls
`_reset_on_horizon`. Harmless while RL federates hold only agents (which declare
no horizon), but it is the same class of hole and should be closed with this work.

