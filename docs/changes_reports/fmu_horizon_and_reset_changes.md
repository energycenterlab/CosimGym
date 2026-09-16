# FMU restart & reset — what changed

Change note for the work adding automatic restart of models that cannot run for
the whole scenario (an EnergyPlus FMU stops at the end of its RunPeriod), and
making `reset.mode: rolling` work with FMUs at all.

Reference docs: `docs/user_guide/fmu_models.md` §4 (user-facing),
`docs/future_and_TODOs/fmu_horizon_and_reset_followups.md` (open items).
Tests: `tests/test_model_local_clock.py`, `tests/test_fmu_state_snapshot.py`,
`tests/test_fmu_horizon_detection.py`, scenario `fmu_horizon_smoketest`.

---

## 1. The one new idea: a model clock

Before, a model's simulated time was the federation tick, full stop. A reset
restored a model's *state* but its clock kept running, which meant nothing that
depended on the date could ever be rewound.

Now every model has its own tick:

```
local_ts(ts) = ts - ts_shift        # ts_shift moves at every restart
state.time   = start_time + local_ts * real_period
```

`ts_shift` is 0 until something moves the model back in time, so a model that is
never reset or restarted behaves exactly as before. One primitive moves it:

```
reposition(target_ts)    # "behave as if the next tick were tick target_ts"
```

Everything routes through it: `full` reset asks for tick 1, `rolling` asks for the
episode's start tick, a model at its horizon asks for tick 1. `_reposition_backend`
is the hook a model implements to make its *contents* follow — a no-op for a plain
Python model, a slave restart for an FMU, a cursor move for a CSV reader.

**`state.time` now rewinds with a reset.** Results are unaffected: they are
timestamped from the federate's own monotonic clock (`self.date_time`), not from
model state.

## 2. FMU restart: two mechanisms

Picked automatically from the FMU's own `modelDescription`, never from a catalog
claim (the catalog can only *disable* a capability, for an FMU that advertises it
but does not honour it).

| `canGetAndSetFMUstate` | Mechanism | Cost |
| ---------------------- | --------- | ---- |
| **true** (Feedthrough_FMI3, most Modelica exports) | save / restore state | instant, exact, either direction |
| **false** (BUI0 and every EnergyPlus export) | free slave → instantiate → replay forward | one real FMU step per tick replayed |

**Save/restore** captures the FMU's entire internal state as one opaque blob. The
state at tick 1 is saved at startup and reused by every full reset and every
horizon restart; under `rolling`, the next episode's start point is saved *in
passing* while the current episode runs, so a rewind is a restore. No replay, and
no need to remember inputs — the blob has everything.

This cannot be emulated by saving parameters and variables by hand: FMI exposes
only the declared I/O (BUI0 publishes **8 variables**), while the building's real
state — zone thermal mass, surface temperatures, HVAC, warm-up history — lives
inside the EnergyPlus process and is never published.

**Restart and replay** is the fallback. The unzip directory is cached, so a
restart never re-downloads or re-extracts; the old slave is properly
`terminate()`d and `freeInstance()`d (it previously was not, leaking one
EnergyPlus process per reset), and each restart writes to its own
`fmu_output/restart_<n>/`.

## 3. Rolling reset, end to end

`BaseFederate._reset()` was already computing `new_starting_point += rolling_window`
and calling `model.reset(mode='rolling', ts=new_starting_point)`. That call
**crashed with a TypeError** on any FMU, because `BaseFMUModel.reset()` took no
arguments. Rolling + FMU simply did not run before this change.

What happens now, per episode:

1. Federate computes the next start tick (`k × rolling_window`) and resets every
   model with it.
2. Each model repositions to that tick. The target is almost always *behind* the
   current tick (`rolling_window` is usually much smaller than `episode_length`),
   so this is a rewind.
3. FMU with state support → restore the snapshot taken in passing. **O(1).**
4. FMU without → restart the slave and replay forward to the start tick, feeding
   it the inputs recorded at those ticks so it arrives in the state it really had.
5. CSV readers move their row cursor to the same tick; plain models restore state.
   `state.time` rewinds for all of them, so a schedule model and the building it
   feeds are at the same simulated moment.

**Cost, on the replay path only:**

```
restarts     = episodes
replay steps ≈ rolling_window × episodes × (episodes − 1) / 2
```

100 episodes, `rolling_window: 10` → ~50 000 replayed steps, minutes. Same
episodes with `rolling_window: 2880` → ~14 million, days. It is **quadratic in
episode count**. The model logs the estimate at startup and proceeds — nothing is
blocked. Setting `rolling_window == episode_length` removes the rewind entirely.

## 4. Horizon restart: a federation-wide reset, not a mode

A model declares its limit in the catalog (`max_sim_time`, `sim_start_date`),
auto-detected at registration from the IDF RunPeriod / `DefaultExperiment`.
`ScenarioManager._apply_simulation_horizon` resolves one horizon for the whole
scenario — shortest declared, or the scenario's `simulation_horizon` override —
and hands it to **every** federate. Each federate then restarts **all** of its
models at that tick (`BaseFederate._reset_on_horizon`), exactly as at an episode
boundary.

Federates need no coordination: they all count the same ticks. Verified in
`fmu_horizon_smoketest` (1-day horizon, 3-day run, horizon declared only on the
FMU) — both the feeder's schedule and the FMU's `TBuilding` replay day 1 exactly
on day 2.

It is **not** an RL reset mode. It fires in training, in testing and in plain
co-simulation with no agent, and composes with whatever `reset.mode` is set.

## 5. Effect on existing scenarios

Nothing changes unless a model declares `max_sim_time`. Only `bui0_building_fmu`
does (31536000 s, from its own IDF), and no current scenario runs a year, so no
restart triggers. Verified: 38/38 regression scenarios pass, 313 unit tests.

One **behaviour change** worth knowing: a `full` RL reset now rewinds `state.time`
for every model. That is a bug fix — previously the FMU restarted at its own start
date while the schedule feeding it kept advancing, so episode 2 fed Jan-3
schedules into a building simulating Jan-1 — but any model that relied on
`state.time` advancing monotonically across episodes will now see it repeat.

---

## Soft spots and possible problems

**1. `parallel_execution` does not see any reset.** *(tracked as future work —
`docs/future_and_TODOs/fmu_horizon_and_reset_followups.md` §6)* `parallel_executor.py` has no
reset path at all: workers rebuild models from config and step their own copies,
so a reset in the main process — episode *or* horizon — never reaches them. RL is
already rejected with `parallel_execution`, so episode resets cannot co-occur, but
**the horizon restart is not RL**: a horizon-declaring model with
`parallel_execution: true` would restart only the main-process copies while the
workers keep stepping past the limit. Untested and expected to misbehave. Either
wire resets through the executor or reject the combination.

**2. `RL_Federate` has its own run loop** and does not call `_reset_on_horizon`.
Harmless today — RL federates hold agents, not physics models, and agents declare
no horizon — but if an RL federate ever holds a horizon-bounded model it will not
restart it.

**3. The restart discontinuity is invisible to the agent.** A restart returns the
model to its initial conditions, so the transition spanning it is not a valid MDP
transition and nothing marks it. Rare when the horizon is a year; **every episode**
when an episode is longer than the horizon. The fix (a `truncated=True` flag, not
`terminated`) is written up in the follow-ups doc but not implemented.

**4. Replay input history costs memory** on the non-snapshot path: `n_ticks ×
n_inputs` floats per instance while a rolling run is active (a year at 15-min steps
is a few MB; at 1-min steps, tens of MB). If the history does not cover the
requested span the initial inputs are held constant and the replayed span is
**physically approximate** — a warning says so, but the results do not record it.

**5. EnergyPlus requires whole-day horizons.** `stopTime − startTime` must be a
multiple of 86400 or `fmi2EnterInitializationMode` fails outright. Now warned at
init, but it also rules out ever restarting at an arbitrary episode boundary for
EP FMUs.

**6. Custom models holding their own cursor.** Anything that counts internally
rather than deriving position from `state.ts` / `local_ts()` will keep counting
through a restart unless it implements `_reposition_backend`. `BaseCSVReader` and
`BaseFMUModel` do; a user model might not, and nothing detects it.

**7. Horizon not an exact number of steps.** `horizon % real_period != 0` means the
restart lands early by the remainder. Warned at parse, not corrected.

**8. FMU calendar vs scenario calendar.** The FMU restarts at its own RunPeriod
begin date, which need not match the scenario's start date. `sim_start_date`
records it but nothing uses it to label results, so a scenario starting in March
with a January RunPeriod has the FMU simulating January while the scenario
calendar says March. Only the labels disagree — every model is at the same
*elapsed* simulated time.

**9. Mixed horizons.** If models declare different limits the shortest wins for
everyone, with a warning. That is safe but may not be what was intended.

**10. The horizon is only as good as the declaration.** No limit is ever inferred:
an EnergyPlus FMU registered without `--idf` or `--max-sim-time` is treated as
unbounded and will still fail when it runs past its run period. The register
script warns at registration time; nothing checks at run time.
