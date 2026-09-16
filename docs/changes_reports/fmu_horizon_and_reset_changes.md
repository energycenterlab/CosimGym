# FMU restart & reset — what changed

Change note for the work adding automatic restart of models that cannot run for
the whole scenario (an EnergyPlus FMU stops at the end of its RunPeriod), and
making `reset.mode: rolling` work with FMUs at all.

Reference docs: `docs/user_guide/fmu_models.md` §4 (user-facing),
`docs/future_and_TODOs/fmu_horizon_and_reset_followups.md` (open items).
Tests: `tests/test_model_local_clock.py`, `tests/test_fmu_state_snapshot.py`,
`tests/test_fmu_horizon_detection.py`, scenario `fmu_horizon_smoketest`.
Validation harness: `scripts/fmu_warmstart_validation/`.

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
blocked. Setting `rolling_window` equal to the reset period removes the rewind
entirely — and is the only thing that does, as §6 establishes.

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

## 6. Second round: a cheaper restart, attempted and rejected

The replay is exact but its cost is the distance travelled, and under `rolling`
that distance grows every episode. The question that opened this round was
whether an EnergyPlus slave could simply be *started* at the tick it has to
reach, with its state set to what it had.

### Three routes, all closed

Checked against `BUI0.fmu` with EnergyPlus 23.1 — the first two in the binary and
the run directory, not in the FMI specification:

| route | result |
| --- | --- |
| FMI `startTime` = the target | `BUI0.so` uses it only for the whole-day check (`"The delta between the FMU stop time and the FMU start time must be a multiple of 86400"`) and the first-communication-point check. A non-zero start time makes the first step fail: `fmi2DoStep failed with status 3 (error)`. |
| rewrite the `RunPeriod` begin date in the IDF the FMU carries | **Ignored.** The wrapper does re-preprocess `resources/` at every instantiation, but it takes the run period's begin date from the original IDF and its *length* from the FMI stop time. Writing `begin 01-31` produced a slave that ran `01-01 .. 12-01`. |
| set the state through the interface | Nothing to set. BUI0's interface is six `ExternalInterface:...:To:Schedule` inputs and two `From:Variable` outputs; zone temperature is an **output**, and EnergyPlus exposes no actuator for zone air or surface node temperatures. That is also why it reports `canGetAndSetFMUstate=false`. |

### What was built, and why it was removed

A `restart_strategy: runperiod_shift` was implemented on the second route: split a
rewind into whole days handed to EnergyPlus by moving its begin date, plus a
short spin-up replayed with recorded inputs. It had a unit test, a smoke scenario
and a passing regression run — **and it never worked.** The test asserted on the
IDF *the code had written*, and the scenario asserted that a restart happened, so
neither noticed that EnergyPlus kept starting on 1 January. The only effect of
the "shift" was to shorten the slave's run period while the model believed it had
jumped forward in the year.

The validation harness (`scripts/fmu_warmstart_validation/`) is what caught it: a
rewind into July came back **6.1 K RMSE** with a heating-energy error of 10⁷ %,
and a spin-up of 0, 1, 3 or 7 days changed nothing — a state error decays and
responds to a spin-up, a calendar error does neither. Reading the run directory's
IDF and the `.eio` `Environment` line settled it: `01/01/2007, 12/01/2007`.

**The lesson, worth keeping:** an FMU test that inspects only what the framework
wrote proves nothing. The assertion has to reach what the slave actually
simulated — the run directory's processed IDF, its `.eio`, or the outputs
compared against a continuous run.

The strategy, its scenario and its tests are gone. What survives is the harness,
the seven bugs below, and the entry in the follow-ups (§7) that records the
three closed routes so nobody spends another session on them.

### What the harness does say about `replay`

Run on BUI0 across three rewinds — 9 days back in winter, 9 days back in summer,
and 119 days back — each compared against the same building stepped straight
through:

| case | replayed | ΔT RMSE | max ΔT | heating energy |
| --- | --- | --- | --- | --- |
| winter (day 40 → 31) | 4 320 ticks | 0.0000 K | 0.0000 K | −0.0000 % |
| summer (day 205 → 196) | 28 080 ticks | 0.0000 K | 0.0000 K | +0.0001 % |
| long rewind (day 150 → 31) | 4 320 ticks | 0.0000 K | 0.0000 K | +0.0000 % |

A replayed rewind reproduces a continuous run **exactly**, at any distance and in
any season, as long as the recorded input history covers the span. That is the
property the whole rolling-reset feature rests on, and it is now measured rather
than assumed.

### Measured cost, on BUI0 (600 s step, 144 ticks/day)

| | time |
| --- | --- |
| free the slave (`terminate` + `freeInstance`) | **3.75 s** |
| instantiate and drive EnergyPlus through its warm-up | **0.55 s** |
| replay one day (144 steps through the socket) | **0.11 s** |

So a rewind of *D* days costs about `4.3 + 0.11·D` seconds, and the teardown is
the floor: `fmi2Terminate` hands control back to EnergyPlus, which runs out the
rest of its run period before exiting. The way to avoid the cost is not to rewind
— `rolling_window` equal to the reset period — or the checkpoint slave sketched in
the follow-ups.

### Six bugs found on the way

1. **`noSetFMUStatePriorToCurrentPoint` was always true.** fmpy defaults that
   `doStep` argument to true, which promises the FMU that the master will never
   put it back before the current point — and every restore this feature performs
   does exactly that. An FMU is entitled to free what a rollback needs once given
   that promise. It is now `not self._can_snapshot`, so only FMUs that are never
   rewound by state get the promise.
2. **Saved states accumulated.** `_take_snapshot` only freed the *same* tick, so a
   rolling run kept one FMU state per episode alive in the FMU's own allocator,
   while the docstring claimed one was kept. The pending rolling start point is
   now taken only when it is not already held and freed the moment a rewind uses
   it (`_consume_rolling_snapshot`), so exactly two exist: the slave's first tick
   and the next start point. The first attempt freed it as soon as a *later* tick
   was passed, which broke the normal case — the reset that consumes a start point
   arrives an episode after the slave has gone past it.
3. **A rolling window longer than the reset period silently degraded.** The next
   start point is saved in passing, so the slave has to reach it before the reset
   that asks for it. When `rolling_window > reset_period` it never does, the
   restore misses, and the rewind falls back to a restart replayed with the
   initial inputs held constant. Now warned at startup. `reset_period` had to be
   plumbed through `ModelConfig` to the model for this.
4. **Three methods were defined twice** (`_check_horizon_granularity`,
   `_warn_rolling_replay_cost`, `_resolve_replay_mode`) — an identical 65-line
   block, silently shadowed. Deleted, along with `_supports_rollback()`, which
   nothing called since capability detection moved to the modelDescription.
5. **Rolling start points were off by one.** `new_starting_point` counted from 0
   while ticks are 1-based, so the start points were `W, 2W, 3W…` instead of
   `1, 1+W, 1+2W…`: every episode re-ran the last tick of the previous window, and
   — visible only once the shift existed — every start point landed one tick
   *before* a day boundary, so a shifted restart still had to replay 143 of 144
   ticks. Now `1 + k × W`, which is also what `_maybe_snapshot_next_rolling_start`
   keys on. **Behaviour change:** every rolling scenario's episode boundaries move
   by one tick.
6. **A reposition that moves nothing still restarted the slave.** With
   `rolling_window == episode_length` the reset asks for the tick the model is
   about to run anyway; the FMU was freed and re-instantiated for it, which cost a
   restart and injected a physical discontinuity into a run that was supposed to
   be continuous — the exact case the docs recommend as free. `_slave_tick` now
   records where the slave stands and `_reposition_backend` returns immediately
   when it is already there.

All six are covered by the existing FMU test files. The rolling-reset scenarios
(`bui_hp_DQN_rollingreset`, `bui_hp_SAC_rollingreset`) matter most here: the
start-point fix moves their episode boundaries by one tick, and both still pass
through the regression suite's runner.

Also simplified: every `doStep` goes through one `_do_step()` rather than three
call sites, which is what made the rollback-promise fix a one-line change.


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
is a few MB; at 1-min steps, tens of MB). It is unbounded: a rewind can ask for
any tick since the run period began. If the history does not cover the requested span
the initial inputs are held constant and the replayed span is **physically
approximate** — a warning says so, but the results do not record it.

**5. EnergyPlus requires whole-day horizons.** `stopTime − startTime` must be a
multiple of 86400 or `fmi2EnterInitializationMode` fails outright. Now warned at
init.

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
*elapsed* simulated time, and §6 establishes that an EnergyPlus FMU's begin date
cannot be moved at runtime, so the only way to align the two calendars is to
export the FMU with the RunPeriod the scenario wants.
