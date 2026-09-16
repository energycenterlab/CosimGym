# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session)

Answer a design question about FMU restarts, then implement the answer: can an EnergyPlus
FMU be restarted at an arbitrary point *without* re-simulating everything before it —
by giving it a new start time and re-imposing the state it had? And, having opened that
code, fix what is broken in it and clean it up.

## Current Progress — done, uncommitted

### The question, answered

**No, not as posed**, and both blockers were verified in the FMU binary rather than
assumed:

- **FMI `startTime` does not move an EnergyPlus calendar.** `BUI0.so` uses it only for the
  whole-day check and the first-communication-point check; it contains no date formatting
  and never writes a `RunPeriod`. The calendar lives in the IDF.
- **There is no state to re-impose.** BUI0's interface is six `To:Schedule` inputs (gains,
  set-point) and two `From:Variable` outputs. Zone temperature is an *output*; EnergyPlus
  exposes no actuator for zone air or surface node temperatures.

**But the neighbouring idea works**: the wrapper re-runs its preprocessor over the FMU's
`resources/` folder at *every* instantiation, so rewriting the RunPeriod's begin date in
the unzipped IDF makes the next slave start on another date. That is the new
`restart_strategy: runperiod_shift`.

### Shipped

- **`restart_strategy: runperiod_shift` + `spinup_days`** in `BaseFMUModel`, opt-in per
  model via `fmu_reset` (catalog entry, or a scenario's `model_configs.user_defined`,
  which now overrides the catalog key by key). A rewind is split into whole days handed to
  EnergyPlus by moving its begin date and a spin-up remainder replayed with recorded
  inputs. Cost stops depending on the distance rewound.
- **Measured on BUI0** (600 s step): teardown 3.75 s, instantiate-on-shifted-date 0.55 s,
  replay of one day 0.11 s. A ten-month-deep rewind goes from ~40 s of replay to ~0.7 s.
- **Six bugs fixed** (full account in the changes report §6): `noSetFMUStatePriorToCurrentPoint`
  was always promised true while the code rolled back; saved FMU states accumulated one per
  episode; `rolling_window > reset_period` degraded silently; three methods were defined
  twice; rolling start points were off by one (`W, 2W…` instead of `1, 1+W…`); and a
  reposition that moves nothing still restarted the slave.
- **Docs updated**: `docs/user_guide/fmu_models.md` (§3 start date, §4 restart strategies
  and the honest description of the warm-up trade), `scenario_configuration/rl.md` (rolling
  start-point arithmetic), `scenario_configuration/federate.md` (which `user_defined` keys
  the framework reads), `docs/KNOWN_ISSUES.md` (§5 amended, §6 added),
  `docs/changes_reports/fmu_horizon_and_reset_changes.md` (§6, new soft spots),
  `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md` (§2, §3 revised; §7, §8 new).
- **Tests**: `tests/test_fmu_runperiod_shift.py` (13 tests, incl. one real EnergyPlus round
  trip), scenario `src/scenarios/fmu_rolling_shift_smoketest.yaml`, registered in
  `tests/regression_suite.py` as "FMU RunPeriod shift".

### Verified

- `tests/test_fmu_runperiod_shift.py` 13 passed; snapshot/clock/horizon tests 35 passed.
- Through the regression suite's runner: `fmu_horizon_smoketest`,
  `fmu_rolling_shift_smoketest`, `bui0_fmu_test`, `bui_hp_DQN_rollingreset`,
  `bui_hp_SAC_rollingreset` — **all PASS** (the last two matter: the start-point fix moves
  their episode boundaries by one tick).
- Scenario log shows the begin date moving 1, 1 and 2 days with only the remainder replayed.

## What Worked

Reading the FMU **binary and its run directory** instead of reasoning about FMI in the
abstract. `strings BUI0.so` gave the two error messages that settle what `startTime` does,
and comparing `resources/BUI0.idf` (63 973 B) with the run directory's preprocessed
`BUI0.idf` (14 760 B) proved the IDF is re-read at every instantiation — which is the whole
mechanism the feature rests on.

Measuring the three costs (teardown / restart / replayed day) before writing a word about
them, so the cost claims in the docs are numbers rather than adjectives.

## What Didn't Work / Watch Out

- **Freeing a rolling snapshot as soon as a later tick is passed** broke the normal case:
  the reset that consumes a start point arrives an *episode* after the slave has gone past
  it. The pending start point must live until a rewind actually uses it.
- **Running the RL rolling scenarios unshortened** (`bui_hp_DQN_rollingreset` is 100 × 2880
  steps) was a mistake — killed it and used `regression_suite.run_scenario`, which shortens
  on a temp copy. Use that for any ad-hoc scenario check.
- `tests/test_scenario_manager_remote.py` has **2 tests failing on clean `main`**
  (`AttributeError: 'types.SimpleNamespace' object has no attribute 'scenario_name'`).
  Pre-existing, logged as `docs/KNOWN_ISSUES.md` §6, not touched.
- Full `pytest tests/` (excluding the regression suite): **328 passed, 2 skipped, 2 failed**
  — the two failures are the pre-existing `test_scenario_manager_remote.py` ones above.

## In-flight, uncommitted state

Branch `main`, nothing committed (this session's work sits on top of the previous
session's FMU horizon/reset work, which was also uncommitted).

- modified: `src/models/base_FMU_model.py` (the bulk), `src/models/base_model.py`,
  `src/core/BaseFederate.py`, `src/utils/config_dataclasses.py`,
  `tests/{regression_suite,test_fmu_state_snapshot}.py`, and the six doc files listed above
- new: `src/scenarios/fmu_rolling_shift_smoketest.yaml`, `tests/test_fmu_runperiod_shift.py`
- `graphify-out/` regenerated (`graphify update .`)

## Next Steps

1. **Validate the warm-start approximation** — `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`
   §7 specifies the experiment (continuous 60-day run vs shifted restart with and without
   spin-up, on BUI0 and on the PCM `adelaide_test`). Until it is run, `runperiod_shift`
   belongs in training, not in validation runs. The scaffolding is in
   `tests/test_fmu_runperiod_shift.py`; it needs a go-ahead before the longer runs.
2. Bound each shifted instance's run period by moving the **end** date too — followups §8.
   It would cut the 3.75 s teardown that every restart currently pays; it has to be
   reconciled with `max_sim_time` and the horizon restart first.
3. Still open from before: `parallel_execution` propagates no reset to worker processes
   (followups §6), and a restart is not signalled to the RL agent as `truncated` (§4).
4. Run the pre-merge gate before merging anything:
   `conda run -n cosim_gym python tests/regression_suite.py`.
5. Nothing is committed without an explicit ask — the FMU horizon/reset work from the
   previous session is still uncommitted in the same tree.
