# Model reset redesign — what changed and what it enables

Pair of `model_reset_redesign_plan.md` (design, decisions, per-phase notes F0–F6). This report is the
condensed "what / why / what it enables / what went wrong / what was rejected". Last update: 2026-11-20.

## What changed

- **One mechanism.** A reset is always "restart the model from `init_state` at relative time X".
  `BaseModel._reset(mode, rel_ts_target, span_ticks)` with modes `full`, `rolling` and the internal
  `horizon`; the optional `reset(mode)` hook restores state kept outside `self.state` (solver, cursor).
- **Two clocks.** Absolute `state.ts/time` (monotonic, never moved by a reset) and relative
  `rel_ts/rel_time`. `horizon_pos_s()` folds the relative clock modulo `max_sim_time`; when it enters a
  new window the model restarts itself (`_reset('horizon')`).
- **Units.** RL lengths in the YAML are agent steps; `ScenarioManager.reset_in_federate_ticks` converts
  them to each federate's ticks and **refuses** a length that is not a whole number of that federate's
  `real_period` (an EnergyPlus FMU additionally needs a whole number of days for `rolling_window`).
- **FMU.** Snapshot / replay machinery deleted. A restart = terminate + free the slave, instantiate a new
  one at the current position, with a stop time covering the span to the next restart. All slaves of a run
  write into one `fmu_output/` directory (no `restart_<n>`); only the last restart's E+ output is kept.
- **Parallel execution.** Reset message `('reset', mode, rel_ts_target, span_ticks)` to every worker,
  acked with `'__ack__'`; `ParallelModelExecutor.reset` mirrors `step()` error handling. The main-process
  model copies are not reset (they never step).
- **Scenarios.** `cs_s2_reset_rolling{,_s42,_s43,_s44}`, `cs_s4a_dqn_s42_rolling`, `cs_s4b_sac_s42_rolling`
  back in `src/scenarios/` with `rolling_window: 15` (900 s, a multiple of the 900 s weather period); new
  `combo_parallel_rl_rollingreset` and `bui0_setpoint_DQN_rolling`.
- **Regression suite.** `_shorten` raises the episode cap from 20 to the shortest value that is a whole
  number of every federate's `real_period`; new entries `cs_s2_reset_rolling`, `bui0_setpoint_DQN_rolling`,
  `combo_parallel_rl_rollingreset`; the 4 parquet `KNOWN_FAIL` entries removed (bug no longer reproduces).
- **Tests.** `tests/test_model_reset.py` covers worker reset and ack; the 2 broken tests in
  `test_scenario_manager_remote.py` fixed (their fake `logger_system` lacked `scenario_name`/`run_timestamp`).
- **Docs.** `rl.md`, `federate.md`, `fmu_models.md`, `paper_casestudies.md`, `scenario_tests.md`,
  `Installation_Setup.md` and `developer_guide.md` (EnergyPlus on `PATH`), `KNOWN_ISSUES.md` (§1 and §6
  deleted, §13 note, §14 rewritten), followups renumbered.

## What it enables

- **Rolling-window RL training on FMU/EnergyPlus buildings.** Each episode restarts the building at the next
  window (one day each for E+) without snapshots or replay, so cost per episode is constant instead of
  growing with the distance travelled. Verified: `bui0_setpoint_DQN_rolling`, 4 episodes, start day advances.
- **Resets that reach `parallel_execution` workers.** Before, workers kept running from the old state, so RL
  episodes on a parallel federate silently continued instead of restarting. Verified with 2 workers
  (`combo_parallel_rl_rollingreset`, resets at relative ticks 10 and 20).
- **Mixed-period federations (e.g. 60 s agent, 900 s weather).** A reset length that cannot be honoured by
  every federate is rejected at start-up with the offending federate named, instead of silently
  desynchronising federates mid-training.
- **Runs longer than a model's horizon.** The same restart mechanism handles `max_sim_time` windows, so
  scenarios and training can exceed an FMU's limit without a separate code path.
- **A trustworthy regression gate.** The suite exercises rolling reset, parallel+RL and FMU+rolling, with
  shortened episodes that satisfy the reset-length rule, so a regression in any reset path fails the gate.
  Last full run: 42 PASS, 4 xfail (tracked bugs), 0 FAIL, exit 0; pytest 332 passed, 2 skipped.

## Why

The old code had several partial reset paths (`local_time(mode=...)`, `ts_shift`, snapshots, replay) that
disagreed with each other, silently desynchronised federates with different `real_period`, and never reached
`parallel_execution` workers.

## Decisions (author)

- Restart from `init_state`, no snapshot, no replay.
- EnergyPlus: error (not a workaround) for a mid-day start — proven on BUI0.fmu that E+ rounds `startTime`
  to midnight silently.
- `cs_*` rolling scenarios: `rolling_window` 10 → 15 (600 s was not a multiple of the 900 s weather period);
  results obtained with window 10 are not comparable.
- One `fmu_output/` per run: hundreds of episodes must not leave hundreds of `restart_<n>` directories.
  Cost: only the last restart's E+ output survives. Revert path if a slow slave shutdown ever collides with
  the next start: `restart_<n>` in `_resolve_fmu_workdir` (not needed in the verified runs).
- The tick that publishes `init_state` after a reset stays a documented limit (KNOWN_ISSUES §5).

## Problems hit

- Parallel workers re-import modules from disk (spawn): tests need an importable module, and editing
  `parallel_executor.py` while pytest runs breaks the workers.
- E+ scenarios hung on `eclab-cloud1`. First diagnosed as an IPv6-only hostname (wrong: `HOSTALIASES` and a
  `gethostname` shim changed nothing); the real cause was no `energyplus` on `PATH` (the FMU waits in
  `accept()` with no child process and no error). Installed 23.1.0 under `~/opt` → KNOWN_ISSUES §14.
- Redis could not write its dump (`/data` permission denied → `MISCONF`); worked around at runtime with
  `stop-writes-on-bgsave-error no` (lost on container restart).
- Distributed scenarios failed only for SSH to `127.0.0.1`: dedicated key `~/.ssh/cosim_localhost`, its
  `authorized_keys` entry and a `Host 127.0.0.1` stanza fixed it (KNOWN_ISSUES §13, machine setup).

## Rejected alternatives

- Resetting the main-process model copies in parallel mode: they never step, and it would restart a useless
  FMU instance per episode.
- Global `_shorten` cap of 30: silently changes every RL scenario; replaced by a per-scenario minimum.
- `real_period: 600` for the weather federate to keep window 10: changes the input data.
- Extra "dry" step or `init_state` computed at date X to fix the init tick: new feature / HELICS timing
  change; deferred (`fmu_horizon_and_reset_followups.md` §5).

## Verification (2026-11-20, `eclab-cloud1`, E+ 23.1.0 on `PATH`)

- Full regression suite: 42 PASS, 4 xfail, 0 FAIL (unit tests 332 passed, 2 skipped; 3 distributed
  scenarios included).
- E+ restarts (`bui0_setpoint_DQN_rolling`): one shared `fmu_output/`, last slave starts 3 Jan; the only
  severe E+ message is the known final socket `-1` (KNOWN_ISSUES §5).
- Not checked: a long E+ run (hundreds of episodes) for accumulated cost; the `.eio` start-day check is
  manual, not a suite assertion.
