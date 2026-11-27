# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file live in `CLAUDE.md` → **Working Rules**. Read that first.

---

## Goal (sessions 2026-11-19/20, machine `eclab-cloud1`)

Finish the reset refactor (`docs/changes_reports/model_reset_redesign_plan.md`): F5 (parallel workers), F6 (closure),
then verify everything and align docs. **Status: plan DONE, everything verified.**

## Current progress — all UNCOMMITTED (the author commits)

- **F5:** `parallel_executor.py` reset message + ack, `BaseFederate._reset` forwards to workers only, tests, docs.
- **F6:** `cs_*` rolling scenarios back in `src/scenarios/` with `rolling_window: 15`; one `fmu_output/` per run
  (`base_FMU_model._resolve_fmu_workdir`); `init_state` tick limit documented (KNOWN_ISSUES §5); suite `_shorten`
  episode cap = shortest multiple of every federate's `real_period`; new scenarios `bui0_setpoint_DQN_rolling`,
  `combo_parallel_rl_rollingreset` (+ suite entries incl. `cs_s2_reset_rolling`).
- **Verification pass (2026-11-20):**
  - Full `tests/regression_suite.py` with E+ on PATH: **42 PASS, 4 xfail, 0 FAIL, exit 0** (unit 332 passed, 2 skipped).
  - E+ hang root cause was **`energyplus` not on PATH** (not the hostname). Installed 23.1.0 in `~/opt/eplus23`; run E+ scenarios with
    `PATH=$HOME/opt/eplus23/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu20.04-x86_64:$PATH`. KNOWN_ISSUES §14 rewritten.
  - `bui0_setpoint_DQN_rolling`: one shared `fmu_output/`, start day advances (last slave 3 Jan).
  - Parquet xfail scenarios now pass (2/2 reruns): removed from `KNOWN_FAIL`, KNOWN_ISSUES §1 deleted.
  - 2 broken remote tests fixed (fake `logger_system` in `tests/test_scenario_manager_remote.py:56`), KNOWN_ISSUES §6 deleted.
  - Distributed scenarios pass after SSH setup for `127.0.0.1`.
- **Docs aligned:** `Installation_Setup.md`, `developer_guide.md`, `fmu_models.md`, `CLAUDE.md` (E+ on PATH prerequisite),
  banner of `fmu_horizon_and_reset_changes.md`, new `model_reset_redesign_changes.md` (incl. "what it enables").
  graphify graph is stale for these docs (`--update` only if the author asks).

## Environment changes made outside the repo (this machine only)
- `~/opt/eplus23/` (EnergyPlus 23.1.0, from NREL GitHub release).
- `~/.ssh/cosim_localhost{,.pub}`, its public key appended to `~/.ssh/authorized_keys`, `Host 127.0.0.1 localhost` stanza in
  `~/.ssh/config`, 127.0.0.1 host key in `~/.ssh/known_hosts` (all authorized by the author).
- Redis `stop-writes-on-bgsave-error no` set at runtime (container volume `/data` not writable); lost on container restart.

## What worked / didn't
- `HOSTALIASES` and an `LD_PRELOAD gethostname` shim did NOT fix the E+ hang (wrong theory).
- Don't `pkill -f <pattern>` from a command containing that pattern; use `[h]elics_broker` style patterns.
- Suite deletes `logs/_regr_*` after each run; monkeypatch `regression_suite.shutil.rmtree` to keep logs.
- Worker test dummy needs `module_path='test_model_reset'`.

## Next steps
1. Author: review and commit (suggest splitting: F5 code+tests, F6 scenarios/suite, docs).
2. Optional: a long E+ rolling run (hundreds of episodes) to check the shared `fmu_output/`; an `.eio` start-day assertion in the suite.
3. Open from earlier sessions: machine-specific `python:` paths in RUN_CLOUD scenarios; `/media/space` in
   `scripts/scaling_study/plot_*.py`; 9 latent bugs (native memory) not in KNOWN_ISSUES; `core_type` "Overriding" warning;
   Mac conda env recreate.

## Uncommitted files
CLAUDE.md, HANDOFF.md, docs (Installation_Setup, developer_guide, KNOWN_ISSUES, plan, changes reports, followups, fmu_models,
federate, paper/scenario md), src/core/{BaseFederate,parallel_executor}.py, src/models/base_FMU_model.py, src/scenarios
(cs_* moved back from `to_review_rolling_window/`, 2 new), tests/{regression_suite,test_model_reset,test_scenario_manager_remote}.py.
