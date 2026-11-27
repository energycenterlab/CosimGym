# Known Issues & Limitations

**Single living register.** Every known bug, limitation or environment gotcha lives here
and nowhere else. Add an entry the moment one is found; **delete the entry** the moment it
is fixed or stops being a problem. This file is a snapshot of *what is broken now* — it is
not a changelog, so nothing stays here for history. Fixed items are described in the
matching changes report under `docs/changes_reports/`.

Scope boundaries:

- **Here:** framework/environment bugs, hard limitations, unsupported combinations.
- **Not here:** planned enhancements and deferred work → `docs/future_and_TODOs/`.
- **Not here:** what a change did and why → `docs/changes_reports/`.

Entry format: `## <n>. <one-line title>  (SEVERITY)`, then what breaks, how it is
reproduced, what is known about the cause, and the workaround if any. Severities:
`HIGH` (blocks a documented feature) / `MEDIUM` (blocks some configurations) /
`LOW` (cosmetic, or an unfinished optional piece).

Scenarios blocked by an entry here are listed in `tests/regression_suite.py`'s
`KNOWN_FAIL` dict, so the pre-merge gate stays green while the bug is tracked. A
`KNOWN_FAIL` that starts passing is reported as **UNEXPECTED-PASS** — when that happens,
remove it from `KNOWN_FAIL` *and* delete the entry here.

Last reviewed: 2026-11-19.

---

## 2. zmq auto-port allocation ignores the paired `port+1`  (MEDIUM)

`ScenarioManager._get_n_available_tcp_ports()` hands out N single free ports, but a `zmq`
broker occupies both `port` and `port+1` (`_broker_ports()`). When ≥2 zmq brokers are
auto-assigned (e.g. any 2-federation RL scenario: the hierarchy broker + the framework-created
`rl_federation` broker, whose port is hardcoded `None` in `_create_RL_federation()`), the
allocator can hand out two adjacent ports; the first broker's `port+1` then collides with the
second broker's advertised port and `_assert_broker_ports_free()` aborts. Deterministic on
`pv_batt_DQN` / `pv_batt_SAC`; probabilistic elsewhere (`simple_DQN_test` happened to get
non-adjacent ports and passed).

- **Fix:** reserve `port+1` for `core_type: zmq` inside `_get_n_available_tcp_ports` (or exclude
  odd/even adjacency).
- **Workaround:** use `core_type: tcp` (single port, no `+1`) or `zmq_ss`.

## 3. `RL_Simple_Agent` catalog model is a non-functional skeleton  (LOW)

`src/models/model_catalog/RL_agents/rl_simple_agent.py` `online_training_loop`/`testing_loop`
just call `super().*()`, whose base loop calls `self.env_step`, a method that exists nowhere in
the hierarchy → `AttributeError: 'RL_Simple_Agent' object has no attribute 'env_step'`. The
file's docstring calls it a template. Blocks `simple_test_rlagent` (its only user).

- **Fix:** finish the skeleton or drop the model and its scenario.
- **Workaround:** use `rl_simple_SACsb3`, `rl_simple_DQN` or `rl_simple_rllib`.

## 4. `Adelaide_test` — missing MinIO object  (LOW, data not code)

`my_fmu_federate` fails with `minio ... NoSuchKey` for
`fmus/adelaide_test/1.0.0/PCMA_1_0_control_2022.fmu`. The `.fmu` exists on disk at
`src/models/model_catalog/physical_models/resources/PCMA_1_0_control_2022.fmu` but was never
uploaded to the local MinIO `fmus` bucket.

- **Fix:** upload it to the bucket, or fix the `catalog.yaml` reference. Not a scenario-YAML fix.

## 5. Model reset & horizon restart — remaining defects  (MEDIUM)

Two independent concepts share the code: the RL **episode reset** and the model's own
**horizon restart** (`max_sim_time`). The refactor is nearly done (plan:
`docs/changes_reports/model_reset_redesign_plan.md`, design context:
`docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`). What is left:

- `RL_Federate` has its own loop and does no horizon handling.
- Horizon restart is per model (`BaseModel._step`), so models with different horizons restart
  out of phase (the federate only warns).
- After a full/rolling reset the federate publishes `init_state.outputs` for one tick instead of
  the values at the new restart instant (CSV: row 0; FMU: `TBuilding` at its default)
  (`fmu_horizon_and_reset_followups.md` §5).
- The last EnergyPlus slave of a run ends with `Socket communication received error value "-1"`
  in its `.err` (terminated by `finalize` without its closing handshake); its own `.csv`/`.eso`
  are not written. Model outputs are unaffected. Pre-existing.

## 7. Loading `helics` before `libstdc++` → SIGSEGV in torch / ray  (LOW, mitigated, environment)

If `helics` is the first module in a process to load a C++ library, a later `import torch`
or `import ray` segfaults (exit 139). Repro: `python -c "import helics, torch"` (or
`"import helics, ray"`). Any import that loads the env's `libstdc++.so` *before* helics
(numpy, pandas, torch, …) avoids it: `python -c "import numpy, helics, torch, ray"` works.

- **Cause:** the PyPI helics wheel's `libhelics.so` is built on CentOS 7 with GCC's
  *static* libstdc++ and **exports** those C++-runtime symbols (old-ABI `std::string`
  internals, locale facet ids such as `std::__cxx11::numpunct<char>::id`, `std::piecewise_construct`).
  cffi loads it into the global symbol scope, so when the real `libstdc++.so.6` is loaded
  afterwards, ~120 of its own symbols (and a few of `libc10`/`libtorch_cpu`'s) bind to
  helics' copies (`LD_DEBUG=bindings`). The process then runs two half-merged C++ runtimes;
  torch crashes during operator registration, ray while dlopen-ing `_raylet`.
- **Not version-specific:** helics 3.5.3 crashes the same way; torch 2.5.1 avoids the
  segfault only by failing with `GLIBCXX_3.4.29 not found` instead. Pinning torch does not fix it.
- **Mitigated in the framework (2026-11-19):** `BaseFederate.py`, `RL_Federate.py` and
  `InterfaceFederate.py` import `numpy` right before `helics`, inside `# isort: off/on` so an
  import sorter cannot undo it. Any *new* entry point that imports `helics` before numpy and
  later loads torch/ray will still crash — follow the same pattern.
- **Root fix (upstream):** the pyhelics wheel should not export its static libstdc++ symbols.

## 8. `synchronization.default_subscription_causality` is ignored  (MEDIUM)

Setting `default_subscription_causality: next_step` has no effect: `FedSubscription.causality`
already defaults to `"same_step"` (`src/utils/config_dataclasses.py:184`), so
`_normalize_subscription_causality` (`src/core/ScenarioManager.py:1261`) never falls back
to the scenario default.

- **Workaround:** set `causality:` on each subscription.
- **Fix:** detail and plan in `docs/future_and_TODOs/synchronization_causality_plan.md` §3.

## 9. RL scenario residual bugs (from code reading, not re-run)  (MEDIUM)

Detail in `docs/future_and_TODOs/scenario_yaml_TODO.md`.

- **A — `simple_DQN_test` crashes at the first episode boundary:** no `experiment.checkpoint`
  block → `save_model(None)` → `TypeError`. Masked in `tests/regression_suite.py`, whose
  `_shorten` forces a checkpoint name, so the suite PASS does not cover the real file.
- **B — `bui0_setpoint_SAC` reward is always 0:** the reward reads
  `feeder_federate.0.ZoneSetPoint`, which the SAC scenario does not observe, and the bare
  `except Exception: return 0.0` in `reward_functions.py` hides the `KeyError`
  (`bui0_setpoint_DQN` was fixed in YAML).
- **D — `simple_rllib_test`:** spring federate sets `core_type: zmq` against a `tcp` broker;
  harmless override warning every run.

## 10. RL configuration keys parsed but not implemented  (LOW)

`role: extra` observations still reach SB3/RLlib (`KeyError`; use `role: state`), `run.eval`,
`experiment.logging`, `history` (frame stacking) and `run.mode: offline|mixed` parse but do
nothing. Table in `docs/user_guide/scenario_configuration/rl.md` → Known Limitations.

## 11. Teardown stall at large scale / heavy `_ss` exchange  (HIGH for scaling runs)

The simulation reaches its last tick but the run never returns: one side force-disconnects,
the other blocks in HELICS forever, and the stranded brokers/federates keep their ports, so
every following run fails with "port already in use". Seen with local plain zmq at ≥256
federates and with distributed `zmq_ss` at ≳1 kB exchanged per tick. Cause not diagnosed.

- **Workaround:** stay under ~256 local federates / ~1 kB per tick distributed; after a stall
  `pkill -f federate_launcher` and kill leftover `helics_broker`s.
- Detail: `scripts/scaling_study/findings/bottlenecks.md` B12 (observed in the exploratory scaling runs; to be re-confirmed).

## 12. `parallel_execution` workers are not sized against the machine  (LOW)

Each federate spawns its own `W` workers, so `N·W` can exceed the core count many times
over (1760 processes on 112 cores measured) and parallel runs lose their speed-up. Nothing
warns. **Workaround:** keep `N·W ≲ cores`, or use `mode: seq` at large N.
Detail: `scripts/scaling_study/findings/bottlenecks.md` B13 (observed in the exploratory scaling runs; to be re-confirmed).

## 13. New machine: distributed runs fail until SSH and remote interpreters are set up  (MEDIUM, setup)

Every distributed scenario — including the localhost-as-remote ones in the regression suite
(`distributed_demo`, `distributed_multifederation_test`, `combo_dist_parallel`) — needs
passwordless SSH from the manager to each machine, **including `127.0.0.1`**. On a fresh machine
this is not set up and the run aborts with `Host key verification failed` or a password refusal
(`BatchMode=yes`). Seen on `eclab-cloud1`, 2026-11-19; fixed there on 2026-11-20 (dedicated key `~/.ssh/cosim_localhost` + `Host 127.0.0.1` entry in `~/.ssh/config`).

- **Fix per machine:** follow [Distributed Deployment → Adding a machine](user_guide/distributed_deployment.md#adding-a-machine-checklist).
- Loopback "remotes" now reuse the manager's interpreter, and local FMU paths in `catalog.yaml`
  are repo-relative, so neither needs editing per machine any more.
- Still machine-specific: the `python:` paths of the real cloud machines in
  `distributed_demo_multi`, `benchmark_scale_distributed*`, `cs_s4c_shard_*` (only run with
  `RUN_CLOUD=1`; check them when the remote machines change).

## 14. EnergyPlus FMU scenarios hang when `energyplus` is not on PATH  (MEDIUM, machine setup)

The `BUI0.fmu` shells out to an `energyplus` executable found on `PATH` and then waits on a socket
for it. If it is missing (as on a fresh `eclab-cloud1`) the FMU federate idles in `accept()` until the
suite's 240 s timeout, with no error message and no child process. Fix: install EnergyPlus 23.1.0 and
put it on `PATH` (verified without root: unpack the NREL Linux tarball in `~/opt` and prefix `PATH`).
The hostname/IPv6 theory recorded earlier was wrong (a shim giving `127.0.0.1` did not help).

## 15. EnergyPlus model (Python API): unsupported or unverified cases  (MEDIUM)

`EnergyPlusModel` (see [EnergyPlus Models](user_guide/eplus_models.md)):

- **EnergyPlus 26.x IDFs cannot run here** (Ubuntu 20.04 / glibc 2.31): the four `2_zones_*` IDFs fail at start with a clear
  "no EnergyPlus >= 26.1" error. Detail: `docs/future_and_TODOs/eplus_model_followups.md`.
- Only `.idf` with a single, first `RunPeriod`; `.imf` (EPMacro) and `.epJSON` are rejected. A blank RunPeriod year is assumed to be 2001 (non-leap).
- IDFs with a blank "Day of Week for Start Day": after a restart on another date the weekday of day 1 is not controlled (warned).
- Restarting at another time costs a full start-up (sizing + warmup, ~41 s on `EC_IdealLoads`) and the thermal history is only recovered with `burn_in_days`.
- Unverified with a real run: PythonPlugin (level 3), `ExternalInterface:To:Variable` (EMS global), meters absent from the IDF listing.
- `real_period` that is not a whole multiple of the IDF zone timestep is refused.
- Inputs only exist if the IDF exposes an actuator / schedule / EMS global for them; the model does not write EMS code (a name that cannot be resolved fails at start-up, after sizing).
- Key patterns (`*`) need one extra EnergyPlus run per IDF (cached); meters and globals are not expanded.
- BaseModel logs a "not defined in catalog, Not initialized" warning per `eplus_io` name (cosmetic).
