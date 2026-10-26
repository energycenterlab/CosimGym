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

Last reviewed: 2026-10-26.

---

## 1. Parquet sink → native SIGSEGV in a federate process  (HIGH)

Any scenario with `memory_config.sink: parquet` crashes a federate with a bare `SIGSEGV`
(exit code `-11`, `error 6 in libstdc++.so.6.0.34`, dmesg-confirmed) — no Python traceback.
The identical scenario with `sink: json` completes cleanly. Reproduced on 4 unrelated
scenarios (`rc_building_parquet_test`, `m2_interface_outbound_smoke_test_parquet`,
`fmu_feedthrough_test`, `stress_multi_building_parquet`) and 2 combo scenarios.

- Crash correlates with the async parquet writer path: `src/utils/async_storage.py`
  (`AsyncStorageWriter`) + `src/utils/parquet_storage.py` (`ParquetStorageWriter`, pyarrow
  `ParquetWriter`). Seen both at writer-thread startup (`weather_csv_reader` federate log empty
  → crash at import/thread spawn) and at close/finalize (interface federate crashes right after
  its last step, no finalize log line).
- `import pyarrow` / `import helics` (separately and together) work fine → not a missing package.
  Most likely a native-library interaction (Arrow C++ vs HELICS bundled libs) that only triggers
  when the writer thread runs inside a federate subprocess.
- The user reports parquet working elsewhere — so this may be specific to this machine's
  pyarrow / libstdc++ build. Worth pinning the pyarrow + libstdc++ versions where it works vs here.
- Unit tests `tests/test_parquet_storage.py` / `tests/test_async_storage.py` pass — the schema/
  writer logic is fine in-process; the crash is in the federate-subprocess runtime.
- **Workaround:** use `sink: json`.

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

## 5. FMU / model reset & horizon restart — mid-refactor defects  (HIGH)

Two independent concepts share the same code: the RL **episode reset** (`full` / `rolling` /
`none`) and the model's own **horizon restart** (`max_sim_time`, any model with an intrinsic
limit, RL or not). The refactor of both (`208f9fb`) is unfinished. Defects verified in the
source on 2026-10-26 (design context: `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`):

- `BaseFMUModel` keeps one `self.snapshot` and uses it after `freeInstance` of the slave that
  produced it; `_start_instance` overwrites it on every restart.
- The `state.ts > 1` restore branch in `_push_initial_state_to_fmu` is unreachable:
  `BaseModel._reset` deep-copies `init_state` (ts = 0) before calling `reset()`.
- `_maybe_snapshot_next_rolling_start`, `_free_snapshot`, `_free_all_snapshots` and
  `_consume_rolling_snapshot` use `_state_snapshots` / `_snapshot_target_ts`, which no longer
  exist: rolling + a state-capable FMU raises `AttributeError` on the first step.
- `_stop_time_seconds()` returns `max_sim_time` as an absolute stop time even when the start
  time is > 0, which breaks the whole-day (86400 s) experiment duration an EnergyPlus FMU needs.
- `reset('rolling')` always tears the slave down, including when `rolling_window == episode_length`
  (nothing needs to move).
- Horizon restart is per model (`BaseModel._step`), so models with different horizons restart
  out of phase (the federate only warns).
- `parallel_execution` never propagates a reset to its worker processes; `RL_Federate` has its
  own loop and does no horizon handling.
- Stale: `tests/test_fmu_state_snapshot.py`, `tests/test_fmu_horizon_detection.py` reference the
  removed API; dead code and commented blocks remain in `base_FMU_model.py`, `base_csv_reader.py`.

## 6. `tests/test_scenario_manager_remote.py` — 2 tests broken  (LOW, test-only)

`TestRemoteFederates::test_verify_and_deploy_called_for_each_machine` and the preflight-failure
test next to it both die with
`AttributeError: 'types.SimpleNamespace' object has no attribute 'scenario_name'` at
`src/core/ScenarioManager.py:870`: the fake config the tests build no longer carries every
attribute the deploy path reads. Pre-existing on `main`, unrelated to the distributed feature
itself (the real `distributed_demo` scenario passes).

- **Fix:** add `scenario_name` (and whatever else has been added since) to the test's
  `SimpleNamespace`, or build the fake config from the real dataclass.
