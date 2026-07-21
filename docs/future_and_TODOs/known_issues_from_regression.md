# Known Issues Surfaced by the Regression Suite (2026-07-21)

Found while building `tests/regression_suite.py` (verified all 45 scenarios). These are
**framework / environment bugs**, not scenario-config bugs — the regression suite runs the
affected scenarios as `xfail` (expected-fail) so the green gate is preserved while they are
tracked here. Fixing them is a code change (out of scope for the scenario-verification pass).

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
- NOTE: the user reports parquet working elsewhere — so this may be specific to this machine's
  pyarrow / libstdc++ build. Worth pinning the pyarrow + libstdc++ versions where it works vs here.
- Unit tests `tests/test_parquet_storage.py` / `tests/test_async_storage.py` pass — the schema/
  writer logic is fine in-process; the crash is in the federate-subprocess runtime.

## 2. zmq auto-port allocation ignores the paired `port+1`  (MEDIUM)

`ScenarioManager._get_n_available_tcp_ports()` hands out N single free ports, but a `zmq`
broker occupies both `port` and `port+1` (`_broker_ports()`). When ≥2 zmq brokers are
auto-assigned (e.g. any 2-federation RL scenario: the hierarchy broker + the framework-created
`rl_federation` broker, whose port is hardcoded `None` in `_create_RL_federation()`), the
allocator can hand out two adjacent ports; the first broker's `port+1` then collides with the
second broker's advertised port and `_assert_broker_ports_free()` aborts. Deterministic on
`pv_batt_DQN` / `pv_batt_SAC`; probabilistic elsewhere (`simple_DQN_test` happened to get
non-adjacent ports and passed).

- Fix: reserve `port+1` for `core_type: zmq` inside `_get_n_available_tcp_ports` (or exclude
  odd/even adjacency). Workaround: use `core_type: tcp` (single port, no `+1`) or `zmq_ss`.

## 3. `RL_Simple_Agent` catalog model is a non-functional skeleton  (LOW)

`src/models/model_catalog/RL_agents/rl_simple_agent.py` `online_training_loop`/`testing_loop`
just call `super().*()`, whose base loop calls `self.env_step`, a method that exists nowhere in
the hierarchy → `AttributeError: 'RL_Simple_Agent' object has no attribute 'env_step'`. The
file's docstring calls it a template. Blocks `simple_test_rlagent` (its only user). Either
finish the skeleton or drop the scenario.

## 4. `Adelaide_test` — missing MinIO object (DATA, not code)

After YAML fixes (federate count, malformed key — done), `my_fmu_federate` fails with
`minio ... NoSuchKey` for `fmus/adelaide_test/1.0.0/PCMA_1_0_control_2022.fmu`. The `.fmu`
exists on disk at `src/models/model_catalog/physical_models/resources/PCMA_1_0_control_2022.fmu`
but was never uploaded to the local MinIO `fmus` bucket. Upload it (or fix the `catalog.yaml`
reference) — not a scenario-YAML fix.
