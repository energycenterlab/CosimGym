# HANDOFF: Distributed SSH Federate Spawning

Branch: `distributed-ssh-spawning-plan`. **NEVER commit — ask user first.**

## Goal

Let `ScenarioManager` spawn federates on remote machines over SSH, declared in scenario
YAML (`deployment:` block + per-federate `host:` key). Brokers stay local. No `host:` =
byte-identical local behavior. Full spec + rationale: `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`
(read this first, it is the source of truth — decisions there are user-approved, do not
relitigate). Alternatives considered: `distributed_ssh_spawning_alternatives.md` same folder.

## Current Progress

**ALL TASKS DONE — T1–T7 complete, live localhost-as-remote E2E PASSED.**

**T6 live E2E result (2026-07-15):** ran `python src/verify_distributed_demo.py` on this machine
with passwordless ssh to 127.0.0.1 + the cosim docker stack up. `distributed_demo` (pv_federate
spawned remotely over ssh) completed: 1 broker, 5 federates. **2304/2304 recorded timeseries
records matched the all-local twin `pv_batt_test_base` within 1e-9.** Remote federate results were
rsynced back into the manager's local `results/` tree (T5 collection verified). Mid-simulation
SIGINT (remote federate confirmed alive first, then interrupted) → remote federate GONE, zero
orphans local & remote (T5 `_cleanup_remote_execution` pkill + master close verified).

**Two bugs found during E2E and fixed:**
1. `remote_executor.py` `_master_cmd()`: used `ssh ... -nNf` (background-fork master). With
   `subprocess.run(capture_output=True)` the backgrounded ssh child inherits the stdout/stderr
   pipes → `run` blocks on them until the 15s timeout even after the connection succeeds
   (`open_master` always timed out). **Fix:** run a trivial `true` over the connection instead of
   `-nNf`; `ControlMaster=auto` + `ControlPersist=60` already keep the master socket alive, and the
   client exits immediately (still captures stderr on auth failure). Test `test_master_cmd_flags`
   updated (`-nNf` → asserts `true` last token, `-nNf` absent).
2. `ScenarioManager.start_scenario()`: on SIGINT during setup the signal handler calls `exit(0)`
   (`SystemExit`, caught by neither `except KeyboardInterrupt` nor `except Exception`), so the
   `finally`'s `_emergency_cleanup(success=success)` hit `UnboundLocalError: cannot access local
   variable 'success'`. **Fix:** initialize `success = False` before the `try`.

**Env note for reproducing the E2E:** scripts must run with the `cosim_gym` env bin on `PATH`
(i.e. `conda activate cosim_gym`, not just calling the env's python by absolute path) — the manager
spawns `helics_broker` (a CLI in the env bin) as a subprocess; without it on PATH, setup fails with
"helics_broker executable not found". Also note the demo's `conda run -n cosim_gym` resolves fine
over ssh on this machine (no `python:` escape hatch needed).

Full suite: `pytest tests/ -q` → **156 passed, 2 skipped** (includes T5's 5, ports' 10, updated
remote-executor master test). `graphify update .` run.

---

**T1, T2, T3, T4: DONE. T5 — Collection + cleanup: DONE. T7 — Docs: DONE.**

Full suite after T5: `pytest tests/ -q` → **145 passed, 2 skipped** (140 + 5 new T5 tests).

**T5 details:**

- `src/core/ScenarioManager.py`:
  - `start_scenario()`: added `self._collect_remote_results()` right after `_monitor_processes()`
    returns on the success path (before `simulation_end` metrics) — federates have all exited, so
    `sink: json` is written / `sink: parquet` finalized, safe to pull back.
  - `_emergency_cleanup()`: added `self._cleanup_remote_execution()` after the broker-kill loop,
    before `cleanup_end` metric. Runs on every exit path (success, exception, signal, atexit) via
    the existing once-only `_cleanup_done` guard.
  - New `_collect_remote_results()`: no-op if `remote_executors` empty. For each remote machine,
    `executor.collect()` the remote `results/<scenario>/<sim_id[-15:]>/` and
    `logs/<scenario_log_dir_rel>/` back into the identical local dirs (rsync, no `--delete` →
    collision-free merge since each federate writes distinct files). Any failure → ERROR log with
    a ready-to-paste manual `rsync` command (built from `machine_conf.user`/`host`), never raises —
    valid remote results must not turn a good run into a reported failure.
  - New `_cleanup_remote_execution()`: no-op if empty. Per machine: `executor.run(['pkill','-f',
    self.simulation_id])` (full sim id = unique per run → only this run's federates; rc=1/no-match
    ignored) then `executor.close()`, each wrapped in try/except → teardown never raises nor masks
    the original error. Clears `self.remote_executors`.
- `tests/test_scenario_manager_remote.py`: added `TestCollectionAndCleanup` (5 tests): collect
  no-op when local, collect issues 2 rsyncs (results+logs) with correct paths, collect swallows
  rsync failure, cleanup pkills `simulation_id` + closes + clears, cleanup never raises on ssh
  failure. Uses the same `object.__new__` fake-manager pattern; sets `simulation_id`/`scenario_name`.
- `graphify update .` run.

**T7 details:**

- New `docs/user_guide/distributed_deployment.md`: what-goes-where (brokers local, federates
  remote), one-time remote setup (ssh keys, conda env / `python:` escape hatch, firewall ports),
  full YAML reference, run lifecycle (preflight→deploy→spawn→collect→cleanup), localhost-as-remote
  demo instructions, troubleshooting table, LAN-trust security note, v1 limitations.
- `CLAUDE.md` Config Reference: added terse `deployment` block entry + `host` federate key.
- `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`: status header updated to
  "T1–T5 + T7 implemented; T6 written, live E2E pending".

**T6 details (partial — code written, live run blocked):**

- New `src/scenarios/distributed_demo.yaml`: exact physics copy of `pv_batt_test_base.yaml` with a
  `deployment:` block (`manager_address: 127.0.0.1`, machine alias `local_box` → `127.0.0.1`,
  `workdir: /tmp/cosimgym_remote_test`, `conda_env: cosim_gym`, `python:` shown commented as the
  escape hatch) and `pv_federate` tagged `host: local_box`. Parses + validates cleanly through
  `read_scenario_config` (deployment present, manager_address set, alias valid).
- New `src/verify_distributed_demo.py`: runs the remote demo + its all-local twin
  (`pv_batt_test_base`), then compares every recorded federate timeseries (reuses
  `dashboard_data.load_all_records`) within `1e-9`. `--no-run` compares latest existing runs only.
  Import/entry verified (`--no-run` runs, correctly reports the demo hasn't executed yet).
- **Live E2E NOT executed — two environment blockers on this machine:**
  1. **Passwordless key-based ssh to 127.0.0.1 is not set up.** Both `id_ed25519` and `id_rsa`
     exist but neither pubkey is in `~/.ssh/authorized_keys`. Enabling it (append own pubkey) was
     **denied by the Claude Code auto-mode classifier as unauthorized persistence** — the USER must
     do this manually: `ssh-copy-id -i ~/.ssh/id_ed25519 127.0.0.1` (or append the pubkey), then
     `ssh 127.0.0.1 true` once to accept the host key.
  2. **cosim Redis+Mosquitto stack is not up** — only an unrelated `project1-boptest_redis_1`
     occupies host port 6379, which will **conflict** with `src/docker-compose.yaml`'s Redis
     (also binds 6379). That container must be stopped (or the compose port remapped) before
     `docker compose -f src/docker-compose.yaml up -d`.
  - Note: `conda run -n cosim_gym python -c "import helics,redis"` DOES work in a non-interactive
    bash on this machine, so the demo's `conda_env: cosim_gym` default should resolve over ssh; if
    not, uncomment the `python:` line in the demo YAML.
- Once both blockers cleared: `python src/verify_distributed_demo.py` is the one-command E2E +
  acceptance check. Also do the Ctrl+C-mid-run orphan check: start the demo, Ctrl+C, then
  `pgrep -f <sim_id>` on the remote must be empty (validates `_cleanup_remote_execution`'s pkill).

---

### Original T1–T4 progress (retained for reference)

**T1, T2, T3: DONE. T4 — Spawn dispatch + address normalization: DONE.**

**T4 details:**

- `src/core/ScenarioManager.py`:
  - `_setup_local_federation`'s federate loop now calls new `_create_federate(...)` instead of
    `_create_local_federate(...)` directly. `_create_federate` dispatches on
    `getattr(federate_config, 'host', None)`: set → `_create_remote_federate`, else →
    `_create_local_federate` (unchanged local behavior). Both append to
    `self.federate_processes`, so `_monitor_processes`/cleanup need zero branching.
  - New `_build_federate_args(federate_name, federate_config, federation_name, redis_url,
    log_file)`: the CLI arg list (minus interpreter/script tokens) shared by both spawn paths —
    extracted verbatim from the old `_create_local_federate` body.
  - New `_redis_url_for(federate_config)`: returns `redis://<manager_address>:<port>/<db>` for a
    remote federate, else `self.redis_url` unchanged. Uses `self._redis_port`/`self._redis_db`,
    two new attrs stashed in `_upload_config_on_redis` (alongside the existing `self.redis_url`)
    specifically so this method never has to re-parse a URL string.
  - `_create_local_federate`: same behavior as before, just now calls `_build_federate_args` +
    prepends `['python', federate_launcher]` — no functional change, verified via a real
    end-to-end run (see below).
  - New `_create_remote_federate(federate_name, federate_config, federation_name, host_alias)`:
    looks up `self.remote_executors[host_alias]` (populated by T3's `_setup_remote_execution`),
    builds a remote log path by joining `machine_conf.workdir` + the *relative*
    `str(self.logger_system.scenario_log_dir)` (this Path is already relative to the project
    root — `FederationLogger.__init__` does `Path("logs") / scenario_name / run_timestamp` — so
    joining it onto a remote `workdir` reproduces the exact same relative layout there; T5's
    rsync-back needs no path translation). Calls `executor.spawn(launcher_args, remote_stdio_file)`
    where `launcher_args = ['src/core/federate_launcher.py'] + _build_federate_args(...)`.
  - `_setup_remote_execution` (T3 method, extended here): after `executor.deploy(project_root)`,
    also `mkdir -p <workdir>/<scenario_log_dir_rel>/federates` via `executor.run(...)` — needed
    because `setup_process_logger`'s `logging.FileHandler` does NOT create parent directories
    (confirmed by reading `utils/logging_config.py`); locally this dir is pre-created by
    `FederationLogger.__init__`, so the remote path needs the equivalent explicit mkdir before
    any federate on that machine spawns.
  - `_normalize_broker_and_core_configs`: added `default_broker_host` (computed once, near the
    top of step 3) = `deployment.manager_address` when `self._has_remote_federates()` and a
    `deployment` block exists, else `'127.0.0.1'` (unchanged default). Used in place of the two
    previously-hardcoded `'127.0.0.1'` spots: (1) each federation broker's `broker_conf.host`
    default, (2) the hierarchy/main broker's `host`/`address` and the
    `main_broker_address` string built for multi-federation scenarios. An explicit YAML
    `broker_config.host` still overrides in both cases. Added a docstring note that HELICS' tcp
    core binds all interfaces by default, so advertising the LAN address needs no
    `--local_interface` flag change — flagged in the plan as something to verify, not proven
    with a real 2-machine run yet (that's T6's job).
  - **Note**: `self.config.multi_computer`/`multi_computer_config` (a pre-existing, separate,
    unimplemented stub predating this feature) was left completely untouched — its `if` branch
    still just logs an error and does nothing; only the `else` branch (the real path, used by
    both local scenarios and the new `deployment:` mechanism) was touched.
- `src/core/federate_launcher.py`: right after parsing `redis_host`/`redis_port` out of
  `--redis-url` (existing code), added
  `os.environ.setdefault('REDIS_HOST', redis_host)` / `.setdefault('REDIS_PORT', ...)` /
  `.setdefault('MQTT_HOST', redis_host)`. **Why this was needed** (audit findings, per plan's T4
  checklist):
  - `core/override_registry.py`'s `OverrideRegistry.__init__` builds its own `RedisClient` from
    `os.getenv('REDIS_HOST', 'localhost')`/`REDIS_PORT` — NOT from any URL passed in. Without
    this fix, a remote federate with `override_enabled: true` would silently try to reach Redis
    on ITS OWN localhost instead of the manager, and fail (or worse, silently connect to an
    unrelated local Redis if one happened to be running there).
  - `BaseFederate._stream_outbound` (the opt-in `streaming: stream: true` MQTT mirror) builds its
    `MqttAdapter` from `os.getenv('MQTT_HOST', 'localhost')` — same problem, same fix.
  - Used `setdefault` (not unconditional set) so an operator's own `REDIS_HOST`/`MQTT_HOST`
    env var override, if they ever set one, still wins.
  - Since Redis and Mosquitto always run together on the manager machine (docker-compose), and
    `--redis-url` was already being correctly resolved per-federate by `_redis_url_for`
    (localhost locally, `manager_address` remotely), this fix needed zero new CLI args — just
    reuse the value already being parsed.
  - **NOT changed**: the `InterfaceFederate`'s adapter-based MQTT path (`interface_config.adapter.params.host`,
    catalog default `localhost`, catalog entry at `src/models/model_catalog/catalog.yaml:2801`).
    That path is already fully YAML-configurable per scenario (unlike the streaming path above,
    which had no such escape hatch) — per the plan's own conditional ("if adapter address comes
    from scenario config already, just document"), this needs a T7 docs note telling users to set
    `adapter.params.host: <manager_address>` for a remote interface federate, not a code change.
- **Real end-to-end verification** (not just pytest): ran two existing local scenarios through
  `ScenarioManager.start_scenario()` against the live `cosim_redis`/`cosim_mosquitto` docker
  containers: `pv_batt_test_base` (plain local dispatch path) and
  `m4_interface_override_smoke_test` (exercises `OverrideRegistry`, proving the
  `REDIS_HOST`/`MQTT_HOST` env-var `setdefault` doesn't break the local case). Both completed
  successfully.
- Test-suite adjustment: `tests/test_scenario_manager_remote.py`'s
  `test_verify_and_deploy_called_for_each_machine` now sets
  `fake_executor.run.return_value = (0, '', '')` and asserts `fake_executor.run.assert_called_once()`,
  since `_setup_remote_execution` now calls `executor.run(...)` once (the federates-dir mkdir)
  after `deploy()`.
- Full suite after T4: `pytest tests/ -q` → **140 passed, 2 skipped** (same 2 pre-existing skips —
  test count unchanged from T3 since T4 added no new test file, only extended existing ones).
- `graphify update .` run after the change.

**T3 details:**

- `src/core/ScenarioManager.py`:
  - New import: `from core.remote_executor import RemoteExecutor`.
  - `__init__`: added `self.remote_executors: Dict[str, RemoteExecutor] = {}` next to the
    existing `broker_processes`/`federate_processes` lists — empty dict is the "fully local"
    state, checked by later remote-dispatch code (T4) with a plain truthiness/`in` check.
  - New `_has_remote_federates()`: walks all federations' federate_configs, `True` if any has
    `host` set.
  - New `_setup_remote_execution()`: no-ops immediately if `_has_remote_federates()` is False
    (zero `RemoteExecutor` construction, zero ssh — verified by a test that makes
    `RemoteExecutor(...)` raise if called on a local-only scenario). Otherwise: for each unique
    `host` alias in use, builds a `RemoteExecutor` (control_dir =
    `<scenario_log_dir>/ssh_control`, project_root = repo root via
    `Path(__file__).resolve().parents[2]`, same pattern `_setup_results_folder` already uses),
    calls `open_master()`, stores in `self.remote_executors`. Then calls `verify()` + `deploy()`
    on every executor. On ANY exception in that second loop: closes every already-opened
    executor, resets `self.remote_executors = {}`, re-raises — so a mid-way preflight failure on
    machine 2 doesn't leak machine 1's control socket, and the scenario aborts before any broker
    starts.
  - Wired into `_setup_classic_scenario()`: `self._setup_remote_execution()` called right after
    `_setup_results_folder()`, before `_normalize_broker_and_core_configs()` — i.e. before any
    broker or federate process is spawned, satisfying the plan's "fail whole scenario on any
    preflight failure BEFORE any broker starts" requirement.
- New `tests/test_scenario_manager_remote.py` — 5 tests. Technique: builds a `ScenarioManager`
  via `object.__new__(ScenarioManager)` + hand-set attrs (`config`, `logger`, `logger_system`,
  `remote_executors`) instead of the real `__init__` (which needs a live Redis connection and
  full logging setup) — isolates `_has_remote_federates`/`_setup_remote_execution` logic cleanly.
  Monkeypatches `RemoteExecutor` via `sys.modules['core.ScenarioManager']` (NOT via the dotted
  string `'core.ScenarioManager.RemoteExecutor'` — `core/__init__.py` does
  `from .ScenarioManager import ScenarioManager`, which rebinds the `core.ScenarioManager`
  *package attribute* to the class, shadowing the submodule; `monkeypatch.setattr` with a string
  path resolves through that attribute chain and hits the class instead of the module. Getting
  the module out of `sys.modules` directly sidesteps it. Worth remembering for any future test
  that needs to monkeypatch something inside `ScenarioManager.py`).
  Covers: no-op for local-only scenario, executor built+verified+deployed for a remote-federate
  scenario, and preflight failure aborts + closes already-opened masters.
- **Real end-to-end verification, not just tests**: ran `pv_batt_test_base` (existing local
  scenario, no `deployment:` block) through `ScenarioManager.start_scenario()` directly against
  the running `cosim_redis` docker container — completed successfully, confirming the new
  no-op call in `_setup_classic_scenario` doesn't disrupt the real local path (not just under
  pytest mocks).
- Full suite after T3: `pytest tests/ -q` → **140 passed, 2 skipped** (same 2 pre-existing skips).
- `graphify update .` run after the change.

**T2 details:**

**T2 details:**

- New `src/core/remote_executor.py`, class `RemoteExecutor(machine_alias, machine_conf,
  manager_address, logger, control_dir)`. No `ScenarioManager` import (standalone/testable).
- Design choice: split every ssh/rsync invocation into a pure `_*_cmd(...)`/`_build_remote_command(...)`
  builder (returns argv list or command string, no subprocess call) and a thin execution method
  that calls `subprocess.run`/`Popen` with that builder's output. This is what let T2's tests
  assert on exact argv/quoting without ever touching a real ssh connection.
- Methods implemented: `open_master()`, `run(cmd, timeout)`, `verify()` (4 preflight checks:
  control-conn alive, workdir writable, `import helics, redis, pydantic` in the target env,
  Redis reachable at `manager_address:6379`), `deploy(project_root)` (rsync `src/` → `<workdir>/src`,
  then `mkdir -p <workdir>/{logs,results}`), `spawn(args_list, remote_log_file)` (returns the ssh
  child `Popen`; `-tt` pty so SIGHUP on kill propagates to the remote process; remote stdout/stderr
  appended into `remote_log_file`), `collect(remote_path, local_path)` (rsync back), `close()`
  (never raises — closes ControlMaster via `ssh -O exit`).
- Control socket path: `<control_dir>/cm-<alias>` — literal per-alias path, not ssh's `%r@%h:%p`
  token syntax (plan shows the token form but since one `RemoteExecutor` = one alias and the
  caller already scopes `control_dir` per run, the alias alone is sufficient to guarantee
  uniqueness — simpler, no behavior difference).
- Quoting: every remote command token goes through `shlex.quote` individually before joining;
  never `shell=True` on the local side. Verified with hostile inputs (`;`, `$(...)`, `$VAR`,
  embedded spaces) in tests.
- `tests/test_remote_executor.py` — 20 tests (19 run, 1 skipped by default): target/control-path
  construction, python-cmd resolution (conda vs explicit `python:`), master/run/rsync/spawn/close
  argv shape, quoting-hostile-input cases, and a `spawn()` test that monkeypatches
  `core.remote_executor.subprocess.Popen` to assert the built argv without launching real ssh.
  One `TestLiveLoopback` class gated behind `COSIMGYM_TEST_SSH_LOCALHOST=1` env var (not run by
  default) for a real ssh-to-127.0.0.1 round trip.
- **Live ssh not verified in this sandbox** — tried `ssh -o BatchMode=yes 127.0.0.1 true`,
  got `Host key verification failed` (no passwordless/known-hosts ssh set up in this dev
  environment). This is an environment limitation, not a code issue. Whoever picks up T6 (the
  real localhost-as-remote E2E test) needs an environment with working key-based ssh to
  127.0.0.1 first — `ssh-keygen`, `ssh-copy-id`/append to `authorized_keys`, accept host key once
  interactively, THEN run the gated live test / T6 demo scenario.
- Full suite after T2: `pytest tests/ -q` → **135 passed, 2 skipped** (both skips expected/pre-existing).
- `graphify update .` run after the change.

**T1 details:**

- `src/utils/config_dataclasses.py`:
  - Added `MachineConfig` (host, user, ssh_port=22, workdir, conda_env='cosim_gym', python=None)
    and `DeploymentConfig` (manager_address: Optional[str], machines: Dict[str, MachineConfig]),
    placed in new section before `TOP-LEVEL SCENARIO CONFIG`.
  - Added `host: Optional[str] = None` to `_FederateConfigBase` (inherited by base/rl/interface).
  - Added `deployment: Optional[DeploymentConfig] = None` to `ScenarioConfig`.
  - Added `ScenarioConfig._validate_deployment` (`model_validator(mode='after')`): walks all
    federates across all federations, no-ops if none set `host:`. If any do: requires
    `deployment` block present, requires `deployment.manager_address` set, requires each
    `host:` alias exist in `deployment.machines`, rejects `host:` on `RLFederateConfig` (v1
    restriction per plan — RL agent must run on manager).
  - Left `MultiComputerConfig`/`multi_computer` field alone — it's a pre-existing unused stub,
    unrelated to this feature, not worth touching.
- New `tests/test_deployment_config.py` — 9 tests: no-deployment-unchanged, MachineConfig
  defaults, full valid deployment scenario, host-without-deployment rejected, unknown-alias
  rejected, host-without-manager_address rejected, host-on-rl rejected, deployment-with-no-host
  is fine.
- Ran `pytest tests/test_deployment_config.py tests/test_rl_config.py -q` (conda env
  `cosim_gym`): **96 passed, 1 skipped** (pre-existing unrelated skip, `Adelaide_test.yaml`
  known-broken YAML). Zero regression on existing scenario parsing.
- Ran `graphify update .` after the change (per CLAUDE.md rule) — graph current.

## What Worked

- Used `graphify query`/full-file `Read` on `config_dataclasses.py` + `config_reader.py` to
  orient before editing (project's graphify hook mandates this before raw file reads).
- Matched existing pydantic v2 style exactly: `model_config = ConfigDict(extra='ignore')` for
  non-RL blocks (this is a non-RL block, so `ignore` not `forbid` — matches `BrokerConfig`/
  `FederationConfig` sibling style, not the RL axis's `forbid` style).
  Cross-field validation lives in a `model_validator(mode='after')` on `ScenarioConfig` (same
  pattern as `FederationConfig._validate`), because alias-existence check needs the full
  federations dict.
- Test fixtures built as plain dicts through `ScenarioConfig.model_validate(...)`, following
  the existing `test_rl_config.py` convention — no new fixture infra needed.

## What Didn't Work

Nothing hit a dead end in T1. One judgment call worth flagging: `DeploymentConfig.manager_address`
is `Optional[str]` at the field level (not required), with the "required if any host: set" rule
enforced in `ScenarioConfig._validate_deployment` instead. This was deliberate — makes the error
message scenario-aware ("required when any federate sets host:") rather than a generic pydantic
missing-field error, and allows a `deployment:` block to exist with only `machines:` defined and
no host-using federates yet (edge case, but no reason to forbid it).

## Next Steps

**Only the live T6 E2E run remains.** All code (T1–T5, T7) is implemented and unit-tested (145
passed, 2 skipped). To finish:

1. USER: enable passwordless ssh to 127.0.0.1 (`ssh-copy-id -i ~/.ssh/id_ed25519 127.0.0.1`;
   `ssh 127.0.0.1 true` once). Agent cannot — classifier blocks self-authorizing ssh keys.
2. Stop `project1-boptest_redis_1` (frees port 6379), then
   `docker compose -f src/docker-compose.yaml up -d` (cosim Redis + Mosquitto).
3. `conda activate cosim_gym && python src/verify_distributed_demo.py` → expect `PASS`.
4. Orphan check: run demo, Ctrl+C mid-run, confirm `pgrep -f <sim_id>` empty afterward.
5. If green: update this file's status to T6 DONE, then ask user before committing.

---

### Original T4–T7 next-steps (retained for reference)

- **T4 — Spawn dispatch + address normalization.** Split `_create_local_federate`
  (`src/core/ScenarioManager.py:1486`) into `_build_federate_args(...)` (shared, parametrized by
  redis_url + log_file) and dispatch on `federate_config.host` (executor.spawn vs local Popen),
  both append to `self.federate_processes`. Fix `_normalize_broker_and_core_configs` (`:1162`) to
  default broker host to `manager_address` when remote federates exist. Audit + thread
  `manager_address` through hardcoded-`localhost` spots: redis url arg builder, `OverrideRegistry`
  (`src/core/override_registry.py`) client construction, `src/adapters/mqtt_adapter.py` broker
  address.
- **T5 — Collection + cleanup.** rsync results/logs back after `_monitor_processes` returns,
  before `_log_execution_summary`; close ControlMaster sockets. `_emergency_cleanup` (`:152`): add
  per-machine `pkill -f <sim_id>` + master close, wrapped in try/except (cleanup must never raise).
- **T6 — Demo scenario + E2E test.** `src/scenarios/distributed_demo.yaml` derived from a simple
  existing 2-federate scenario (e.g. `pv_batt_test_base.yaml`), `deployment` block pointing at
  `127.0.0.1` (real ssh, real rsync, localhost-as-remote = CI-able pattern), one federate `host:`ed.
  Verify output matches an all-local twin run.
- **T7 — Docs.** `docs/user_guide/distributed_deployment.md` (one-time remote setup, YAML
  reference, troubleshooting table, LAN-trust security note), add `deployment`/`host` to
  `CLAUDE.md` Config Reference (terse), update plan doc status header.

Constraints that apply to every remaining task (from plan doc, do not drop):
- Never commit without asking.
- `deployment:` absent → provably identical behavior to main branch.
- All ssh/rsync knowledge stays inside `RemoteExecutor`; `ScenarioManager` only decides where to spawn.
- No hardcoded hosts/paths/users/ports anywhere — everything from `DeploymentConfig`.
- Run `graphify query` before reading source files (project hook enforces this), `graphify update .`
  after code changes.
- Run `pytest tests/` + demo scenario before declaring any task done.
- Update this handoff file after each task.
