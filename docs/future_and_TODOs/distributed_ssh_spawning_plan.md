# Distributed SSH Federate Spawning — Implementation Plan

> Handoff doc for implementer agent (Sonnet). Caveman prose, full technical content. Code blocks normal.
> Rationale + alternatives for every choice: see `distributed_ssh_spawning_alternatives.md` (same folder).
> Status: **COMPLETE — T1–T7 implemented, live localhost-as-remote E2E PASSED (2026-07-15).**
> `distributed_demo` (pv_federate spawned remotely over ssh) reproduced the all-local twin exactly
> (2304/2304 records within 1e-9); results collected back; mid-run SIGINT left zero orphans. Two bugs
> found+fixed during E2E (ssh master `-nNf` pipe-deadlock in `remote_executor`; unbound `success` on
> SIGINT-during-setup in `ScenarioManager`) — see HANDOFF.md. Full pytest suite green (156 passed, 2
> skipped). NEVER commit — ask user.

## Objective

Today `ScenarioManager` spawn all brokers + federates as local subprocesses on one machine. Goal: allow federates run on remote machines over SSH. Declarative: user adds `deployment:` block + per-federate `host:` key in scenario YAML. Everything else unchanged. Priorities: speed, simplicity, no overengineering, no disruption of local path (no `host:` = behave exactly as today).

## Decisions Taken (user-approved, do not relitigate)

1. **SSH mechanism**: plain OpenSSH client via `subprocess.Popen` + ControlMaster multiplexing. NOT paramiko/fabric/asyncssh. Reason: `ssh` child process == remote process handle → existing `_monitor_processes` poll loop and cleanup work almost unchanged; exit code propagates; one TCP+auth handshake per machine (master), each extra spawn ~ms. Zero new pip deps.
2. **Deploy**: pre-provisioned remote env (conda env `cosim_gym` exists, one-time manual setup, documented) + automatic `rsync` of code each run. NOT docker.
3. **Results/logs**: remote federates write to local disk as today; manager `rsync`s back after run ends. Document MinIO/S3 live-sink as future enhancement (see bottom), do NOT implement now.
4. **YAML shape**: scenario-level `deployment:` block defines machines; each federate optional `host: <machine-alias>`. Per-federate granularity.

## Current Architecture Facts (verified, with anchors)

- `ScenarioManager._setup_local_federation()` (`src/core/ScenarioManager.py:1124`) → starts federation broker (`_start_local_federation_broker`, `:1385`) then loops `_create_local_federate` (`:1486`).
- Hierarchy broker for multi-federation: `_start_local_hierarchy_broker` (`:1330`). All brokers = `helics_broker` CLI via Popen, `preexec_fn=os.setsid`.
- Federate spawn cmd (`:1486`): `python src/core/federate_launcher.py --name .. --scenario_name .. --federation_name .. --type .. --simid .. --redis-url .. --redis-key .. --log-file .. --log-level ..`. Stdout/stderr → local stdio log file.
- **Config distribution ALREADY remote-friendly**: full scenario config serialized to Redis by manager; launcher fetches by `--redis-url`/`--redis-key` (`src/core/federate_launcher.py:91-134`). Remote federate only needs reachable Redis URL.
- `federate_launcher.py:84` reads `src/core/mappings.yaml` **relative to CWD** → remote invocation MUST `cd <workdir>` first (workdir = synced repo root).
- Broker/port normalization: `_normalize_broker_and_core_configs` (`:1162`). Defaults `broker_conf.host = '127.0.0.1'`; auto-assign ports 20000-30000 via **local socket bind check** (`_get_n_available_tcp_ports`, `:1143`). Local check stays valid because (design decision below) brokers stay on manager machine.
- Monitor: `_monitor_processes` (`:1553`) polls `Popen.poll()` on `self.federate_processes` + `self.broker_processes`. Cleanup: `_emergency_cleanup` (`:152`) kills process groups.
- Redis in docker-compose binds host `0.0.0.0:6379` (`src/docker-compose.yaml`) → already reachable from LAN, only URL needs manager LAN IP instead of localhost.
- HELICS sync/time management lives inside broker protocol over TCP — works cross-machine natively, nothing to change in `BaseFederate` stepping logic.

## Core Design (v1 scope)

### Placement rule: brokers stay on manager machine

Only **federates** go remote. Hierarchy broker + all federation brokers spawn locally on manager as today. Consequences:

- Local port bind check stays correct.
- Remote federates connect to broker at `manager_address:port` (HELICS TCP core, native).
- Data plane has no extra hop: federate cores talk broker directly, same as today just over LAN.
- Massive simplification vs remote brokers (no remote port discovery, no broker log streaming over ssh). Remote broker placement = future work.

### Declarative schema

```yaml
# scenario YAML, top level (all optional — absent = fully local, zero behavior change)
deployment:
  manager_address: 192.168.1.10        # REQUIRED if any federate has host:. LAN IP that remote machines reach (used for broker addresses + redis url + mqtt)
  machines:
    gpu_box:                           # alias referenced by federates
      host: 192.168.1.42               # ssh target
      user: rando                      # optional, default current user
      ssh_port: 22                     # optional, default 22
      workdir: /home/rando/cosimgym_rt # remote repo root, code synced here
      conda_env: cosim_gym             # optional, default "cosim_gym"
      python: null                     # optional explicit interpreter path; overrides conda_env if set

federations:
  power_grid:
    federate_configs:
      building_fed:
        host: gpu_box                  # NEW optional key on any base/interface federate
        type: base
        ...
```

Rules:
- `host:` absent → local spawn, code path identical to today.
- `host:` references unknown alias → validation error at parse time.
- `host:` on `type: rl` federate → raise `NotImplementedError` in v1 (RL agent trains on manager; consistent with existing `parallel_execution`+rl restriction style). `parallel_execution: true` + remote host = ALLOWED (workers spawn on remote machine, use its CPUs — that is the whole point).
- No hardcoded IPs/paths/users anywhere in code. Everything from `deployment` block.

### Spawn mechanics

Per machine, once at scenario setup:

```bash
ssh -o ControlMaster=auto -o ControlPath=<scratch>/cm-%r@%h:%p -o ControlPersist=60 \
    -o BatchMode=yes -o ConnectTimeout=10 -p <ssh_port> -nNf user@host
```

Per remote federate:

```bash
ssh -S <controlpath> -tt -p <ssh_port> user@host \
  'cd <workdir> && conda run --no-capture-output -n <conda_env> python src/core/federate_launcher.py \
     --name ... --scenario_name ... --federation_name ... --type ... --simid ... \
     --redis-url redis://<manager_address>:6379/0 --redis-key ... \
     --log-file <workdir>/logs/<scenario>/<simid>/federates/federate_<name>.log --log-level ...'
```

Key points:
- `-tt` forces pty → when manager kills ssh child, remote gets SIGHUP → remote federate dies. Cleanup for free, no orphans.
- `BatchMode=yes` → fail fast if no key auth, never hang on password prompt.
- Returned `Popen` object appended to `self.federate_processes` → `_monitor_processes` untouched.
- If `machine.python` set, use it directly instead of `conda run` (escape hatch for venv users).
- Remote log paths under `<workdir>/logs/...` mirroring local layout → rsync-back merges cleanly.
- Quote/escape remote command properly (`shlex.quote` each arg, join). No string interpolation of unvalidated YAML into shell without quoting.

### Address normalization changes

In `_normalize_broker_and_core_configs`:
- Detect `scenario_has_remote = any(fed.host for ...)`.
- If remote: broker `host` default becomes `deployment.manager_address` instead of `'127.0.0.1'` (user-explicit `broker_config.host` still wins). Broker must LISTEN on reachable interface — check `helics_broker` binding: default binds all interfaces for tcp core, but VERIFY during implementation (`--local_interface` flag exists if needed).
- Redis URL: manager keeps using localhost for itself; the `--redis-url` arg passed to REMOTE federates uses `manager_address`. Local federates keep current URL. Small helper: `_redis_url_for(federate_config)`.
- MQTT (streaming/interface federates): audit `src/adapters/mqtt_adapter.py` + catalog params for hardcoded `localhost` broker address; remote streaming federate must receive `manager_address:11883`. If adapter address comes from scenario config already, just document; if hardcoded, thread `manager_address` through same way as redis. (Audit task T4.)
- `OverrideRegistry` (`src/core/override_registry.py`) is Redis-backed → follows redis url fix automatically IF it builds its client from the same passed url. Audit: if it constructs `RedisClient()` with defaults (`localhost`), must receive url. (Task T4.)

### Deploy step (before spawn)

For each machine used by ≥1 federate:

```bash
rsync -az --delete \
  --exclude '__pycache__' --exclude '.git' --exclude 'results' --exclude 'logs' \
  --exclude 'graphify-out' --exclude '*.pyc' \
  -e 'ssh -S <controlpath> -p <ssh_port>' \
  ./src user@host:<workdir>/
```

- Sync `src/` only (launcher, core, utils, models, adapters, scenarios). Delta transfer → after first run, near-instant.
- Uses master connection → no extra auth.
- `--delete` scoped inside synced `src/` only, never touches rest of workdir.
- Create remote log dir: `ssh ... 'mkdir -p <workdir>/logs <workdir>/results'`.

### Preflight verification (fail fast, clear errors)

Before deploy, per machine:
1. Master connection opens (ssh reachable + key auth).
2. `test -d <workdir> || mkdir -p <workdir>` writable.
3. `conda run -n <env> python -c "import helics, redis, pydantic"` → env sane.
4. `python -c "import socket; socket.create_connection(('<manager_address>', 6379), 5)"` → Redis reachable from remote.

Any failure → RuntimeError with machine alias + which check + remediation hint. Do NOT silently fall back to local.

### Results + logs collection (after run)

After `_monitor_processes` returns (and before `_log_execution_summary`), per used machine:

```bash
rsync -az -e 'ssh -S <controlpath>' \
  user@host:<workdir>/results/<scenario_name>/<sim_id>/ results/<scenario_name>/<sim_id>/
rsync -az -e 'ssh -S <controlpath>' \
  user@host:<workdir>/logs/<scenario_log_dir_name>/ <local scenario_log_dir>/
```

- Works for `sink: json` (written at end) and `sink: parquet` (finalized before process exit) identically — collection happens after processes exit.
- Merge is safe: each federate writes distinct files (`<federate>_<mode>_storage.*`), no collisions.
- Collection failure → log ERROR with manual rsync command, do not crash run summary.

### Cleanup

- Normal end: federate exits → remote conda/python exits → ssh child exits → poll() sees it. Then close masters: `ssh -S <controlpath> -O exit user@host`.
- `_emergency_cleanup`: existing group-kill kills ssh children → `-tt` pty → SIGHUP kills remotes. Add belt-and-suspenders remote sweep: `ssh ... 'pkill -f <sim_id>'` per machine (sim_id unique per run → safe pattern), then close masters. ControlPath sockets in session scratch dir → stale sockets no problem.

## Micro Tasks (implement in order)

### T1 — Config schema
Files: `src/utils/config_dataclasses.py`, `src/utils/config_reader.py`, `tests/`
- Add `MachineConfig` (host, user, ssh_port=22, workdir, conda_env='cosim_gym', python=None) + `DeploymentConfig` (manager_address, machines: Dict[str, MachineConfig]). Follow existing dataclass/pydantic style of file (`extra='forbid'` where module uses pydantic; match whichever pattern `ScenarioConfig` uses).
- Add optional `host: Optional[str]` to `_FederateConfigBase`.
- Validation: every referenced alias exists; `deployment.manager_address` required when any `host:` set; `host:` on rl federate → error.
- Tests: parse good/bad YAML fixtures. Accept: `pytest` green, existing scenarios (no deployment block) parse unchanged.

### T2 — RemoteExecutor
Files: new `src/core/remote_executor.py`, tests
- Class `RemoteExecutor(machine_alias, machine_conf, manager_address, logger, control_dir)`.
- Methods: `open_master()`, `verify(preflight list above)`, `deploy(project_root)` (rsync), `spawn(args_list, remote_log_file) -> Popen`, `run(cmd, timeout) -> (rc, out, err)` (for preflight/mkdir/pkill), `collect(remote_path, local_path)`, `close()`.
- All ssh/rsync invocations via `subprocess`, args as lists, remote command built with `shlex.quote`. No shell=True on local side.
- Module must be importable/testable standalone (no ScenarioManager import).
- Tests: unit-test command construction (assert argv lists, no live ssh needed) + optional integration test against `127.0.0.1` guarded by env var / `pytest.mark.skipif` (ssh-to-localhost pattern, workdir in /tmp). Accept: command strings correct, quoting proven with hostile inputs (space, `;`, `$`).

### T3 — ScenarioManager integration: preflight + deploy
Files: `src/core/ScenarioManager.py`
- In `start_scenario` setup phase: if config has remote federates → build `{alias: RemoteExecutor}` map, open masters, run `verify()` all machines, `deploy()` all machines. Fail whole scenario on any preflight failure BEFORE any broker starts.
- Keep fully skipped when no `deployment` used — zero new work on local path.
- Accept: local-only scenario runs byte-identical to main branch behavior.

### T4 — Spawn dispatch + address normalization
Files: `src/core/ScenarioManager.py`
- `_create_local_federate` → split: `_build_federate_args(...)` (shared arg list builder, parametrized redis_url + log_file path) + dispatch `_create_federate`: `federate_config.host` → `executor.spawn(...)`, else existing local Popen. Both append to `self.federate_processes`.
- `_normalize_broker_and_core_configs`: broker default host = `manager_address` when remote federates exist. Verify helics_broker listens non-loopback (test with 2-machine or localhost-alias run).
- Audit + fix hardcoded `localhost` for remote consumers: redis url arg, `OverrideRegistry` client construction, `mqtt_adapter` broker address. Thread from config, never hardcode.
- Accept: mixed scenario (1 local + 1 "remote" via 127.0.0.1 machine alias) completes; HELICS time sync converges; results identical to all-local run of same scenario.

### T5 — Collection + cleanup
Files: `src/core/ScenarioManager.py`
- After `_monitor_processes`: `collect()` results + logs per machine; then `close()` masters.
- `_emergency_cleanup`: add per-machine `pkill -f <sim_id>` + master close, wrapped in try/except (cleanup must never raise).
- Accept: after run, `results/<scenario>/<simid>/` on manager contains remote federate files; Ctrl+C mid-run leaves no `federate_launcher` process on remote (check with `pgrep -f simid`).

### T6 — Demo scenario + E2E test
Files: `src/scenarios/distributed_demo.yaml`, test script or pytest marker
- Copy simple existing 2-federate scenario (e.g. derive from `pv_batt_test_base.yaml`), add `deployment` block with machine = `127.0.0.1`, workdir `/tmp/cosimgym_remote_test`, one federate `host:`ed.
- This is the CI-able localhost-as-remote pattern: real ssh, real rsync, one machine.
- Accept: `python src/test_script.py`-style run of demo scenario green; output values match all-local twin scenario.

### T7 — Docs
Files: `docs/user_guide/distributed_deployment.md`, update `CLAUDE.md` Config Reference, update this file status
- User guide: one-time remote setup (create conda env, install helics deps, ssh keypair, open ports 6379 + broker range 20000-30000 + 11883 on manager firewall), YAML reference, troubleshooting table (BatchMode auth fail, redis unreachable, conda env missing), security note (LAN-trusted assumption: Redis/MQTT unauthenticated — do not expose on untrusted networks).
- CLAUDE.md: add `deployment` + `host` keys to Config Reference section (terse).

## Performance Rationale (why this is fast)

- ControlMaster: 1 handshake per machine per run; each spawn/rsync reuses socket (~few ms overhead vs local Popen).
- rsync delta: first deploy copies `src/` (~small), subsequent runs near-zero.
- Data plane unchanged: HELICS federate↔broker TCP direct, no proxy/tunnel. Per-tick cost = LAN RTT inside broker sync protocol — inherent to distribution, not to this design.
- Spawns are non-blocking Popen exactly like today → all federates across all machines start in parallel.
- No new Python deps, no serialization changes, no async rewrite.

## Constraints For Implementer

- Work on fresh branch. NEVER commit — ask user.
- Do not touch local execution path semantics. `deployment:` absent → identical behavior, provable by running existing scenarios.
- Modular: all ssh/rsync knowledge inside `RemoteExecutor`. `ScenarioManager` only knows "spawn here or there".
- No hardcoded hosts/paths/users/ports. All from `DeploymentConfig`.
- Match existing code style (logging patterns, error messages, docstrings w/ Args/Returns).
- Use graphify to navigate (`graphify query "..."`) before reading raw files; run `graphify update .` after code changes.
- Run `pytest tests/` + demo scenario before declaring any task done.
- genrate or update handoff file after each task so that session ends or compact requires you have the right instructions for next session.

## Future Enhancements (document, do NOT build now)

1. **MinIO/S3 live results sink**: federates upload parquet batches to MinIO (already in `src/docker-compose.yaml`, creds there) during run → no rsync-back, dashboard reads central store, survives remote-disk loss. Natural extension of `AsyncStorageWriter`: add `sink: minio` writing via `pyarrow.fs.S3FileSystem` or `minio` client; schema already long/tidy. Entry point: `src/utils/async_storage.py` + `memory_config.sink`.
2. **Remote brokers**: place federation brokers next to their federates (cuts cross-machine per-tick traffic for co-located federations). Needs remote port allocation (run port-picker via `RemoteExecutor.run`) + broker log streaming over ssh.
3. **asyncssh backend**: if machine count grows large (>10), swap RemoteExecutor internals for asyncssh persistent connections; interface stays.
4. **Docker provisioning**: replace conda-env assumption with image pull for reproducibility.
5. **RL federate remote**: lift v1 `NotImplementedError` — needs GPU-aware placement + checkpoint path collection.
