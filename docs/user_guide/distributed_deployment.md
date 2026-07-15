# Distributed Deployment — Spawning Federates on Remote Machines over SSH

By default `ScenarioManager` runs every broker and federate as a local subprocess on one machine.
This page covers the opt-in `deployment:` mechanism that lets individual **federates** run on
**remote machines** over SSH, declared entirely in scenario YAML. Everything else — HELICS time
sync, Redis config distribution, results layout — is unchanged.

**Off by default.** A scenario with no `deployment:` block and no `host:` keys behaves
byte-for-byte as it always has. Nothing on the local path changes.

## What goes where

Only federates move. Brokers stay on the manager machine (the machine you launch the run from):

- **Manager machine**: hierarchy broker + every federation broker, Redis, Mosquitto, and any
  federate *without* a `host:` key.
- **Remote machine(s)**: any federate tagged `host: <alias>`. It connects back to the broker and
  Redis on the manager over the LAN — HELICS' TCP/ZMQ cores work cross-machine natively, so the
  co-simulation logic is identical to an all-local run.

Keeping brokers local means local TCP port allocation stays valid and there is no remote port
discovery or broker-log streaming to manage. Remote broker placement is future work — see the
plan doc `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`.

## One-time setup on each remote machine

Do this once per remote host (and once for `127.0.0.1` if you want to try the localhost-as-remote
demo below):

1. **Passwordless key-based SSH** from the manager to the remote, for the user the federates run
   as. Password prompts are refused (`BatchMode=yes`) — the run fails fast instead of hanging.

   ```bash
   ssh-keygen -t ed25519            # if you don't already have a key
   ssh-copy-id user@remote-host     # append your public key to the remote authorized_keys
   ssh user@remote-host true        # accept the host key once, confirm it's non-interactive now
   ```

   For the `127.0.0.1` demo, `ssh-copy-id` your own key into your own `~/.ssh/authorized_keys`
   and `ssh 127.0.0.1 true` once to accept the host key.

2. **The `cosim_gym` conda env** (or your own venv) exists on the remote with the CosimGym deps
   installed — at minimum `helics`, `redis`, `pydantic`, plus whatever your models need. Preflight
   runs `python -c "import helics, redis, pydantic"` in the target env and aborts the whole
   scenario with a clear message if it fails.

   > If `conda run -n <env>` is not on the remote's **non-interactive** SSH PATH (common — SSH
   > command shells often don't source `~/.bashrc`), skip conda entirely and set an explicit
   > interpreter with the `python:` machine key (see YAML reference).

3. **Network reachability from the remote to the manager**:
   - Redis on `manager_address:6379` (docker-compose binds `0.0.0.0:6379`, so it's LAN-reachable;
     open the port on the manager's firewall).
   - The HELICS broker port range (auto-assigned in `20000–30000`).
   - Mosquitto on `manager_address:11883` only if you use streaming / interface federates.

   > Ports here are the defaults. If you moved any via `src/.env` (see the "Ports" section in
   > `CLAUDE.md` / `src/.env.example`), the preflight Redis check and firewall rules follow that
   > value automatically — `RemoteExecutor` reads the port from `src/utils/ports.py`, not a literal.

   Preflight actively checks Redis reachability from the remote and aborts with a remediation hint
   if it can't connect.

The manager `rsync`s the `src/` tree to each remote's `workdir` on every run (delta transfer, so
near-instant after the first), so you do **not** manually copy code to the remotes.

## YAML reference

Two additions to a normal scenario: a top-level `deployment:` block, and a `host:` key on each
federate you want to run remotely.

```yaml
deployment:
  manager_address: 192.168.1.10   # REQUIRED when any federate has host:. The LAN IP the remote
                                  # machines use to reach this manager (broker + Redis + MQTT).
  machines:
    gpu_box:                      # alias, referenced by federates' host:
      host: 192.168.1.42          # ssh target (hostname or IP)
      user: rando                 # optional — default: current local user
      ssh_port: 22                # optional — default 22
      workdir: /home/rando/cosimgym_rt   # remote repo root; src/ is rsync'd here each run
      conda_env: cosim_gym        # optional — default "cosim_gym"
      python: null                # optional explicit interpreter path; if set, overrides
                                  #   conda_env (bypasses `conda run` entirely)

federations:
  power_grid:
    federate_configs:
      pv_federate:
        type: base
        host: gpu_box             # NEW — run this federate on the gpu_box machine
        # ... rest of the federate config is unchanged
```

Rules (enforced at parse time — a violation raises before any process starts):

- `host:` **absent** → local spawn, identical to today.
- `host:` referencing an alias not in `deployment.machines` → validation error.
- Any federate has `host:` but `deployment.manager_address` is unset → validation error.
- `host:` on a `type: rl` federate → rejected in v1 (the RL agent trains on the manager).
  `parallel_execution: true` **is** allowed on a remote federate — its worker processes then use
  the remote machine's CPUs, which is the point.
- No hardcoded IPs/paths/users/ports anywhere in code — everything comes from this block.

## How a run works

1. **Preflight + deploy** (before any broker starts): for each machine used by ≥1 federate, open
   one SSH ControlMaster connection, verify (connection, writable `workdir`, env imports, Redis
   reachability), then `rsync` `src/` into `<workdir>/src`. Any machine failing preflight aborts
   the entire scenario — no silent fallback to local.
2. **Spawn**: local federates start as before. Remote federates start via the master connection
   (`ssh -tt … conda run … python src/core/federate_launcher.py …`). The local `ssh` child *is*
   the federate's process handle — the existing monitor/cleanup loop needs no special-casing.
3. **Run**: unchanged. Federates step over HELICS; the manager's Redis holds the serialized config
   each federate fetches by URL.
4. **Collect**: after all processes exit, the manager `rsync`s each remote's
   `results/<scenario>/<sim_id>/` and `logs/<scenario>/<run_timestamp>/` back into the identical
   local directories. Remote federates write distinct files, so the merge is collision-free. A
   collection failure logs an ERROR with a manual `rsync` command — it never fails an otherwise-good
   run.
5. **Cleanup**: on normal end, signal, or exception, the manager sweeps each remote
   (`pkill -f <simulation_id>` — the sim id is unique per run) and closes every ControlMaster
   socket. Cleanup never raises.

## Try it: localhost-as-remote demo

`src/scenarios/distributed_demo.yaml` is a copy of `pv_batt_test_base.yaml` with `pv_federate`
tagged `host: local_box`, where `local_box` = `127.0.0.1`. It's the CI-able pattern: real SSH,
real `rsync`, one physical machine.

```bash
# 1. one-time: passwordless ssh to 127.0.0.1 (see setup above)
# 2. bring up Redis + Mosquitto
docker compose -f src/docker-compose.yaml up -d
# 3. run + verify against the all-local twin (pv_batt_test_base)
conda activate cosim_gym
python src/verify_distributed_demo.py          # runs both, compares every timeseries
```

`verify_distributed_demo.py` asserts the remote-spawn run reproduces the all-local numbers within
`1e-9`. Use `--no-run` to compare the latest existing runs without re-running.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `failed to open ssh control master … Permission denied (publickey)` | No passwordless key auth to the remote | `ssh-copy-id user@host`; confirm `ssh user@host true` is non-interactive |
| Preflight: `remote env sane-check failed (import helics/redis/pydantic)` | conda env missing/incomplete, or `conda run` not on the SSH PATH | Create/complete the env on the remote, **or** set `python:` to an explicit interpreter path |
| Preflight: `cannot reach Redis at <addr>:6379 from remote` | `manager_address` wrong, or firewall blocks 6379 | Set `manager_address` to a LAN IP the remote can reach; open port 6379 on the manager |
| Federate starts then exits immediately | Remote `workdir` missing synced code, or wrong CWD | Check the remote `federate_*.stdio.log` under `<workdir>/logs/…`; confirm `deploy` rsync succeeded |
| Results missing after run | Collection rsync failed | The manager logs an ERROR with the exact manual `rsync` command — run it |
| Streaming/interface federate can't reach MQTT | Adapter host defaults to `localhost` on the remote | Set `interface_config.adapter.params.host: <manager_address>` in the scenario |

## Security note

This mechanism assumes a **LAN-trusted** deployment. CosimGym's Redis and Mosquitto run
**unauthenticated** and are exposed on all interfaces so remote federates can reach them. Do **not**
expose ports 6379 / 11883 (or the broker range) on an untrusted network. Restrict them to the
simulation LAN via firewall rules or a private subnet. SSH itself is key-authenticated; the trust
boundary is the manager↔remote network, not SSH.

## Limitations (v1)

- Brokers run only on the manager. (Remote brokers = future work.)
- `type: rl` federates cannot be remote — the RL agent trains on the manager.
- Results are collected by `rsync` after the run, not streamed live. (A MinIO/S3 live sink is a
  documented future enhancement.)
- Assumes a pre-provisioned conda env / interpreter on each remote (no Docker provisioning yet).
