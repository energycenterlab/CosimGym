# Multi-Machine Distributed Test — Step-by-Step Walkthrough

Hands-on runbook for testing **real multi-machine** federate distribution over SSH (federates on
2 remote machines + the manager). For the full feature reference see
[`distributed_deployment.md`](distributed_deployment.md).

Demo scenario: **`src/scenarios/distributed_demo_multi.yaml`** (already in the repo).

---

## Topology

```
        ┌──────────────── MANAGER (this machine, 130.192.177.14) ────────────────┐
        │  Redis + Mosquitto (docker)   HELICS broker(s)                          │
        │  rb_controller · battery_federate · load_federate  (local subprocesses) │
        └───────────▲───────────────────────▲────────────────────────────────────┘
                    │ ssh spawn + HELICS/LAN │
       ┌────────────┴─────────┐   ┌──────────┴───────────┐
       │ MACHINE_A            │   │ MACHINE_B            │
       │ weather_federate     │──▶│ pv_federate          │   ← weather→pv is REMOTE→REMOTE
       │ (publishes DryBulb…) │   │ (subs weather, pubs  │      over HELICS across machines
       └──────────────────────┘   │  PV_power → manager) │
                                   └──────────────────────┘
```

- **Manager** (the machine you launch from): all brokers, Redis, Mosquitto, and the federates with
  no `host:` (`rb_controller`, `battery_federate`, `load_federate`).
- **machine_a**: `weather_federate`.
- **machine_b**: `pv_federate`.
- Cross-machine HELICS data flow: `weather (A) → pv (B)` (remote→remote), `pv (B) → controller/battery (manager)` (remote→manager).

---

## The 3 real-network gotchas (vs the localhost test)

1. **`manager_address` = real LAN IP** (`130.192.177.14`), not `127.0.0.1`. Remotes dial this for Redis + broker.
2. **Firewall / reachability** — remotes must reach the manager's Redis (`6379`) + HELICS broker
   port (`23404`, plus `23405` for ZMQ's paired socket). ZMQ may use extra ports → open the range to be safe.
3. **Per-machine env** — the `cosim_gym` conda env (model deps + CSV data) must exist on *each* remote.
   `rsync` copies `src/` (code + CSV resources), but **not** the conda env.

---

## Prerequisites checklist

- [ ] Manager LAN IP known (`hostname -I` → here `130.192.177.14`).
- [ ] Two remote machines on the same LAN, reachable by ssh.
- [ ] An ssh keypair on the manager (`~/.ssh/id_ed25519`).
- [ ] `environment.yml` from this repo (to build the env on each remote).

---

## PART A — Manager machine (this one)

### A1. Confirm LAN IP
```bash
hostname -I            # pick the LAN-reachable address, e.g. 130.192.177.14
```

### A2. Firewall — open inbound from the remotes (Redis + broker ports)
```bash
# If you have sudo/ufw:
sudo ufw allow from <MACHINE_A_IP> to any port 6379,23404,23405 proto tcp
sudo ufw allow from <MACHINE_B_IP> to any port 6379,23404,23405 proto tcp
# (optional, robust) the full HELICS auto-assign range:
sudo ufw allow from <REMOTE_SUBNET> to any port 20000:30000 proto tcp
```
> ⚠️ **Shared no-sudo server:** if you can't run `ufw`, skip for now — many LANs allow internal
> traffic. Step **B4** *tests* reachability; only if it fails do you need an admin to open ports.

### A3. Bring up the manager stack
```bash
docker compose -f src/docker-compose.yaml up -d
docker compose -f src/docker-compose.yaml ps
docker port cosim_redis          # must show 0.0.0.0:6379 (LAN-reachable, NOT 127.0.0.1)
```
Redis binds `0.0.0.0:6379` by default → LAN-reachable.

> **Port conflict?** All infra ports are configurable in one place — copy `src/.env.example` to
> `src/.env` and change `COSIM_REDIS_PORT` etc. Both docker-compose and the Python code read it.
> See the "Ports" section in `CLAUDE.md`.

---

## PART B — EACH remote machine (do on BOTH A and B)

### B1. Create the `cosim_gym` conda env (one-time, ON the remote)
```bash
# on the remote machine:
git clone <your CosimGym repo> ~/CosimGym      # or scp just environment.yml
cd ~/CosimGym
conda env create -f environment.yml            # cosim_gym: helics 3.6.1, redis, pydantic, model deps
conda activate cosim_gym
python -c "import helics,redis,pydantic; print('env OK')"
which helics_broker || echo "no broker CLI here (fine — brokers run on the manager only)"
```
> The CSV resource files (`src/models/.../resources/*.csv`) arrive via `rsync` each run — the env
> only needs the Python packages.

### B2. Passwordless ssh FROM manager TO this remote (run ON the manager)
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <USER>@<REMOTE_IP>
ssh -o BatchMode=yes <USER>@<REMOTE_IP> 'echo SSH_OK; hostname'   # must print SSH_OK, no password
```
`RemoteExecutor` connects with `BatchMode=yes` — it never prompts for a password.

### B3. Workdir writable
The `workdir` you'll set in YAML (e.g. `/home/<user>/cosimgym_rt`) must be creatable/writable by
that user. `deploy()` does `mkdir -p` + rsync there.

### B4. CRITICAL — test the remote can reach the manager (run ON each remote)
```bash
python -c "import socket; socket.create_connection(('130.192.177.14',6379),5); print('REDIS_REACHABLE')"
nc -zv 130.192.177.14 23404      # broker port
```
- `REDIS_REACHABLE` → good.
- Timeout/refused → firewall blocks it → open the Part A2 ports (admin if no sudo). **Fix before
  running** — preflight fails this same check otherwise.

---

## PART C — fill in the demo YAML

Edit `src/scenarios/distributed_demo_multi.yaml`, replace every `<<...>>`:

```yaml
deployment:
  manager_address: "130.192.177.14"      # confirm your LAN IP
  machines:
    machine_a:
      host: "<<MACHINE_A_IP>>"
      user: "<<MACHINE_A_USER>>"
      ssh_port: 22
      workdir: "<<MACHINE_A_WORKDIR>>"   # e.g. /home/<user>/cosimgym_rt
      conda_env: "cosim_gym"
      # python: "/home/<user>/miniconda3/envs/cosim_gym/bin/python"   # if conda not on ssh PATH
    machine_b:
      host: "<<MACHINE_B_IP>>"
      user: "<<MACHINE_B_USER>>"
      ssh_port: 22
      workdir: "<<MACHINE_B_WORKDIR>>"
      conda_env: "cosim_gym"
```

Test whether `conda run` resolves over ssh (decides if you need the `python:` line):
```bash
ssh <USER>@<REMOTE_IP> 'conda run -n cosim_gym python -V'
```
- Prints a Python version → leave `conda_env` as-is.
- `conda: command not found` → uncomment `python:` with the env's interpreter path
  (find it on the remote: `conda run -n cosim_gym which python`).

---

## PART D — run + verify

```bash
conda activate cosim_gym                 # puts helics_broker on PATH (manager needs it)
python -c "from core.ScenarioManager import main; main('distributed_demo_multi')"
```

Watch the flow in the manager log:
`preflight (per machine) → deploy (rsync) → broker up → 5 federates spawn (2 remote) → run → collect → cleanup`.

On success, remote federate results are rsynced back to `results/distributed_demo_multi/<sim_id>/`.
Same physics as `pv_batt_test_base`, so values must match an all-local run.

**Verify against the all-local twin** (physics identical):
```bash
python -c "from core.ScenarioManager import main; main('pv_batt_test_base')"
# then diff the pv/battery timeseries between the two results/<scenario>/<sim_id>/ trees
```

**Orphan check after an interrupt** (Ctrl+C mid-run):
```bash
# on each remote, afterwards — must be empty:
ps -eo args | grep -F federate_launcher.py | grep -Fv grep | grep -Fv 'bash -c'
```

---

## PART E — troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| preflight `Permission denied (publickey)` | B2 not done for that host | `ssh-copy-id` to it |
| preflight `cannot reach Redis …:6379` | firewall (A2) or wrong `manager_address` | open port / fix IP (test with B4) |
| preflight `import helics/redis/pydantic failed` | env missing on remote | B1 on that machine, or set `python:` |
| federates spawn but HELICS never converges / hangs at init | broker port (23404/23405) blocked from remote | open broker ports on manager firewall |
| `helics_broker not found` on manager | env bin not on PATH | `conda activate cosim_gym` before running |
| results missing after run | collection rsync failed | manager logs an ERROR with the manual rsync command — run it |

---

## Fill-in table (record your setup)

| field | machine_a | machine_b |
|---|---|---|
| IP / hostname |  |  |
| ssh user |  |  |
| workdir |  |  |
| `cosim_gym` env present? |  |  |
| reaches manager (B4 pass)? |  |  |

Manager LAN IP: `130.192.177.14`   ·   Manager has sudo/firewall access: ______
