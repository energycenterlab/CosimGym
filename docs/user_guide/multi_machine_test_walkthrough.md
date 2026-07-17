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

### A0. Set `core_type: "zmq_ss"` — do this first

In your scenario's `broker_config`, use the **single-socket** protocol for distributed runs:

```yaml
federations:
  federation_1:
    broker_config:
      core_type: "zmq_ss"     # NOT "zmq" — see below
```

`ScenarioManager` propagates one protocol scenario-wide, so this single line covers every federate.

**Why this matters more than anything else in this guide.** Measured on a real 3-machine run:

| | plain `zmq` / `tcp` | `zmq_ss` / `tcp_ss` |
|---|---|---|
| federate → broker | outbound to `<port>`/`<port>+1` | outbound to `<port>` |
| broker → federate | **dials back** into a listener each core binds at `broker_port+10+n` | same outbound socket — **no listener** |
| remote needs inbound rules | **yes**, whole core port range | **no** |
| works behind NAT | **no — impossible** | **yes** |

Plain `zmq` requires the manager to open a TCP connection *into* every remote federate core. If a
remote is behind NAT (very common — check with `ip -4 -o addr show` on the remote; if its real IP
isn't the address you put in the YAML, it's NAT'd), the core advertises the **private** address it
bound, the broker's reply is unroutable, and registration times out. No firewall rule can fix that,
because the address itself is unreachable.

The failure is deceptive: the TCP connect **succeeds** (`fed1 (0)[connected]`), then it dies ~30s
later with `core is unable to register and has timed out` and `broker id=0` — which looks like a
broker bug, not a NAT problem.

`zmq_ss` sidesteps all of it: everything rides the core's own outbound connection.

### A1. Confirm LAN IP
```bash
hostname -I            # pick the LAN-reachable address, e.g. 130.192.177.14
```

### A2. Firewall — what actually needs opening depends on `core_type`

> ✅ **Use `core_type: "zmq_ss"` and this step is small: only the MANAGER needs inbound rules.**
> This is the recommended setup for distributed runs — see A0 below for why.

**With `zmq_ss` / `tcp_ss` (single socket — recommended):** federate cores make only **outbound**
connections and bind **no inbound listener** (measured). So:

- **manager**: needs inbound on the broker ports from each remote;
- **remotes**: need **nothing** — no inbound rules at all. Works behind NAT.

**With plain `zmq` / `tcp`:** traffic is **bidirectional** and you need rules on *both* sides:

1. **federate → broker**: each federate dials the manager on `<port>` + `<port>+1`.
2. **broker → federate**: the broker dials **each federate core back** at
   **`broker_port + 10 + n`** (23414, 23415, … for federates 1..n).

That second half means every remote must be *directly reachable* from the manager — which is
impossible behind NAT, no matter what you open. Symptom: `core is unable to register and has timed
out` **after** a successful TCP connect (`[connected]`, `broker id=0`). If you must use plain `zmq`,
apply the manager rules below **and** the same range on every remote, from `<MANAGER_IP>`.

**Open the whole configured HELICS range once, and every scenario works** — don't hand-pick ports
per scenario. A rule cut to one scenario's broker port silently breaks the next one (scenarios in
this repo pin anything from `23404` to `23622`, and auto-assigned brokers can land anywhere in the
range). Derive the range from the project's own config so the rule tracks `src/.env` instead of
duplicating numbers:

```bash
cd /path/to/CosimGym
read LO HI <<< "$(PYTHONPATH=src python -c 'from utils.ports import helics_port_range; print(*helics_port_range())')"
echo "HELICS range: $LO:$HI"          # default: 20000 30000
```

```bash
# --- ON THE MANAGER: let the remotes reach the broker ---
sudo ufw allow proto tcp from <MACHINE_A_IP> to any port $LO:$HI comment 'CosimGym HELICS'
sudo ufw allow proto tcp from <MACHINE_B_IP> to any port $LO:$HI comment 'CosimGym HELICS'

# --- ON EACH REMOTE: let the broker reach the federate cores ---
sudo ufw allow proto tcp from <MANAGER_IP> to any port $LO:$HI comment 'CosimGym HELICS'
```

Why the single range is the right unit, rather than the exact ports in play:

- **auto-assigned brokers** are scanned out of `COSIM_HELICS_PORT_MIN..MAX` (`src/.env`) — by
  definition anywhere in it;
- **pinned** `broker_config.port` values live in this range too (check yours:
  `grep -A6 broker_config: src/scenarios/*.yaml | grep 'port:'`);
- **federate cores** sit at `broker_port + 10 + n`, so they follow the broker wherever it lands.

Change `COSIM_HELICS_PORT_MIN/MAX` in `src/.env` and you must re-issue these rules with the new
range. If you ever pin a `broker_config.port` **outside** it, that port and its cores need their own
rule — prefer keeping pinned ports inside the range.

**Tightening it:** the range is already scoped to specific source IPs, so only those machines can
reach it. If your site requires narrower rules, set `COSIM_HELICS_PORT_MIN/MAX` to a small window
(e.g. `23400:23600`), keep every pinned port inside it, and open exactly that.

**No rule is needed for Redis (6379)** — Redis runs in Docker, and Docker's iptables DNAT rules
bypass ufw entirely, so it is already reachable. That is exactly why a Redis-only reachability
test gives a false "all good" (see B4).

> ⚠️ **Shared no-sudo server:** don't skip this — an unreachable broker is the single most common
> failure, and B4 (as written below) is the only thing that catches it. If you can't run `ufw`,
> send these rules to whoever administers the machines. Don't tunnel around an institutional
> firewall on a shared box without agreeing it with them first.

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
> ⚠️ **Testing Redis alone is NOT enough and will mislead you.** Redis runs in **Docker**, and
> Docker installs its own iptables DNAT rules that **bypass the host firewall**. So `6379` can be
> perfectly reachable while every *native* port — including the HELICS broker's `23404`/`23405` —
> is silently dropped. A passing Redis check proves nothing about the broker. The `nc` line above
> is also useless while no broker is listening. Test the broker ports **with something actually
> bound to them**:

**B4a — on the MANAGER**, start throwaway listeners on the broker ports:
```bash
python3 -m http.server 23404 --bind 0.0.0.0 &     # stand-ins for the broker
python3 -m http.server 23405 --bind 0.0.0.0 &
```

**B4b — on EACH REMOTE**, probe all three ports:
```bash
for P in 6379 23404 23405; do
  python3 -c "
import socket
try:
    socket.create_connection(('130.192.177.14',$P),5); print('port $P: OPEN')
except Exception as e: print('port $P: BLOCKED ->', type(e).__name__)
"
done
```

**B4c — back on the MANAGER**, stop them: `kill %1 %2`

| Result | Meaning |
|---|---|
| all three `OPEN` | good — proceed |
| `6379 OPEN`, `23404/23405 BLOCKED` | **the common case.** Host firewall drops native inbound while Docker bypasses it for Redis. The run *will* fail with every federate logging `zmq broker connection timed out`. Fix via A2 before running |
| all `BLOCKED` | wrong `manager_address`, or a network ACL between subnets |

A `timeout` means packets are **dropped** (firewall). `ConnectionRefusedError` means they arrive but
nothing is listening (listener not up, or wrong port) — that is *not* a firewall problem.

**No sudo to fix it?** The ports must be opened by whoever administers the manager — send them A2.
Don't tunnel around an institutional firewall on a shared machine on your own; if SSH
port-forwarding would be acceptable in your environment, agree that with the admin first.

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

Run from the **project root** (not `src/`) — `logs/` and `results/` are created relative to
cwd, and every other script in this repo (`test_script.py` etc.) assumes root. `python -c` does
NOT auto-add `src/` to `sys.path` the way `python src/test_script.py` does (that trick only
works when Python is given a *script path* — for `-c` you must set `PYTHONPATH` yourself):

```bash
conda activate cosim_gym                 # puts helics_broker on PATH (manager needs it)
cd /path/to/CosimGym                     # project root
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('distributed_demo_multi')"
```

> Ran it as `cd src && python -c "..."` instead? It'll "work" (import resolves) but `logs/` and
> `results/` land under `src/logs` / `src/results` instead of the repo root — inconsistent with
> every other script here. Move/delete that stray `src/logs` and rerun with `PYTHONPATH=src` from
> root instead.

Watch the flow in the manager log:
`preflight (per machine) → deploy (rsync) → broker up → 5 federates spawn (2 remote) → run → collect → cleanup`.

On success, remote federate results are rsynced back to `results/distributed_demo_multi/<sim_id>/`.
Same physics as `pv_batt_test_base`, so values must match an all-local run.

**Verify against the all-local twin** (physics identical):
```bash
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('pv_batt_test_base')"
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
| `conda: command not found` over ssh | non-interactive ssh doesn't source `~/.bashrc`, so conda isn't on PATH | set `python:` to the env's interpreter (PART C) |
| `port(s) 23404, 23405 already in use — most likely an orphaned broker` | a broker from an earlier interrupted run is still holding the port | `ps -eo pid,args \| grep -F helics_broker`, then `kill -TERM <pid>` (escalate to `kill -KILL -<pid>` for the whole group) |
| `Broker ... did not start listening` / `exited during startup` | broker couldn't bind — port taken, or bad `--coreType` | read the broker log under `brokers/`; check ports as above |
| federates log `[connected]` then `core is unable to register and has timed out`, `broker id=0` | **TCP works; the broker's reply can't get back.** Remote is behind NAT and its core advertised a private address — or plain `zmq` with the core port range closed on the remote | set `core_type: "zmq_ss"` (A0). No firewall change can fix the NAT case |
| manager log warns `core_type is 'zmq'` with remote federates | you're on a bidirectional protocol for a distributed run | switch to `zmq_ss` (A0) |
| every federate logs `zmq broker connection timed out`, then `N/5 federate(s) exited with a non-zero code` | broker not reachable from the remotes: firewall (A2) or wrong `manager_address` | re-run B4 from each remote; check the broker log says it bound `0.0.0.0`, not `127.0.0.1` |
| `helics_broker not found` on manager | env bin not on PATH | `conda activate cosim_gym` before running |
| results missing after run | collection rsync failed | manager logs an ERROR per machine with the manual rsync command — run it |
| remote `results` rsync errors with `No such file or directory` | the remote federates failed before writing any results | not the real problem — read the collected remote logs under `logs/<scenario>/<ts>/federates/` for the actual error |

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
