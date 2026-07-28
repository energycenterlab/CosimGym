# Phase H — Design Note: Brokers on Remote Machines

Status: **draft only, no code.** Written per `scaling_study_plan.md` §3 Phase H
("FUTURE FEATURE — brokers on remote machines"). This is the design a future
implementer session should follow; it is not scheduled.

> Read first: `docs/user_guide/distributed_deployment.md` (current remote-**federate**
> mechanism, which this note extends) and `scaling_study_plan.md` §2 (the
> `T_tick = max_m(compute_m + sync_m + comms_m)` cost framework this note's benefit
> analysis is grounded in).

## 0. Problem statement

Today **every** HELICS broker — the auto-inserted hierarchy broker and every
per-federation broker — runs on the manager machine. Only federates can be placed
remotely via `deployment.machines` + a federate's `host:` key
(`docs/user_guide/distributed_deployment.md`). For a scenario sharded
one-federation-per-machine (the natural scale-out for, e.g., a district with one
federation per substation), that means every intra-federation HELICS message
between two federates on the *same* remote machine still crosses the LAN twice —
federate → manager broker → federate — even though both federates live on the same
box. Placing that federation's broker on the same remote machine as its federates
would make that traffic local (in-process/loopback) and confine LAN traffic to the
one hierarchy-level hop between federation brokers. That is the feature this note
designs.

## 1. Current code, cited

### 1.1 Broker launch (manager-only, today)

`ScenarioManager._normalize_broker_and_core_configs` (`src/core/ScenarioManager.py:1593`)
resolves one scenario-wide `core_type`, allocates ports, and assigns broker
addresses — entirely for local placement:

- Port allocation is a **local bind probe**: `_get_n_available_tcp_ports`
  (`src/core/ScenarioManager.py:1494`) walks `helics_port_range()`
  (`src/utils/ports.py:98`, default `20000–30000`) and does
  `socket.bind(('', port))` on the machine `ScenarioManager` is running on. This
  only proves a port is free *on the manager* — it says nothing about any other
  machine.
- The zmq port-pairing note: `_broker_ports` (`src/core/ScenarioManager.py:1546`)
  — `"Every port a broker occupies: the advertised one, plus zmq's paired reply
  socket"` — appends `port + 1` when `core_type == 'zmq'`. `_assert_broker_ports_free`
  (`:1553`) and `_wait_for_broker_listening` (`:1567`) both probe/wait on this pair.
  Any remote-broker design must replicate this pairing check *on the broker's own
  host*, not the manager's.
- Default broker host: `default_broker_host` (`:1698`) is `127.0.0.1` unless the
  scenario has remote **federates**, in which case it becomes
  `deployment.manager_address` — i.e. today "remote-aware" only ever means "the
  broker binds a LAN address so remote federates can dial in", never "the broker
  itself lives elsewhere."
- Hierarchy-broker insertion for multi-federation scenarios: `:1770-1785`
  constructs a `BrokerConfig` with `sub_brokers=n_federations` and a fresh port;
  federation brokers get `broker_config.broker_address` pointed at it (`:1793-1794`).
- **The bare `host:port` uplink note** (verbatim from the code, `:1786-1791`):
  > "Bare host:port, NO '{core_type}://' scheme: the single-socket cores
  > (zmq_ss/tcp_ss) are HELICS coreTypes, not URI schemes, so a
  > 'zmq_ss://host:port' broker_address is malformed — the sub_broker never
  > connects to the main broker and never binds its own port, hanging the whole
  > run. helics_broker accepts a bare 'host:port' for every core_type (same form
  > the federates already use for their broker_address)."

  This was a real, fixed bug — see `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`
  changelog: *"Fixed a hierarchy-broker uplink bug (sub-broker `--broker_address`
  was built as `{core_type}://host:port` → `zmq_ss://…`, a malformed URI that hung
  the sub-broker; now bare `host:port`)."* Any remote-broker uplink string
  construction must follow the same bare `host:port` convention — do not
  reintroduce a scheme prefix.
- Actual subprocess construction: `_start_local_hierarchy_broker` (`:1802`) and
  `_start_local_federation_broker` (`:1864`) both build a `helics_broker` argv
  (`--sub_brokers=N` / `--federates=N`, `--port`, `--loglevel`, `--coreType`,
  `--name`, optional `--broker_address`), conditionally append
  `--local_interface=0.0.0.0` via `_broker_binds_externally` (`:1513`) when the
  advertised host isn't loopback, then `subprocess.Popen(..., preexec_fn=os.setsid,
  stdout=PIPE, stderr=PIPE, env=env_with_BROKER_LOG_FILE)`. Both are **local**
  `subprocess.Popen` calls — there is no SSH path for brokers today.
- Readiness: `_wait_for_broker_listening` (`:1567`) polls `_port_is_free` (`:1523`,
  a local bind probe) until the broker's port(s) are occupied, or the process exits
  early (`_assert_broker_ports_free` at `:1553` fails fast on stale orphans first).
  Both are **local-only** checks — they cannot observe a remote machine's ports.
- Log capture: `_start_broker_log_reader` (`:1926`) drains the Popen's own
  `stdout`/`stderr` pipes in daemon threads into `broker_logger`. This depends on
  Python holding the literal `subprocess.PIPE` file objects — impossible for a
  process launched over SSH the way federates are (see 1.2), unless redirected to
  a file and tailed, or streamed back some other way.
- Teardown: `_emergency_cleanup` (`:205`) does `os.killpg(os.getpgid(pid),
  SIGTERM)` then `SIGKILL` on every `self.broker_processes` entry — again, a
  **local** process-group operation; it cannot signal a broker on another machine.

### 1.2 Remote federate spawn/monitor/teardown (the pattern to imitate)

`RemoteExecutor` (`src/core/remote_executor.py`) is a thin SSH/rsync wrapper, one
instance per `deployment.machines` alias:

- `open_master()` (`:135`) opens one multiplexed `ControlMaster` per machine
  (`_master_cmd`, `:72`); every later ssh/rsync call reuses that socket via `-S
  <control_path>` (`_run_cmd`, `:103`).
- `verify()` (`:157`): connection check, `mkdir -p workdir`, `import
  helics,redis,pydantic` sane-check, and a live Redis-reachability probe from the
  remote back to `manager_address` — all before anything is spawned.
- `deploy()` (`:190`): `rsync -az --delete` of `src/` into `<workdir>/src`
  (`_rsync_deploy_cmd`, `:110`, excludes `__pycache__ .git results logs
  graphify-out *.pyc`), then creates remote `logs/`/`results/` dirs.
- `spawn_many()` (`:206`): **one** SSH session per machine (not per federate) runs
  `remote_spawner.py`, which reads that machine's federate list from a Redis key
  the manager published (`ScenarioManager._spawn_remote_batches`,
  `src/core/ScenarioManager.py:2094`) and supervises them locally. `-tt`
  allocates a pty so a SIGHUP (killing the local ssh Popen) propagates to the
  remote supervisor and its child federates share its process group — the
  "cleanup for free" property that makes the ssh child a valid process handle.
  `stdin=DEVNULL` so pty allocation doesn't drag the manager's own terminal into
  raw mode.
- Manager-side addressing: `ScenarioManager._redis_url_for` (`:1977`) points a
  remote federate at `redis://<manager_address>:<redis_port>/...` instead of
  `self.redis_url`; `deployment.manager_address` is the single LAN IP every
  remote artifact (Redis, and — in this design — a remote broker's uplink) is
  built from. It is validated present whenever any federate has `host:`
  (`ScenarioConfig._validate_deployment`, `src/utils/config_dataclasses.py:756`,
  raises at `:769` if unset, and at `:772` if `host:` names an alias missing from
  `deployment.machines`, and at `:777` if `host:` is on a `type: rl` federate).
- Monitoring: `ScenarioManager._monitor_processes` (`:2143`) polls
  `Popen.poll()`/`returncode` uniformly over `self.federate_processes` — remote
  batches carry `_cosim_weight`/`_cosim_label` (set in `_spawn_remote_batches`,
  `:2126-2127`) so counters stay in federate units, but the poll loop itself has
  **zero branching** between local and remote handles. This "process handle is
  the SSH child" trick is exactly what a remote-broker design should reuse.
- Teardown: `ScenarioManager._cleanup_remote_execution` (`:2264`), called from
  `_emergency_cleanup` (`:256`), runs `pkill -f <simulation_id>` on each remote
  (belt-and-suspenders on top of the `-tt` SIGHUP propagation) then closes every
  `ControlMaster`. `simulation_id` is unique per run, so the pattern only ever
  matches this run's own processes — this is what "orphan prevention on manager
  crash" currently means; it depends on `atexit`/signal handlers firing
  `_emergency_cleanup` (`src/core/ScenarioManager.py:140` registers it) — a
  `kill -9` on the manager process itself defeats it (same weakness for federates
  today; a remote broker inherits it identically unless something stronger is
  added, see §4).
- Results/log collection: `_collect_remote_results` (`:2216`) `rsync`s
  `results/<scenario>/<sim_id>/` and the scenario's `logs/` dir back after
  `_monitor_processes` returns; failures are logged (with a manual `rsync` hint),
  never raised.

### 1.3 Where broker config is parsed

- `BrokerConfig` (`src/utils/config_dataclasses.py:634`): `core_type, port,
  federates, log_level, host, address, broker_address, sub_brokers`. **No `host`
  validation today** — `host` is read by `ScenarioManager` (`:1708`,
  `broker_conf.host = broker_conf.host or default_broker_host`) as an
  advertised-listen-address override, never as a placement directive. This is the
  field that needs a placement meaning.
- `FederationConfig` (`:649`) owns one `BrokerConfig` (`broker_config`, default
  factory) plus its `federate_configs` dict; validated (`:679`) for federate-count
  match, unique ids/names, `n_instances >= 1`.
- `MachineConfig` (`:703`): `host, user, ssh_port, workdir, conda_env, python` —
  the SSH-target shape a remote broker placement would reuse verbatim (same alias
  namespace as `deployment.machines`).
- `DeploymentConfig` (`:716`): `manager_address` + `machines: Dict[str,
  MachineConfig]`.
- `ScenarioConfig._validate_deployment` (`:756-780`) is the single validation
  chokepoint for `host:`-on-federate today; a `broker_config.host` placement
  directive would need an equivalent validator (see §2).
- `config_reader.read_scenario_config` (`src/utils/config_reader.py:18`) is a
  thin `yaml.safe_load` → pydantic `ScenarioConfig(**data)` — no custom broker
  logic lives here; everything above is pydantic model validation plus
  `ScenarioManager`'s own normalization pass.

## 2. Config surface

Add `host: Optional[str]` to `BrokerConfig` (`src/utils/config_dataclasses.py:634`),
same alias-into-`deployment.machines` semantics as a federate's `host:` field
(`_FederateConfigBase`, `:588`).

```yaml
federations:
  substation_a:
    broker_config:
      host: sub_a_box        # NEW — alias into deployment.machines; absent = today's behavior (manager)
    federate_configs: {...}
```

### Hierarchy broker placement

Two independent questions: can the hierarchy broker be remote, and if so, where?

- **v1 recommendation: hierarchy broker stays on the manager, always.** It is the
  one node every federation broker must reach, and it is already where Redis,
  Mosquitto, and the manager's own bookkeeping live. Making it placeable too adds
  a second remote-broker code path (spawn/monitor/teardown of *the* broker whose
  loss brings down the whole scenario) for a win that is smaller than placing
  *federation* brokers — the hierarchy broker's own traffic is inherently
  cross-machine no matter where it sits, unlike a federation broker's
  intra-federation traffic. Defer hierarchy-broker placement past v1; document it
  as an open question (§8).
- If revisited later, the natural encoding is `hierarchy_broker.host:` at the
  scenario's top level next to `deployment:` (there is no existing top-level
  hierarchy-broker config block — `ScenarioManager` builds `self._hierarchy_broker_config`
  in-memory at `:1778-1785`; it would need to become YAML-visible only if it can be
  placed).

### Validation rules (extend `ScenarioConfig._validate_deployment`, `config_dataclasses.py:756`)

Reject at parse time, before any process starts (matching the existing federate
`host:` validator's contract):

- `broker_config.host` set but no top-level `deployment:` block → error (mirrors
  `:768`).
- `broker_config.host` set but `deployment.manager_address` unset → error (mirrors
  `:769`) — a remote broker still needs to know the manager's reachable address for
  Redis-config-fetch-adjacent bookkeeping and (see §3) the hierarchy uplink.
- `broker_config.host` referencing an alias absent from `deployment.machines` →
  error (mirrors `:772`).
- `broker_config.host` set with `core_type` not ending in `_ss` → **hard
  validation error, not the current warning-only path** federates get
  (`_normalize_broker_and_core_configs`, `:1648-1656`, only warns). Rationale in
  §3: a remote broker with a non-`_ss` core is not just discouraged, it is the one
  configuration that is *structurally* unworkable behind NAT (inbound listener the
  far side of NAT that nothing can dial). Because the plan's Config A
  (`scaling_study_plan.md` §5) is explicitly a NAT rig, silently degrading to a
  warning here would let a scenario "pass" locally and then hang irrecoverably on
  the real machine set — the exact bug class the bare-`host:port` fix (§1.1) was
  already needed for once.
- A federation whose broker is remote (`broker_config.host = X`) but whose
  federate_configs are a **mix** of `host: X` and other hosts/local: allowed
  (nothing stops a federate on a third machine from dialing a broker on X — HELICS
  doesn't care), but should be a **warning** noting that any federate not
  co-located with its own broker gets no local-traffic benefit — the whole point
  of the feature. This isn't a correctness issue, just a "you probably didn't mean
  this" signal.
- Local-only placements to reject: `broker_config.host` resolving (via
  `deployment.machines[alias].host`) to a loopback address (`127.0.0.1`,
  `localhost`, `::1` — reuse `ScenarioManager.LOOPBACK_HOSTS`, `:68`) should be
  accepted but treated as "local" for port-management purposes (§5) — it is the
  localhost-as-remote demo pattern already used for federates
  (`distributed_demo.yaml`) and should keep working identically for brokers, not
  be rejected.

## 3. Inverted uplink discovery (the hard part)

### Today's direction of dial

Every connection today is **manager-initiated inbound** or **remote-initiated
outbound-to-manager**:

- Federate → its federation broker: federate cores (on the manager or remote)
  dial the broker's advertised `host:port` (`FedConfig.broker_address`,
  `_normalize_broker_and_core_configs:1762`) — broker is always on the manager, so
  this is either loopback or remote→manager.
- Federation broker → hierarchy broker: the federation broker process (on the
  manager) dials the hierarchy broker's bare `host:port` uplink
  (`broker_config.broker_address`, `:1794`) — also manager-local, loopback.
- SSH control connections (deploy, spawn, collect, pkill): always
  manager-initiated outbound to the remote (`RemoteExecutor._master_cmd`, `:72`).

**Nothing today requires a socket bound on a remote machine to be dialed *into*
from the manager or from a different remote.** That is exactly what a remote
broker requires: the sub-broker's `helics_broker` process binds a listening
socket on the remote host, and the hierarchy broker (running on the manager, per
§2) — plus every federate connecting to that federation, wherever it runs — must
be able to open a connection **to** that remote-bound socket. This is a direction
of connection the codebase has never needed before.

### Per-core-type consequences

| core_type | binds a listener? | who dials in? | remote-broker feasible? |
|---|---|---|---|
| `zmq` | Yes — the core additionally binds its own reply port(s) per connected peer, independent of the broker's advertised port (see the code comment at `_normalize_broker_and_core_configs:1639-1647`: *"with zmq/tcp every federate core binds its OWN inbound listener (broker_port + 10 + n) and the broker dials back into it"*). | The broker dials back into each federate's bound port, **and** now every federate / the hierarchy broker must dial into the remote broker's bound port. | **No, behind NAT.** Two inbound-listener requirements stack (the broker's own port *and* the per-federate reply ports it must reach) — both need router/firewall configuration on the remote's NAT boundary. Feasible only on a flat, fully-routable LAN (no NAT) where every machine can bind-and-be-reached. |
| `tcp` | Same shape as `zmq` — a plain TCP core, symmetric bidirectional dialing. | Same as `zmq`. | Same as `zmq`: LAN-only, no NAT. |
| `zmq_ss` (single-socket) | The core carries all its own traffic over **one outbound** connection it initiates (see distributed-deployment note: *"single-socket cores... need no inbound listener, and are NAT-proof"*, `config_dataclasses`/`ScenarioManager` comment at `:1646-1647`). But **the broker itself still binds a listening socket for its port** — `_ss` makes the *federate cores* outbound-only, it does not make the *broker* need no listener. | Federates and the hierarchy broker dial **into** the remote broker's one bound port. | **Conditionally yes** — only if the remote broker's bound port is itself reachable inbound (port-forwarded / DNAT'd through the NAT boundary, or the remote machine is NAT-free). `_ss` solves the *federate* side of NAT; it does nothing for the fact that a broker, wherever it lives, is always the one node that must accept inbound connections. Placing the broker behind the same NAT that `_ss` was adopted to route around reintroduces exactly the problem `_ss` was for — just moved from "N federates need inbound" to "1 broker needs inbound," which is a smaller ask (one port-forward rule) but not zero. |
| `tcp_ss` | Same shape as `zmq_ss`. | Same as `zmq_ss`. | Same conditional-yes as `zmq_ss`. |

### What Config A (the scaling plan's NAT rig) can and cannot do

`scaling_study_plan.md` §5 Config A is manager `130.192.177.14` (112c, presumably
the one machine with a public/reachable address) plus `cloud1`/`cloud5` behind
NAT, `zmq_ss` forced for federate cores specifically because the remotes cannot be
dialed into. Under that topology:

- **Cannot work at all**: a federation broker placed on `cloud1` or `cloud5` with
  `zmq`/`tcp` core — the broker's own listening port is unreachable from the
  manager or from `cloud5`, full stop. No amount of `_ss` on the *federate* side
  fixes this because the broker's inbound requirement is independent of the
  federate cores' protocol.
- **Structurally strained even with `zmq_ss`**: placing a broker on `cloud1`
  additionally requires either (a) a port-forward/DNAT rule on `cloud1`'s NAT
  boundary for that broker's port — infrastructure outside CosimGym's control,
  the user/sysadmin must provision it per remote machine, or (b) some rendezvous
  layer (a relay, a reverse tunnel, a VPN mesh) that this codebase has no
  equivalent of today. Either way this is materially more operational burden than
  today's "just SSH key auth," and it is per-machine, not per-scenario — it does
  not amortize.
- **Works cleanly**: manager stays the only broker host (today's behavior, no
  change) — remains the safe default recommendation for any NAT'd machine set.
  Remote-broker placement only pays off on Config B's shape (§7): a **directly
  reachable** (non-NAT) second machine, or a flat LAN with router-level
  reachability, where a bound port is a bound port from anywhere.

**Bottom line to state plainly**: remote broker placement is not a drop-in
extension of the existing `_ss`-for-NAT mechanism. It requires either NAT-free
machines or manual port-forwarding per remote broker, on top of everything `_ss`
already does for federates. Behind NAT with no port-forwarding (Config A as
specified), remote brokers are **infeasible**, independent of core_type choice.

## 4. Lifecycle

| Concern | Federates today | Broker — what must be added |
|---|---|---|
| Spawn transport | `RemoteExecutor.spawn_many` — one SSH session per machine, `-tt` pty, reads a federate list from Redis (`remote_spawner.py`) | New: a broker isn't in the per-machine federate list — it needs its own SSH-spawned `helics_broker` argv (same shape as `_start_local_federation_broker`'s, `:1870-1878`) run via `RemoteExecutor.run`/a new `spawn_broker` method. Must precede federate spawn — federates dial the broker, so the broker must exist and be listening first (same ordering already enforced locally: `_setup_local_federation` calls `_start_local_federation_broker` before `_create_federate`, `:1476-1487`). |
| Readiness detection | N/A (federates dial out; nothing waits on them to "listen") | Today, `_wait_for_broker_listening` (`:1567`) is a **local** `socket.bind` probe — it cannot run against a remote host's ports at all (binding locally proves nothing about a remote machine). Needs a remote-side equivalent: either (a) have `remote_spawner.py`/a new remote-side helper poll the broker's own ports with the same bind-probe logic and report readiness back over Redis (a `cosim:broker:ready:<sim_id>:<alias>` key the manager polls, mirroring the existing `cosim:spawn:...` pattern at `ScenarioManager._spawn_remote_batches:2112`), or (b) have the manager attempt a real TCP connect to the (now cross-machine) port — which changes `_port_is_free`'s semantics from a passive bind-probe (deliberately chosen, `:1523-1537`, specifically to avoid ever opening a connection to a broker's zmq socket) to an active connect, a meaningfully different and riskier check the code today explicitly avoids. (a) is safer and preserves the existing passive-probe design intent. |
| Log capture | `remote_log_file` — SSH command redirects remote stdout/stderr `>> file 2>&1` (`RemoteExecutor._build_remote_command`, `:91-101`); nothing is streamed live to the manager process | Same technique applies directly: redirect `helics_broker`'s stdout/stderr into a remote log file the way `_spawn_remote_batches` already does for the federate supervisor (`supervisor_log`, `:2117-2119`). The manager loses the live in-process `_start_broker_log_reader` (`:1926`) drain-thread pattern (which depends on holding the literal Popen PIPE handles) — that becomes "tail the remote file after rsync-back," a real regression in log-during-run visibility versus local brokers, worth flagging as a limitation, not silently dropping. |
| Failure detection | SSH child's own exit code / `Popen.poll()`, polled uniformly by `_monitor_processes` (`:2143`) — reused with zero branching | A remote broker's SSH child (the local `Popen` running the SSH command that launched it) becomes another entry in `self.broker_processes` (or a new `self.remote_broker_processes` list `_monitor_processes`/`_collect_completed` also polls) — reuse the exact same poll-loop trick that already makes remote federates transparent to `_monitor_processes`. The all-or-nothing HELICS-federation-death logic at `:2165-2178` ("once one has died the survivors can never proceed") extends naturally: a dead remote broker should trigger the same abandon-and-cleanup path a dead federate does today. |
| Teardown / orphan prevention | `_cleanup_remote_execution` (`:2264`): `pkill -f <simulation_id>` per remote (belt-and-suspenders on top of `-tt` SIGHUP propagation), then close ControlMasters | Same `pkill -f <simulation_id>` sweep naturally also kills a remote `helics_broker` launched under the same SSH session pattern, **if** it was spawned with the same `-tt`-pty-plus-simulation_id-taggable argv convention. Needs the broker's argv or environment to embed `simulation_id` somewhere `pkill -f` can match (today's federate argv already includes `--simid`, `_build_federate_args:1965-1975`; the broker command (`:1809-1816`) has no such flag — would need one added, e.g. an extra `--name` suffix or a wrapper env var, purely for the `pkill` pattern match). Manager-crash orphan risk is **identical** to the existing gap for remote federates (`atexit`-registered cleanup, defeated by `kill -9` on the manager) — not solved for federates today, and a remote broker adds no *new* class of risk, just one more orphan-prone process type subject to the same known limitation. Worth a documented TODO, not a blocker, since it doesn't regress the status quo. |

## 5. Port management across machines

Today: `_get_n_available_tcp_ports` (`:1494`) allocates from a **single shared
pool** by binding on the manager only; `helics_port_range()` (`src/utils/ports.py:98`)
is one global range for the whole scenario, `20000–30000` by default.

For remote brokers this breaks down two ways:

1. **A manager-side bind probe says nothing about a remote host's ports.** Port
   `24001` being free on the manager does not mean it is free on `sub_a_box`.
   Allocation must become **per-machine**: for any `broker_config.host = X`, the
   candidate port must be probed on `X` itself — either via a remote bind-probe
   helper run over SSH (`RemoteExecutor.run(['python', '-c', '<same bind-probe
   snippet as _port_is_free>'])`), or by pre-declaring a disjoint port range per
   machine in `deployment.machines.<alias>` (simpler, no extra SSH round-trip
   per candidate port, but pushes the collision-avoidance burden onto the user's
   YAML rather than the framework — matches the "no hardcoded IPs/paths" spirit
   of the existing deployment doc only if the range is declared, not invented, by
   the same YAML that declares the machine).
2. **Collision rules multiply**: today one scenario needs `n_federations +
   (1 if hierarchy)` free ports on **one** host. With remote brokers, each
   machine hosting ≥1 broker needs its own reservation, and the zmq
   `port+1` pairing (`_broker_ports`, `:1546`) must be checked **on that
   machine**, not the manager's. Recommend extending `_assert_broker_ports_free`
   (`:1553`) to dispatch to a remote bind-probe when `broker_conf.host` resolves
   to a non-loopback machine, mirroring the `_broker_binds_externally` (`:1513`)
   dispatch already used for the `--local_interface=0.0.0.0` decision.

No change needed to the *hierarchy* broker's own port bookkeeping if it stays on
the manager (§2 v1 recommendation) — only per-federation broker ports move.

## 6. Refactor sketch (ordered, minimal)

| # | Change | Files / functions | Risk |
|---|---|---|---|
| 1 | Add `host: Optional[str]` to `BrokerConfig`; extend `ScenarioConfig._validate_deployment` with the broker-host rules in §2 | `src/utils/config_dataclasses.py:634` (`BrokerConfig`), `:756` (`_validate_deployment`) | **Low** — pure schema/validation addition, `extra='ignore'` on `BrokerConfig` already tolerates the new optional field with zero migration risk to existing YAML. |
| 2 | Per-machine port allocation + remote bind-probe helper | `ScenarioManager._get_n_available_tcp_ports` (`:1494`), `_assert_broker_ports_free` (`:1553`), `_port_is_free` (`:1523`) — each needs a remote-dispatch branch keyed on `broker_conf.host` | **Medium** — touches code with delicate SO_REUSEADDR/TIME_WAIT semantics (`_port_is_free`'s docstring, `:1524-1537`) that must be replicated correctly on the remote side, not just relocated. |
| 3 | `RemoteExecutor.spawn_broker(...)` — SSH-launch a `helics_broker` argv (reuse `_start_local_federation_broker`'s command construction, parameterized by target host) | New method on `src/core/remote_executor.py`; argv-building logic factored out of `ScenarioManager._start_local_federation_broker` (`:1864`) / `_start_local_hierarchy_broker` (`:1802`) so both local and remote paths share one `_build_broker_cmd()` (mirrors the existing `_build_federate_args` shared-builder pattern, `:1953`) | **Medium** — must preserve the bare-`host:port` uplink convention (§1.1) exactly; a scheme-prefix regression here reintroduces the exact bug already fixed once. |
| 4 | Remote broker readiness signal — Redis-key handshake (§4) | New: a remote-side readiness probe (extend `remote_spawner.py` or a small new script), plus `ScenarioManager._wait_for_broker_listening` gaining a remote-poll branch (`:1567`) | **Medium-high** — readiness is the crux: get it wrong and every federate on that federation times out identically to today's stale-orphan-broker failure mode (`_assert_broker_ports_free`'s docstring, `:1554-1558`), just one layer further from the real cause. |
| 5 | Broker log capture over SSH (redirect-to-file + rsync-back, drop the live-drain-thread illusion for remote brokers) | `ScenarioManager._start_broker_log_reader` (`:1926`) gains a no-op/skip branch for remote brokers; rely on `_collect_remote_results`' existing log rsync (`:2216`) | **Low** — accepted feature regression (no live tail), not a correctness risk. |
| 6 | Teardown: track remote broker SSH children in `self.broker_processes` (or a parallel list `_monitor_processes`/`_collect_completed` also walks); ensure `pkill -f <simulation_id>` pattern-matches the remote broker argv (§4) | `ScenarioManager._emergency_cleanup` (`:205`), `_cleanup_remote_execution` (`:2264`), broker argv needs a `simulation_id`-taggable token added | **Medium** — teardown correctness is exactly where orphan processes come from if missed; needs explicit kill-a-scenario-mid-run testing, not just happy-path. |
| 7 | Wire it all together: `_setup_local_federation` (`:1466`) branches to a remote-broker path when `federation_conf.broker_config.host` is set, ordered before `_create_federate` as today | `ScenarioManager._setup_local_federation` (`:1466`) | **High** — this is the integration point; every ordering assumption in §4 (broker-before-federates, preflight-before-broker) converges here. |

## 7. Test plan

Per repo convention, `tests/regression_suite.py` is the living contract — a new
feature needs both a standalone scenario in `SCENARIOS` and a combination entry in
`COMBOS` (`tests/regression_suite.py:44`, `:81`).

- **New scenario** (localhost-as-remote broker, mirrors `distributed_demo.yaml`'s
  pattern for federates): a single-federation scenario with `broker_config.host:
  local_box` (`127.0.0.1` alias) and all federates local — proves a remote broker
  works at all, and that manager-local federates can dial out to it. Add to
  `SCENARIOS` as e.g. `("DIST remote-broker", "distributed_broker_demo")`.
- **New combination**: remote broker **and** remote federate on the *same* alias
  (the actual point of the feature — co-located broker + federates, intra-
  federation traffic never leaving the box) vs. remote broker with federates
  scattered across other machines (the degraded case flagged as a warning in §2).
  Add to `COMBOS`, e.g. `("DIST-BROKER + DIST-FED (co-located)",
  "combo_remote_broker_colocated")`.
- **Multi-federation + remote broker**: one federation's broker remote, the
  hierarchy broker on the manager (§2 v1) — validates the inverted-uplink
  discovery (§3) actually resolves for the `zmq_ss`/`tcp_ss` case on a
  NAT-free/localhost target. Extend `distributed_multifederation_test.yaml`'s
  pattern or add a sibling scenario.
- **Negative/validation tests** (pytest, alongside `tests/test_rl_config.py`'s
  `extra='forbid'`-style parse-gate tests): `broker_config.host` with no
  `deployment` block → `ValueError`; unknown alias → `ValueError`; non-`_ss`
  `core_type` with a remote broker → `ValueError` (the hard-reject rule from §2,
  distinct from the federate path's warning-only precedent at
  `_normalize_broker_and_core_configs:1648-1656`).
- **Teardown test**: extend `tests/test_scenario_manager_remote.py`'s
  `TestCollectionAndCleanup` pattern (`:123`) to assert a remote broker's SSH
  child is swept by `_cleanup_remote_execution` alongside remote federates.
- **`RUN_CLOUD=1` real-machine variant**: once Config B (a directly-reachable,
  non-NAT second machine — still TBD per `scaling_study_plan.md` §7.1) exists,
  add a `CLOUD_OPTIONAL` entry exercising a real cross-machine remote broker; do
  **not** attempt this against Config A (NAT) per §3's feasibility finding.

## 8. Expected benefit + when it does not pay off

In the plan's framework (`scaling_study_plan.md` §2):

```
T_tick = max_m ( compute_m + sync_m + comms_m )
```

A remote per-federation broker converts that federation's **intra-federation**
traffic (federate ↔ its own broker, today always a `comms` term crossing the LAN
to the manager whenever the federate is remote) into **local** traffic (loopback/
IPC on the shared machine), at the cost of adding one **hierarchy-level** LAN hop
that did not exist for that federation before (federation broker ↔ hierarchy
broker, now cross-machine instead of loopback-on-manager).

It wins when:

```
comms_saved(intra-federation, N_local_federates_per_broker)  >  comms_added(hierarchy hop, cross-federation message volume)
```

i.e. when a federation has **many federates exchanging data with each other**
relative to how much it exchanges **across** federations — the classic
shard-locality argument. A federation that is nearly self-contained (little
cross-federation traffic) benefits a lot; a federation whose federates barely
talk to each other but constantly reach across federation boundaries could get
*worse* by adding a hierarchy hop for traffic that used to be manager-local.

**Do not invent numbers here.** The only two facts already measured that bound
this argument, both from `scripts/scaling_study/findings/README.md`:

- Hierarchy-broker cost is **setup-time only, not per-tick** — "hierarchy-broker
  cost is setup-only (~0.37 s/fed); tick-flat; 256 feds local"
  (`findings/README.md`, Phase 3 row). This means the *existence* of a hierarchy
  broker is cheap regardless of placement; the open question this note cannot
  answer is the **per-tick cost of hierarchy-level messages once they carry
  actual data**, which is exactly the still-open Phase D gap: *"Data exchange
  never studied... needs a `comms` cost term the model lacks"*
  (`findings/README.md`, "Known gaps / open" #1).
- Because `comms_m` is not yet fitted for any distance (`findings/README.md`
  scope caveat: *"all controlled sweeps used heavy_compute_dummy with zero data
  exchange... Data-exchange coupling has not been studied yet"*), **this note
  cannot quantify the crossover point** — that requires Phase D's `comms(...)`
  fit (intra-federation vs. cross-federation vs. cross-machine coefficients) to
  exist first. Phase H is explicitly sequenced as needing no hardware and no
  dependency on Phase D to *design*, but a real go/no-go recommendation for any
  specific scenario needs Phase D's numbers, not this note's.

**Where it clearly does not pay off**: any NAT'd machine set without manual
port-forwarding (§3) — infeasible regardless of the cost tradeoff. Also any
federation that is small (few federates) or already local-only (no federate on
it has `host:` set) — nothing to save by moving its broker, only the setup and
operational cost of one more SSH-managed process.

## 9. Open questions (explicit, unresolved by this note)

1. **Hierarchy broker placement** — deferred to "stays on manager" for v1 (§2).
   If a district truly has no natural manager (every federation remote, no
   central always-on host), the hierarchy broker itself becomes a placement
   decision this note does not design.
2. **Readiness handshake mechanism** — §4 proposes a Redis-key poll as the safer
   option over an active remote connect, but this is a design choice, not a
   verified one; needs prototyping before landing.
3. **`pkill` argv tagging for brokers** — the exact mechanism to make a remote
   `helics_broker` process pattern-match `simulation_id` (today's federate argv
   has `--simid`; the broker command does not) is unspecified — pick a
   convention when implementing (§6, item 6).
4. **Port allocation strategy** — §5 offers two options (remote bind-probe over
   SSH vs. pre-declared disjoint per-machine ranges in YAML) without picking one;
   the SSH round-trip cost of the former vs. the configuration burden of the
   latter needs a decision.
5. **Live broker log visibility during a run** — accepted as a regression in §4/§6
   item 5 (file + rsync-after instead of live drain-thread), but no design for
   closing that gap (e.g. a lightweight remote tail-and-forward) is attempted
   here.
6. **Quantified crossover point** — explicitly blocked on Phase D's `comms(...)`
   fit (§8); this note states the shape of the tradeoff, not the numbers.
7. **Manager-crash orphan prevention** — identical unsolved gap as remote
   federates today (`atexit`/signal-handler cleanup defeated by `kill -9`);
   whether a remote broker's higher blast-radius (its failure kills an entire
   federation, not one federate) justifies a stronger mechanism (e.g. a
   remote watchdog / heartbeat) than what federates get today is unresolved.
