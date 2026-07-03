# Distributed SSH Spawning — Design Choices: Pros, Cons, Alternatives

> Companion to `distributed_ssh_spawning_plan.md`. Rationale record: why each choice won, what else considered, when to revisit. Caveman prose, full technical content.

## 1. Remote Process Spawn Mechanism

### CHOSEN: OpenSSH client subprocess + ControlMaster

**Pros**
- Zero new pip deps. `ssh`/`rsync` present on every Linux box.
- `ssh` child Popen IS remote process handle → `_monitor_processes` poll loop, exit-code propagation, `_emergency_cleanup` group-kill all work unchanged. Biggest single simplification of whole design.
- `-tt` pty → kill ssh = SIGHUP remote = no orphans, free cleanup.
- ControlMaster: one auth handshake per machine; subsequent spawns/rsyncs ~ms over shared socket.
- User's ssh config (`~/.ssh/config`, jump hosts, keys, agent) works for free.
- Debuggable: copy failing command from log, run by hand.

**Cons**
- One ssh process per remote federate on manager (~10-20MB RSS each). Fine ≤50 federates; heavy at hundreds.
- Remote stderr/stdout mixed through pty; pty can mangle line endings, no clean stream separation.
- String-through-shell surface: remote command must be shlex-quoted carefully (plan mandates).
- Windows remote hosts: effectively unsupported.
- No structured API: parsing failures = parsing text.

### ALT A: paramiko / Fabric

**Pros**: Pythonic API, structured exec results (rc/out/err separate), SFTP built-in, no ssh binary needed, testable with mocks.
**Cons**: new dependency (paramiko C-deps sometimes painful); connection objects ≠ Popen → need adapter layer in monitor loop (poll thread per channel or callback plumbing) → MORE code than chosen option; slower than native ssh for bulk transfer; ControlMaster-style multiplexing manual; ~none of user's ssh config respected by default.
**Verdict**: richer API not worth breaking the Popen-uniformity that keeps ScenarioManager untouched.

### ALT B: asyncssh

**Pros**: fastest at scale (100s of hosts), single event loop, persistent sessions, clean streams.
**Cons**: forces asyncio into fully-synchronous ScenarioManager → invasive rewrite or awkward thread-bridge; new dep; harder handoff/debug. Overengineering at current scale (2-5 machines).
**Verdict**: revisit only if machine count >10 and spawn latency measured as bottleneck. RemoteExecutor interface designed so internals swappable.

### ALT C: cluster/distributed frameworks (Ray, Dask, Celery, mpirun/SLURM)

**Pros**: battle-tested scheduling, retries, autoscaling, dashboards; Ray already adjacent (RLlib backend exists in repo).
**Cons**: heavyweight runtime daemon on every machine; own serialization + networking stack colliding with HELICS's own broker topology (two orchestration layers fighting); federates are long-lived stateful HELICS processes, not stateless tasks — poor fit for task-queue model; huge conceptual surface for Sonnet-maintained codebase; deployment block would leak framework concepts into YAML.
**Verdict**: rejected. HELICS already IS the distributed runtime; only need dumb remote process start. Exception: if RL training itself scales out later, Ray for the AGENT side (not federates) is natural — orthogonal decision.

### ALT D: remote agent daemon (custom RPC server per machine, or systemd units)

**Pros**: no per-federate ssh process; structured control channel; survives manager network blips.
**Cons**: must build/install/version/secure a daemon = new service to babysit; auth story from scratch; failure modes multiply; classic overengineering for "start N processes".
**Verdict**: rejected outright.

## 2. Code + Environment Deployment

### CHOSEN: pre-provisioned conda env + rsync `src/` each run

**Pros**
- Runs start fast: delta rsync ≈ instant after first sync.
- No stale-code bugs: code on remote always matches manager working tree (including uncommitted edits — good for research iteration).
- One-time env setup documented, matches how user already provisions machines.
- No registry/build infra.

**Cons**
- Env drift: conda env on remote can diverge (different helics version) → subtle bugs. Mitigated by preflight import check; NOT fully mitigated (version mismatch passes import). Possible hardening: preflight compares `helics.__version__` + key package versions vs manager.
- Assumes rsync+ssh present, POSIX remote.
- Uncommitted-code sync is double-edged: irreproducible runs if user forgets what state was synced. (Log git SHA + `git diff --stat` at deploy time = cheap mitigation, worth adding.)

### ALT A: Docker image per run

**Pros**: hermetic, reproducible, env drift impossible; same image manager+remote.
**Cons**: image build+push+pull per code change = slow inner loop (minutes vs ms); needs registry reachable by all machines; HELICS/Redis networking through container needs host-mode or port maps; GPU passthrough extra config for future RL case; heavier docs. Also conda env inside image ~GB-scale.
**Verdict**: right choice for production/paper-artifact reproducibility, wrong for daily research iteration. Future enhancement; compose file already exists as base.

### ALT B: pip-installable package (wheel push)

**Pros**: versioned deploys, clean.
**Cons**: repo not packaged today (scripts + relative paths, `src/core/mappings.yaml` CWD-relative); packaging refactor is own project; still needs env for helics binaries.
**Verdict**: rejected; prerequisite refactor out of scope.

### ALT C: shared filesystem (NFS) for code

**Pros**: zero sync code, edits instantly visible everywhere.
**Cons**: infra assumption (NFS server, mounts, permissions) framework can't verify or install; Python imports over NFS slow cold-start; locking/caching weirdness; couples framework to site setup.
**Verdict**: rejected as requirement. Nothing PREVENTS user pointing `workdir` at NFS mount and rsync becoming no-op — compatible by accident, fine.

## 3. Results + Log Collection

### CHOSEN: rsync-back after run end

**Pros**
- Zero change to storage code paths (`sink: json` writes at end, `sink: parquet` finalized before exit — both complete on remote disk before collection runs).
- One collection step, trivially debuggable, manual recovery command printable on failure.
- Uses existing master connection.

**Cons**
- No live/central results during run (dashboard "Results" page blind until run ends; "Live" MQTT page unaffected — streaming still works if federate streams).
- Remote disk fills for very long runs; loss of remote machine mid-run = loss of its partial results.
- Crash before collection = data stranded remote (mitigate: also attempt collection in emergency cleanup path).

### ALT A: MinIO/S3 sink (user-selected future enhancement — documented, not built)

**Pros**: central store during run; survives remote disk loss; dashboard reads one place; MinIO already in `src/docker-compose.yaml` with creds+bucket init; parquet long/tidy schema designed dashboard-compatible → natural fit as `sink: minio` extension of `AsyncStorageWriter`.
**Cons**: new writer code path + retry/backoff logic on flaky LAN; per-batch upload latency (background thread absorbs, but queue-blocking semantics on outage need thought — current queue blocks rather than drops, network outage would stall writer thread and eventually sim thread); credentials management in YAML; more moving parts to debug.
**Path**: `src/utils/async_storage.py` + `memory_config.sink: minio`. Implement AFTER v1 proves distribution works.

### ALT B: NFS shared results dir

**Pros**: zero code.
**Cons**: same infra coupling as §2 ALT C; concurrent writers to same tree over NFS = corruption risk with parquet writer file handles.
**Verdict**: rejected as design; again works by accident if user mounts it.

### ALT C: stream results through Redis (already reachable by all)

**Pros**: no new service; live central data.
**Cons**: Redis = RAM store — timeseries of long runs blow memory; would reinvent MinIO poorly; RedisJSON not columnar → dashboard/parquet mismatch.
**Verdict**: rejected. Redis stays config/coordination plane, not data plane.

## 4. Declarative Granularity (`host:` placement)

### CHOSEN: per-federate `host:` + scenario-level `deployment:` block

**Pros**
- Finest useful control: split heavy federate out of federation, co-locate chatty pairs.
- `deployment` block = single place machines defined; federates reference by alias → no IP repetition, no hardcoding.
- Absent key = local → old YAMLs valid untouched, zero migration.
- Matches existing per-federate config philosophy (`parallel_execution`, `streaming` all live per-federate).

**Cons**
- Many-federate scenarios need `host:` repeated per federate (verbose).
- User CAN create pathological placements (chatty pair split across slow link) — framework won't warn. Doc guidance instead.

### ALT A: per-federation host

**Pros**: fewer keys; federation = natural co-simulation unit; broker co-location trivial later.
**Cons**: cannot split federates within federation — exactly the case that matters (one heavy building model vs light grid models in same federation); forces federation restructuring for placement reasons = config semantics polluted by deployment concerns.

### ALT B: both levels (federation default + federate override)

**Pros**: terse for bulk placement, precise when needed. Genuinely nice UX.
**Cons**: two-level precedence logic + validation matrix; more doc surface; v1 doesn't need it.
**Verdict**: best candidate for later ergonomics patch — additive, non-breaking (add `host:` to federation = default for its federates). Deliberately deferred, not rejected.

### ALT C: separate deployment file (scenario YAML untouched, placement in second file)

**Pros**: same scenario runnable local or distributed by swapping deployment file; clean separation logical-vs-physical (classic HPC pattern).
**Cons**: two files to keep consistent; alias/name drift between files; CLI/API needs second path argument; against repo's single-YAML-scenario convention.
**Verdict**: rejected for consistency with existing single-file declarative approach. Note: `deployment:` block being top-level+optional gets 80% of benefit — copy scenario, delete block, it's local.

## 5. Broker Placement (implicit decision in plan)

### CHOSEN: all brokers (hierarchy + federation) stay on manager

**Pros**
- Local port allocation (`_get_n_available_tcp_ports` bind-check) stays valid — remote port discovery is genuinely annoying (race between check and bind, needs remote helper).
- Broker log capture (`_start_broker_log_reader` stdout drain) unchanged.
- Single machine to firewall (broker range 20000-30000 inbound on manager only).
- Star topology trivially debuggable.

**Cons**
- Every HELICS message crosses to manager even between two federates co-located on same remote machine → per-tick latency = 2×LAN RTT worst case. For tick-heavy tight-coupled federates this is THE performance ceiling of v1.
- Manager = single point of failure + bandwidth funnel.

### ALT: federation brokers co-located with their federates

**Pros**: co-located federates talk loopback; only inter-federation traffic crosses LAN via hierarchy broker → best possible latency.
**Cons**: remote port allocation problem; remote broker lifecycle (start/monitor/log/kill) needs RemoteExecutor treatment same as federates; multi-machine federation still needs choosing "which machine hosts broker"; more failure modes.
**Verdict**: correct v2 optimization, listed in plan future work. Measure first: if benchmark shows broker-hop dominates, do it; if model `step()` dominates (current benchmarks suggest compute-bound scenarios), skip forever.

## 6. Config Distribution to Remote Federates

### CHOSEN (inherited, no change): Redis fetch by key

**Pros**: already built, already the pattern (`federate_launcher.py` fetches by `--redis-url`/`--redis-key`); remote = same code path with different URL; single source of truth; RedisJSON path queries fetch only federate's slice.
**Cons**: Redis must be LAN-exposed (0.0.0.0:6379, unauthenticated redis-stack) → real security surface on untrusted networks. Doc must state LAN-trusted assumption loudly. Adding `requirepass` = small hardening task, touches RedisClient + compose + launcher URL format (`redis://:pass@host`) — cheap, worth doing when needed.

### ALT: push config file to remote via rsync, launcher reads file

**Pros**: no Redis network dependency for config; works if Redis unreachable.
**Cons**: forks launcher into two config paths; loses single-source-of-truth (stale file bugs); Redis still needed anyway (OverrideRegistry, catalog) → saves nothing.
**Verdict**: rejected. Redis reachability is hard requirement regardless; preflight checks it.

## 7. Remote Process Lifecycle / Orphan Prevention

### CHOSEN: `-tt` pty tying remote lifetime to ssh child + `pkill -f <sim_id>` sweep in emergency cleanup

**Pros**: kill local ssh = remote dies (SIGHUP via pty) — cleanup logic stays local-only; sim_id unique per run → pkill pattern safe; matches existing escalation philosophy (`parallel_executor` sentinel→join→terminate→kill).
**Cons**: pty merges stdout/stderr + can mangle output (mitigated: federate logs to file, stdio only backstop); if MANAGER machine dies hard, ssh children die with it → remotes get SIGHUP → actually still cleaned. True orphan window: sshd killed but remote process ignores SIGHUP — rare, pkill sweep unreachable if manager gone. Acceptable residual risk.

### ALT: detached remote processes (nohup + pidfile) + explicit kill on teardown

**Pros**: clean stream capture to remote files; remote survives manager network blip (is that even wanted? — for co-sim NO, federation is dead without manager-hosted brokers anyway).
**Cons**: monitor loop needs remote polling (ssh exec `kill -0` per federate per interval = chatty); pidfile management; orphan risk INVERTED (forget to kill = definitely orphaned). More code, worse default.
**Verdict**: rejected. Broker-on-manager makes remote-survives-manager pointless.

## 8. Cross-Cutting: When to Revisit

| Trigger | Revisit |
| --- | --- |
| >10 machines / spawn latency matters | §1 asyncssh internals swap |
| Paper artifact / reproducibility demanded | §2 Docker image deploy |
| Long runs, remote disk risk, live central results wanted | §3 MinIO sink (user already earmarked) |
| Placement verbosity complaints | §4 federation-level default + override |
| Benchmark shows broker-hop latency dominates tick time | §5 remote federation brokers |
| Untrusted network segment | §6 Redis auth + MQTT auth + ssh tunneling of 6379 |
| RL training scale-out | §1 ALT C note: Ray for agent side only |
