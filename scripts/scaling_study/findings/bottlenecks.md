# CosimGym Scaling — Bottleneck Catalog (what limits scale, and WHY)

Every performance limiter found across Phases 0–5, each with its **mechanism**
(the physical/architectural reason it appears), the **evidence** (phase + numbers),
**when it dominates**, and **how to push it back**. Companion to
`all_phases_synthesis.md` (which answers "best config"); this file answers
"what breaks, and why".

A CosimGym run is always in one of four **regimes** — the whole story reduces to
*which bottleneck is currently binding*:

| Regime | Binding bottleneck | Tell-tale |
|--------|--------------------|-----------|
| **Sync-bound** | HELICS per-tick barrier `s(N)` | cheap models, `tick_mean_s` flat vs work |
| **Compute-bound** | model `step()` cost `M·c` | `tick_mean_s` scales with `work`/`M` |
| **Contention-bound** | cores/processes oversubscribed (incl. co-users) | `tick_mean_s` rises with total process count / co-user load |
| **Memory-bound** | ~300 MB RSS per federate process | RSS ≈ 300 MB × federate count; OOM is the wall |

---

## B1 — HELICS per-tick synchronization barrier `s(N)`  *(the floor)*

- **Mechanism.** HELICS is lockstep: every federate must request-and-be-granted
  the next time before *any* federate advances. Each tick pays a broker round of
  time-grant coordination — a fixed per-tick cost independent of how little the
  models compute. This is the irreducible floor of a co-simulation tick.
- **Evidence (Phase 0a, 2).** Fitted `s(N)_zmq = 1.3e-4 + 2.65e-6·N` s/tick;
  near-flat for 2≤N≤16, plateauing ~2.65e-4 s by N=32–64. Only ~2.6 µs added per
  extra federate — sync scales *well* with N; it is the *baseline*, not the
  N-scaling problem.
- **When it dominates.** Any cheap/analytic model: compute is ~µs/step while sync
  is ~0.1–0.3 ms/step, so the run measures synchronization, not physics. This is
  why `generate_scale_benchmark.py`'s analytic district was sync-bound and why
  distribution *lost* there (LAN latency adds to every tick's barrier).
- **Push it back.** Nothing to "fix" — it's the co-sim contract. To *amortize* it,
  make each tick do more useful compute (larger `real_period` steps, heavier
  models) so sync is a smaller fraction. Fewer, coarser ticks beat many fine ones.

## B2 — Parallel-worker dispatch overhead `O_par`  *(the reason par usually loses)*

- **Mechanism.** `parallel_execution` steps model instances in persistent worker
  **processes** (GIL-bound Python → processes, not threads). Every tick must
  serialize each instance's inputs, hand them across a process boundary (IPC),
  wake the workers, and collect results. That handshake is a **fixed per-tick
  tax** that exists no matter how trivial the instances are.
- **Evidence (Phase 1, 5).** `O_par ≈ 0.039 s/tick`, and par `sim_wall` stayed
  ~0.8 s **flat** as `work` swept 1→8000 (compute invisible under the fixed tax)
  while seq scaled with work. Crossover only at `work ≈ 24,000` for M=16/W=8 —
  three independent estimates agreed to ~4%, matching the law below.
- **When it dominates.** Whenever per-instance `step()` is cheap: `(M−⌈M/W⌉)·c <
  O_par` ⇒ sequential wins. For most analytic/light models, **always**.
- **Push it back.** Only enable `parallel_execution` when
  `(M − ⌈M/W⌉)·c(work) > O_par` — i.e. genuinely CPU-heavy `step()`s. Otherwise
  leave it off (it's off by default for good reason). Bigger M and heavier work
  both help cross; more workers W helps only up to the core count.

## B3 — Federation hierarchy-broker startup tax  *(one-time, not per-tick)*

- **Mechanism.** Each federation spawns its own `helics_broker` process, and with
  F>1 a hierarchy broker is inserted above them; every sub-broker must dial and
  register with it at startup. That is process-spawn + registration handshake
  cost, paid **once** at setup, per federation.
- **Evidence (Phase 3a).** `broker_setup_s ≈ -0.070 + 0.369·F` — ~**0.37 s per
  added federation** — while `tick_mean_s` stayed *flat-to-improving* at fixed
  total federate count (fewer federates per broker offsets the extra layer). So F
  costs setup, never steady-state.
- **When it dominates.** Short runs with many federations (setup is a large
  fraction of a brief sim). Irrelevant for long runs.
- **Push it back.** Don't shard "for speed" — sharding buys placement/fault-
  isolation/ceiling-headroom, not throughput. Keep F minimal unless you need those.
  (Cost-model note: the plan's §1 `T_tick` has no F term; add a one-time
  `F·0.37 s` setup term instead.)

## B4 — Host CPU / process-count contention  *(the real N-scaling wall on one box)*

- **Mechanism.** Each federate is a separate GIL-bound Python process pinned to
  ~1 core. Once the number of concurrently-stepping federate processes approaches
  or exceeds the host's core count, the OS scheduler time-slices them and every
  tick's lockstep barrier waits on whichever process was descheduled — so per-tick
  time inflates.
- **Evidence (Phase 3b).** At fixed hierarchy structure, `tick_mean_s` grew
  0.30 ms → 1.09 ms as **total** federates went 64 → 256 on the one 112-core
  manager — *regardless* of the F/N split. Phase 3a proved it isn't hierarchy
  sync (flat at fixed total), so it's host contention.
- **When it dominates.** Total federate (process) count ≳ host cores, single-machine.
- **Push it back.** Distribute federates across machines (spreads processes over
  more cores) — this is the *legitimate* reason to go distributed, distinct from
  the compute-roofline reason in B6. Or consolidate instances into fewer federates
  (raise M, not N) since instances step within one process.

## B5 — Shared-machine co-user contention  *(measurement hazard + real slowdown)*

- **Mechanism.** The manager (ipazia) is a shared box with no resource isolation.
  Other users' processes compete for the same cores/memory bus, so a CosimGym run's
  per-tick time depends on *strangers' load* at that moment.
- **Evidence (Phase 4).** Load avg ~67 from co-users (`bottaccioli`,`montaldo`)
  during runs made "local" so slow that distribution beat even the *idealized*
  1.571× roofline (ratio 0.548 at N=88) — for the wrong reason (dodging
  contention, not a compute crossover).
- **When it dominates.** Any timing comparison on the shared manager while others
  are active — it can *invert* a placement decision.
- **Push it back.** Check `uptime`/`free -g` immediately before timing; re-validate
  numbers at real deployment time; for clean roofline experiments use an idle or
  exclusive machine. On a shared box, "distribute" is a reasonable hedge against
  unpredictable neighbours — but that's an infra artifact, not a framework property.

## B6 — Distribution / LAN overhead  *(setup cost + per-tick RTT)*

- **Mechanism.** Two costs. (1) **Setup**: remote federates are spawned over SSH
  and `src/` is rsync'd to each machine before the run. (2) **Per-tick**: every
  cross-machine HELICS message traverses the LAN, adding round-trip latency to the
  lockstep barrier (B1) for any tick that crosses a machine boundary.
- **Evidence (Phase 3c, 4).** Distributed `setup_s` ran ~1.7–1.8 s above the local
  twin (SSH spawn + rsync), while `broker_setup_s` itself matched local — so the
  gap is remote-spawn, not brokering. Per-tick RTT couldn't be isolated (Phase 4:
  `rtt_s` floored at 0 because co-user contention on local swamped it) — meaning on
  this LAN, RTT is *smaller* than the sync/contention terms, not larger.
- **When it dominates.** Short distributed runs (setup tax); and sync-bound
  workloads, where LAN RTT is pure added cost with no compute benefit to offset it
  → distribution loses (the `generate_scale_benchmark.py` result).
- **Push it back.** Distribute only when B4 (host contention) or B6-compute
  (manager cores saturated by heavy compute) actually bind; keep runs long enough
  that the one-time setup tax amortizes; pre-stage code on remotes to cut rsync.

## B7 — Memory: ~300 MB RSS per federate  *(the true max-scale ceiling)*

- **Mechanism.** Every federate is a full Python process (interpreter + HELICS
  bindings + imports + model objects). That base footprint dominates; **added
  model instances inside a federate are nearly free** in memory.
- **Evidence (Phase 0d, 5).** RSS ~304–305 MB flat across M∈{1,4,16,64}
  (`rss_per_instance ≈ 0`). So **federate count**, not instance count, sets memory.
  Manager ceiling ≈ **207 federates** at a 40%-of-free-RAM budget (~61 GB).
  Max *stable* actually run: 200 federates / 1,600 instances (N8×M200), inside budget.
- **When it dominates.** The wall for max-scale: you run out of RAM (federates)
  long before CPU on a big-memory box, and instances are cheap — so to pack more
  work per GB, prefer **many instances in few federates** (raise M) over many
  federates (raise N).
- **Push it back.** Consolidate instances into fewer federates; spread federates
  across machines' aggregate RAM; a lighter federate base process would raise the
  ceiling directly.

## B8 — SSH `ControlPath` > 108-byte AF_UNIX limit  *(FIXED — was a hard cap)*

- **Mechanism.** OpenSSH ControlMaster binds an AF_UNIX socket whose path
  (`sun_path`) caps at 108 bytes; OpenSSH also appends a ~17-char random suffix
  before an atomic bind-rename. The old `ControlPath` was
  `logs/<scenario_name>/<timestamp>/ssh_control/cm-<alias>`, and the scenario name
  **embeds the federate count N** — so at N≥112 the base hit ~91 chars, +17 = 108,
  overflowing by one byte. One extra digit in N tipped it over.
- **Evidence (Phase 2, fixed+verified Phase 5).** Deterministic 6/6 failure at
  N≥112 with `unix_listener: path "…" too long for Unix domain socket`; after the
  fix (hashed `/tmp/cosim-ssh/<8hash>/cm-<alias>`, length independent of N) N=200
  distributed passes 6/6.
- **Status.** Fixed in `ScenarioManager._setup_remote_execution`. This had been
  masquerading as / conflated with the (non-existent) zmq_ss ceiling.

## B9 — `gen_scenario.py` zmq broker `port+1` collision  *(FIXED — was F≥2 blocker)*

- **Mechanism.** Plain `zmq` cores reserve **two** ports — `port` and `port+1`
  (paired reply socket). The generator spaced per-federation broker ports by 1, so
  federation_f's `port+1` collided with federation_(f+1)'s `port` → deterministic
  bind failure for any F≥2 on zmq. (`zmq_ss` reserves only one, so it dodged it —
  which is why Phase 3 could work around it by switching core type.)
- **Evidence (Phase 3, fixed+verified).** Fixed by striding federation ports by 10;
  F=2 local zmq now passes. Same class as the "zmq auto-port +1" known issue in
  `CLAUDE.md`.
- **Status.** Fixed in `gen_scenario.py`.

---

## Not a runtime bottleneck, but a modeling gap: N×work interaction

The cost model itself under-predicts by **2.3–3.3×** when a config has *both*
nontrivial N *and* nontrivial per-instance work (Phase 5 validation). **Why:** the
additive `T_tick = compute + sync` form has no interaction term, and no calibration
matrix ever varied N and work *together* — so the region is unmodeled, not
mismeasured. This bounds how much you can trust `recommend()`'s absolute `T_sim`
(read it as an optimistic floor); the *structure* (crossover law, sync curve,
regime boundaries) held up. Fix: add a joint N×work calibration sweep.

## Known I/O bottleneck not exercised this campaign (from `CLAUDE.md`)

The `parquet` sink offloads writes to a background `AsyncStorageWriter`; its queue
**blocks** (never drops) if the writer thread falls behind, so a slow disk becomes
back-pressure on the sim thread. There is also a tracked native `libstdc++`
SIGSEGV in the parquet path (`known_issues_from_regression.md`). These runs used
`sink: none`/`json`, so storage I/O was not on the critical path here — flagged for
completeness since it's the obvious next bottleneck for write-heavy long runs.

---

### One-line summary per bottleneck

| ID | Bottleneck | Per-tick or setup | Fixed? | Push-back |
|----|-----------|-------------------|--------|-----------|
| B1 | HELICS lockstep sync `s(N)` | per-tick (floor) | inherent | coarser ticks / heavier steps |
| B2 | Parallel worker dispatch `O_par` | per-tick | inherent | only par when `(M−⌈M/W⌉)c>O_par` |
| B3 | Federation broker startup | setup | inherent | minimize F |
| B4 | Host process-count contention | per-tick | inherent | distribute / raise M not N |
| B5 | Shared-machine co-users | per-tick | infra | idle/exclusive machine |
| B6 | Distribution / LAN | setup + per-tick | inherent | distribute only when B4/compute binds |
| B7 | 300 MB/federate memory | capacity | inherent | many instances, few federates |
| B8 | SSH ControlPath 108 B | hard cap @N≥112 | **FIXED** | — |
| B9 | zmq port+1 collision | hard fail @F≥2 | **FIXED** | — |
