# Scaling Study & Optimal-Configuration Framework — Plan

Status: PROPOSED (awaiting execution go-ahead). Written fresh 2026-07-24.

## Goal

Answer three coupled questions for CosimGym at large scale:

1. **Optimize** — given a set of machines (each with its own cores/RAM), a set of
   models to install, and a target scenario, what is the *best* configuration
   across the three scaling axes?
2. **Stress-test** — what is the *maximum* number of model instances this repo +
   its features can run on a given machine set, and what fails first?
3. **Theory** — is there a predictive framework (a fitted cost model) that answers
   "best settings for scenario X on machine set Y" *without* brute-forcing every
   configuration?

### The three scaling axes (from CosimGym features)

1. **Federations** — shard load across multiple HELICS federations (hierarchy
   broker auto-inserted; cross-federation via global key namespace).
2. **Federates** — more federates per federation.
3. **Model instances per federate** — `sequential` loop vs `parallel_execution`
   (persistent worker processes, `max_parallel_workers`).

Plus two orthogonal knobs that gate scale:
- **Placement** — which machine each federate runs on (`deployment` + `host:`).
- **Core type** — `zmq` / `tcp` / `zmq_ss` / `tcp_ss`; forced to `*_ss` behind NAT.

---

## 1. Theoretical framework

HELICS is lockstep → each tick is gated by the slowest broker/machine. Model:

```
T_sim  ≈ n_ticks · T_tick
T_tick =  max over machines m ( compute_m + sync_m + comms_m )
```

Per federate:
- **compute** — sequential: `M · c`  (M = instances, c = per-instance step cost).
             parallel: `ceil(M / W) · c + O_par`  (W = workers, O_par = dispatch/IPC per tick).
- **sync**    — `s(N_broker, core_type)` — per-tick HELICS cost; rises with federate count on a broker.
- **comms**  — remote federate adds LAN RTT per tick.

### Predictions the fitted model must reproduce
- Parallel beats sequential when `c > O_par / (M − M/W)` → crossover cost `c*` and optimal W.
- Distribution beats local only when compute dominates sync **and**
  `Σ cores_remote > cores_local`; roofline ceiling of the speedup = `Σ cores / cores_local`.
- Sharding N federates across F federations raises the per-broker federate ceiling
  at the cost of one extra hierarchy-broker sync layer.

### Optimization problem ("best config" defined)

```
decision vars:  F (federations), N (federates/fed), M (instances/fed),
                mode ∈ {seq, par}, W (workers), placement: federate → machine, core_type
minimize    T_sim         (or: maximize stable M_total within a real-time budget)
subject to  N_broker ≤ ceiling(core_type, network)     # the ceiling we are investigating
            Σ_m RSS ≤ RAM_m                             # per-machine memory
            placement compute-balanced ∝ cores_m        # slowest machine gates every tick
```

Once the primitives (c, s, O_par, RTT, RSS/instance, ceiling) are measured, this is
a small search — not a brute force over every scenario.

---

## 2. First-class investigation: the zmq_ss federate ceiling

Prior note in `generate_scale_benchmark.py` claims a hard ~33-federate zmq_ss
ceiling on this LAN (49 flaky, 65 fails, all dying `[-101] lost comms` after ~52s).
**Treat this as a hypothesis to verify and explain, not a fact.** Two questions:
is it real, and is it *intrinsic to zmq_ss* or an artifact (single-socket
saturation, a tunable timeout/buffer, broker config, or the NAT topology)?

Isolate with **two network topologies**, same load, same models:

- **Config A — 3 machines, NAT (forces zmq_ss).** manager (ipazia, 112c) + cloud1
  (32c) + cloud5 (32c). Sweep federate count through and past the claimed ceiling
  (17 → 33 → 49 → 65 → 89). Record exact failure mode, timing, and whether tuning
  HELICS/zmq knobs (timeouts, socket/buffer limits, broker `--maxfeds`, sub-broker
  fan-out) moves the ceiling.
- **Config B — 2 machines, NON-NAT (plain `zmq` / `tcp`).** manager + one second
  machine reachable directly (no NAT). Same federate sweep. Does the ceiling
  vanish or shift? This is the control that tells us if the limit is `zmq_ss`-
  specific or a deeper federate/sync limit.

Deliverable of this sub-study: a clear statement of the real ceiling per
`(core_type, network)`, its root cause, and any tuning that raises it — feeding the
`ceiling(...)` constraint in the optimization problem above.

---

## 3. Measurement plan (phased; each phase feeds the model)

Every run: **3 repeats**, report median + spread. Metrics captured each run:
setup time, sim wall time, T_tick, throughput (instance-steps/s), peak RSS per
machine, CPU utilization, failure mode.

**Phase 0 — calibrate primitives** (local, cheap)
- `c` per model type: sweep `heavy_compute_dummy` work param (clean control axis)
  **and** measure real models — building, heatpump, pv, grid/pandapower, fmu → cost table.
- `s(N, core_type)`: N-sweep with a trivial model, per core_type → sync curve.
- `O_par`, worker spawn cost: parallel harness with near-zero `c`.
- LAN RTT between all machines.

**Phase 1 — Axis 3 (instances, seq vs par).** Sweep M × several c values, seq vs
par, W-sweep. Locate crossover `c*` and optimal W. Extends `benchmark_parallel_*`.

**Phase 2 — Axis 2 (federates) + ceiling study.** Sweep N on one broker → sync
curve and per-`(core_type, network)` ceiling (Section 2, both topologies).

**Phase 3 — Axis 1 (federations).** Sweep F via multifed sharding → hierarchy-broker
overhead; confirm the ceiling is per-broker, not per-machine (raise total feds by
adding federations).

**Phase 4 — placement / distribution.** Fixed load, local vs distributed,
core-proportional split, heavy vs light model → confirm speedup only when
compute-bound; measure distribution overhead when sync-bound.

**Phase 5 — max-instances stress.** For each machine-set config, push M_total until
failure. Record max stable instance count + first failure mode (comms `[-101]`,
OOM, missed real-time deadline).

**Phase 6 — validate the framework.** Predict the optimal config for a held-out
scenario+machine set, run it, compare predicted vs measured T_sim and max instances.

---

## 4. Deliverables to build

1. **Parametric scenario generator** — one spec → any `(F, N, M, mode, W, placement,
   core_type)`, with a mix of dummy and real models. Generalize
   `src/scenarios/generate_scale_benchmark.py` across all three axes + both network
   topologies.
2. **Bench driver** — runs the matrix in isolated subprocesses, collects metrics →
   CSV/parquet. Adds `psutil` per-process / per-machine resource sampling.
3. **Structured perf log** — light instrumentation in `ScenarioManager` /
   `BaseFederate` (scattered `time.time` already present) emitting per-tick + setup
   timings to a machine-readable file.
4. **Cost-model fitter + recommender** — fit c / s / O_par / RTT / RSS / ceiling from
   Phases 0–2, predict T_tick for an arbitrary config, and output a recommended
   configuration for a target scenario + machine set.
5. **Report + plots** — crossover curves, sync curves, roofline, ceiling-vs-network,
   predicted-vs-measured.

---

## 5. Machine configs under study

- **Config A (NAT, zmq_ss):** manager 130.192.177.14 (ipazia, 112c) + machine_a
  130.192.238.9 (32c) + machine_b 130.192.238.13 (32c). Ceiling investigation.
- **Config B (non-NAT, zmq/tcp):** manager + one directly-reachable second machine
  (TBD — provide host/user/workdir). Ceiling control.
- **Local baseline:** manager only, for Phases 0–1 fast iteration.

---

## 6. Sequencing

Build (1)+(3) → Phase 0 → build (4) skeleton → Phase 1 (crossover) → Phase 2 +
ceiling study on both topologies → Phase 3 (federations) → Phase 4 (placement) →
Phase 5 (stress) → Phase 6 (validation) → report.

> Per repo convention (`known_issues_from_regression.md`, "ask before massive runs"):
> get explicit go-ahead before each large / long real-hardware phase.
