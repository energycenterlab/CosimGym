# Scaling Study & Optimal-Configuration Framework — Final Comprehensive Plan

Status: **Part A (Phases 0–5 + 1b) EXECUTED (2026-07-24 → 07-27).**
**Part B (data-exchange → placement → real case study) DESIGNED, awaiting
go-ahead.** Written 2026-07-24; rewritten as the final comprehensive plan 07-27.

> **Single source of truth.** Executed results, trusted primitives, figures, and
> known gaps live ONLY in `scripts/scaling_study/findings/README.md` (canonical
> index). This doc is **design + forward plan**; it does **not** duplicate result
> numbers — where a number matters, it points to README/leaf. If a number here
> ever disagrees with the findings README, README wins.

---

## 0. Goal (unchanged)

Answer three coupled questions for CosimGym at large scale:

1. **Optimize** — given a machine set (cores/RAM each), a model set, and a target
   scenario, what is the *best* configuration across the three scaling axes?
2. **Stress-test** — the *maximum* model-instance / federate count a machine set
   can run, and what fails first?
3. **Theory** — a fitted cost model that answers "best settings for scenario X on
   machine set Y" *without* brute-forcing every configuration.

### Three scaling axes (CosimGym features)

1. **Federations** — shard across HELICS federations (hierarchy broker auto-inserted; cross-fed via global key namespace).
2. **Federates** — more federates per federation.
3. **Model instances per federate** — `sequential` vs `parallel_execution` (persistent worker processes, `max_parallel_workers`).

Two orthogonal gating knobs:
- **Placement** — which machine each federate runs on (`deployment` + `host:`).
- **Core type** — `zmq` / `tcp` / `zmq_ss` / `tcp_ss`; forced `*_ss` behind NAT.

---

## 1. Part A — what is already done (condensed; numbers in README)

All Part A runs used **ZERO data exchange** (self-contained federations, no
cross-wiring) — the deliberate control that isolates compute + sync. That is also
the central limitation motivating Part B.

| Phase | What | Canonical leaf |
|-------|------|----------------|
| 0 | Calibrate primitives: `c(work)`, `s(N,core_type)`, `O_par`, RSS/instance | README + `phase01.md` |
| 1 | Axis 3 — instance **cost**-crossover (seq vs par, sweep `work`); law `(M−⌈M/W⌉)·c > O_par` | `phase01.md`, `paper_ready_sentences.md` |
| 1b | Axis 3 — instance **count**-crossover (sweep M to 1024/fed), speedup ceiling → W, parallel staircase | `phase01.md`, figs 07/08/09 |
| 2 | Axis 2 — federate sweep + zmq_ss ceiling study (**debunked**: flaky LAN, not architectural; SSH-ControlPath + port-stride bugs found + FIXED) | `phase2_ceiling.md` |
| 3 | Axis 1 — federation sharding; hierarchy-broker cost is **setup-time only**, tick flat; 256 feds local | `phase3_federations.md` |
| 4 | Placement/distribution roofline — **CONFOUNDED** by shared-host co-user load; needs idle redo | `phase4_distribution.md` |
| 5 | Max-scale + framework validation; recommend() works, absolute T_sim is a **floor** (missing N×work interaction) | `phase5_validation.md` |

**Trusted primitives, max scale reached, and every known gap: see
`findings/README.md`.** Synthesis (best-config narrative) = `all_phases_synthesis.md`;
bottleneck catalog (what breaks + why) = `bottlenecks.md`.

### The gaps Part B must close
1. **No data-exchange term.** Cost model is `compute + sync` only; `comms` was
   never measured because no run exchanged data. Unknown how message coupling
   scales with fan-out / distance / size / frequency.
2. **Distribution confounded.** Phase 4 roofline invalid on a shared host.
3. **No real-model validation.** Everything ran `heavy/light_compute_dummy`.
4. **Placement is manual.** No optimizer emits a federate→machine map.

---

## 2. Theoretical framework (reused, extended with the comms term)

HELICS is lockstep → each tick gated by the slowest machine:

```
T_sim  ≈ n_ticks · T_tick
T_tick =  max over machines m ( compute_m + sync_m + comms_m )
```

Per federate:
- **compute** — seq: `M·c`; par: `⌈M/W⌉·c + O_par`.   **[FITTED, Part A]**
- **sync** — `s(N_broker, core_type) = s0 + s1·N`.        **[FITTED, Part A]**
- **comms** — LAN/IPC cost of exchanged data.              **[UNKNOWN — Part B fits it]**

### The comms term to fit (Part B, Phase D)
```
comms_m ≈ Σ_edges  msg_size · frequency · κ(distance)   +   fixed_per_sub · n_subs
```
where **distance ∈ {intra-federation, cross-federation, cross-machine}** sets the
per-byte coefficient `κ` (in-process HELICS core ≪ same-machine socket ≪ LAN RTT),
and **causality** (`same_step` vs `next_step`) sets whether a subscription adds to
the *critical path* of the tick or is absorbed by the offset.

### Optimization problem (reused; placement objective now includes data edges)
```
decision vars:  F, N, M, mode∈{seq,par}, W, placement: federate→machine, core_type
minimize    T_sim
subject to  N_broker ≤ ceiling(core_type, network)     # flaky, not hard (Phase 2)
            Σ_m RSS ≤ RAM_m                             # per-machine memory = true wall
            placement compute-balanced ∝ cores_m        # slowest machine gates tick
            placement MINIMIZES cross-machine data-exchange edges   # NEW, needs comms term
```
Once `comms(...)` is fitted, "best config" stays a small search, not a brute force.

---

## 3. Part B — the new study (per the 5 constraints)

Constraints (user, 2026-07-27):
1. **Dummy models only** — until the final real case study.
2. **Instances first, low/no exchange (DONE, Part A) → THEN data-exchange effects
   and combinations across entities.**
3. **Stress-test AND find optimal / near-optimal placement while stressing.**
4. **Final largest-feasible real-world case study** — real energy models + real
   data, topology/models **TBC by user**.
5. **Future feature (draft only):** brokers on remote machines.

Phases below are ordered by dependency. Each large/long hardware phase is gated by
"ask before massive runs" and preceded by a short smoke run.

### Phase D — Data-exchange characterization  *(the core new work)*
**Goal:** fit the missing `comms(...)` term; show how coupling degrades throughput.

**Taxonomy (the axes to sweep):**
- **Topology distance:** intra-federation · cross-federation · cross-machine.
- **Fan-out pattern:** 1→1 · 1→N (broadcast) · N→1 (aggregation) · all-to-all.
- **Message size:** scalar → vector (sweep payload width).
- **Frequency:** every tick → every k-th tick.
- **Causality:** `same_step` (on critical path) vs `next_step` (offset-absorbed).

**Harness work (reuse + extend):**
- `gen_scenario.py` currently emits **self-contained** federations (no cross-wiring).
  Extend it to wire configurable pub/sub graphs: intra-fed, cross-fed (global keys),
  and cross-machine edges, parameterized by the taxonomy above. Dummy models get a
  configurable output-vector width + subscription list.
- `run_bench.py` / perf-log already capture per-tick timing — reuse unchanged.
- `cost_model.py fit` — add `comms` as a fitted component; validate it recovers a
  synthetic ground truth (same discipline as Part A's `compute`/`sync` fits).

**Matrices:** new `matrices/phaseD_*.yaml` (distance × fanout × size × freq × causality).
Start LOCAL (intra + cross-fed, IPC-bound), then distributed (cross-machine, LAN-bound).

**Deliverables:** fitted `comms(...)`; a plot of throughput vs coupling per distance;
`paper_ready_sentences.md` section on data-exchange cost.

### Phase E — Idle-machine distribution redo (clean roofline)
Phase 4 was confounded (shared host, load ~67). Rerun local-vs-distributed on an
**exclusive/idle manager** to get a clean roofline; validate the ceiling
`Σcores / local_cores`. **Requires an exclusive-machine window (user schedules).**
Reuse Phase-4 matrices + `make_report.py` roofline plot as-is.

### Phase F — Optimal placement under stress
Depends on Phase D's `comms` term.
- **Optimizer:** extend `cost_model.py recommend` to emit a federate→machine map —
  graph-partition objective: **minimize cross-machine data-exchange edges** while
  **balancing compute ∝ cores** (slowest machine gates the tick). Cross-machine
  edges are the expensive ones per Phase D.
- **Stress while placing:** for each candidate placement, push `M_total`/`N_total`
  until first failure; record the failure mode (comms `[-101]`, OOM, missed
  real-time deadline). Confirm memory (RSS × federates) is the true wall.
- **Deliverable:** `recommend()` emits+justifies a placement; validated against a
  measured run (predicted vs measured T_sim + max stable scale).

### Phase G — Real-world case study (Phase 6 validation)
The capstone. Real energy models + real data exchange at largest feasible scale.
- **Design its data-exchange graph to mirror a Phase-D dummy graph** so the fitted
  framework's prediction can be validated end-to-end on real models.
- **Characterize real-model RSS first** — memory is the true ceiling; a real model's
  footprint ≫ dummy's 300 MB base.
- **Models / data / topology TBC by user** (energy-related, to be specified).
- Hard-gated by "ask before massive runs"; smoke-test the shortened scenario first.

### Phase H — FUTURE FEATURE (draft only): brokers on remote machines
Today ALL brokers (hierarchy + per-federation) stay on the manager; only federates
go remote and dial back. Placing a per-federation broker on its own machine keeps
intra-federation traffic **local** and sends only hierarchy traffic over the LAN —
the natural scale-out for a district sharded one-federation-per-machine.
**Refactor sketch (not scheduled):**
- broker placement becomes declarable (`broker_config.host`).
- uplink discovery inverts — a remote broker needs a reachable address; NAT/firewall
  becomes a real problem (today only the manager binds).
- `ScenarioManager` must spawn + monitor brokers over SSH (today it only spawns
  remote *federates*).
Design sketch only — deliver as a short design note, not code.

---

## 4. Deliverables (reuse existing harness; note extensions)

Existing, reuse as-is (`scripts/scaling_study/`, contract in `CONTRACT.md`):
- `gen_scenario.py` — parametric scenario generator (F/N/M/mode/W/core_type/model/work/placement).
- `run_bench.py` — matrix → isolated subprocess run → psutil + perf.json → append `bench.csv`.
  **⚠ APPENDS to CSV — delete the target CSV before any rerun or fits mix work levels.**
- `cost_model.py` — `fit | predict | recommend`.
- `make_report.py` — crossover / sync / roofline / ceiling / predicted-vs-measured plots.
- perf-log instrumentation (`COSIM_PERF_LOG=1`) in `BaseFederate` + `ScenarioManager`.

Extensions Part B needs:
1. `gen_scenario.py` — **cross-wiring** (data-exchange graphs by the Phase-D taxonomy).
2. `cost_model.py` — **`comms` fitted term** + **placement optimizer** (graph-partition).
3. New `matrices/phaseD_*.yaml`, `phaseF_*.yaml`.
4. New plots: throughput-vs-coupling, placement map.

### 4.1 Harness file map (this plan ↔ `scripts/scaling_study/`)

This plan is design/forward; the folder below is the implementation + results.
Executed numbers are canonical in the findings README, NOT here.

| This plan | File | Role |
|-----------|------|------|
| §2 framework | [`cost_model.py`](../../scripts/scaling_study/cost_model.py) | `fit\|predict\|recommend` — implements `T_tick=compute+sync+comms` + optimizer |
| §4 D1 generator | [`gen_scenario.py`](../../scripts/scaling_study/gen_scenario.py) | parametric scenario gen (F/N/M/mode/W/core/work/placement); **Part B: add cross-wiring** |
| §4 D2 bench | [`run_bench.py`](../../scripts/scaling_study/run_bench.py) | matrix → isolated run → append `bench.csv` (⚠ appends — delete CSV before rerun) |
| §4 D5 plots | [`make_report.py`](../../scripts/scaling_study/make_report.py), [`plot_crossover_clean.py`](../../scripts/scaling_study/plot_crossover_clean.py), [`plot_instance_crossover.py`](../../scripts/scaling_study/plot_instance_crossover.py) | report + fig 01 + figs 07/08/09 |
| §4 CSV schema | [`CONTRACT.md`](../../scripts/scaling_study/CONTRACT.md) | locked interface/CSV schema |
| §3 run specs | [`matrices/`](../../scripts/scaling_study/matrices/) | matrices (`phase1b_*` exist; `phaseD_*`/`phaseF_*` = Part-B TODO) |
| §5 machine set | [`machines.example.json`](../../scripts/scaling_study/machines.example.json) | Config A/B machine defs |
| §1 Part A results | [`findings/`](../../scripts/scaling_study/findings/) | executed output (see below) |

Inside [`findings/`](../../scripts/scaling_study/findings/):

| File | Role |
|------|------|
| [`README.md`](../../scripts/scaling_study/findings/README.md) | **CANONICAL** — done/gaps/primitives/max-scale. If a number here disagrees, this wins |
| [`phase01.md`](../../scripts/scaling_study/findings/phase01.md) · [`phase2_ceiling.md`](../../scripts/scaling_study/findings/phase2_ceiling.md) · [`phase3_federations.md`](../../scripts/scaling_study/findings/phase3_federations.md) · [`phase4_distribution.md`](../../scripts/scaling_study/findings/phase4_distribution.md) · [`phase5_validation.md`](../../scripts/scaling_study/findings/phase5_validation.md) | per-phase leaves |
| [`all_phases_synthesis.md`](../../scripts/scaling_study/findings/all_phases_synthesis.md) · [`bottlenecks.md`](../../scripts/scaling_study/findings/bottlenecks.md) · [`paper_ready_sentences.md`](../../scripts/scaling_study/findings/paper_ready_sentences.md) | best-config narrative · what-breaks-and-why · paper text |
| `*.csv` · `*.png` (figs 01–09) · `*fit_params.json` | raw bench data · figures · fitted primitives |

**Part B = edits to these same files** (extend `gen_scenario.py`, `cost_model.py`;
add `matrices/phaseD_*.yaml`) — no parallel harness.

---

## 5. Machine configs under study (reused)

- **Config A (NAT, zmq_ss):** manager 130.192.177.14 (112c) + cloud1 `machine_a`
  130.192.238.9 (32c) + cloud5 `machine_b` 130.192.238.13 (32c). Distributed / comms-LAN.
- **Config B (non-NAT, zmq/tcp):** manager + one directly-reachable second machine —
  **still TBD (host/user/workdir/python).** Isolates NAT vs multi-machine.
- **Local baseline:** manager only, fast iteration for Phase D intra/cross-fed.
- **Exclusive window:** Phases E/F/G need the manager **idle** (shared box wrecks
  distribution/stress timing — Phase 4 lesson). User schedules a co-user-free window.

---

## 6. Sequencing & gates

```
Part A: 0 → 1 → 1b → 2 → 3 → 4(confounded) → 5      [DONE]
Part B: D (data-exchange) → E (idle roofline) → F (placement+stress) → G (real case study)
        H (broker-remote draft) — anytime, no hardware
```

Gates (repo convention `known_issues_from_regression.md` + "ask before massive runs"):
- Explicit go-ahead before each large/long hardware phase (E, F, G).
- Smoke-test shortened scenario before any full run.
- `uptime` / `free -g` immediately before scaling up; abort if co-user load is high.
- Delete target CSV before any `run_bench.py` rerun.

---

## 7. Open decisions needed from the user

1. **Config B** second machine details (host / user / workdir / python) — blocks the
   NAT-vs-multimachine control.
2. **Real case-study spec (Phase G):** which energy models, what data, what topology
   / scale — needed to design G's graph to mirror a Phase-D dummy graph.
3. **Exclusive-machine window** for Phases E/F/G (clean distribution + stress).
