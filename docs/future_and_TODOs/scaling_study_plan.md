# Scaling Study & Optimal-Configuration Framework — Final Comprehensive Plan

Status: **Part A (Phases 0–5 + 1b) EXECUTED (2026-07-24 → 07-27).**
**Phase D (data exchange) EXECUTED 2026-07-28** — local + cross-machine; see
`scripts/scaling_study/findings/phaseD_exchange.md`. **Phases E/F/G still open.**
Written 2026-07-24; rewritten as the final comprehensive plan 07-27.

> **⚠ Phase D invalidated this plan's placement premise.** §2 and Phase F below
> assume placement should *minimise cross-machine data-exchange edges*, on the
> expectation that `κ(cross_machine) ≫ κ(local)`. **Measured, that is false:**
> κ_LAN ≈ 0.8 × κ_local — a LAN edge is slightly *cheaper* than a local one,
> because HELICS's lockstep barrier hides the round-trip while moving federates
> off the manager relieves core contention. Phase F's objective must be rewritten
> around **compute/contention balance**, not edge cuts. Details + scope limits in
> the findings leaf §9.

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
- **Core type** — `zmq` / `tcp` / `zmq_ss` / `tcp_ss`; forced `*_ss` behind NAT but try the canonical core type on NAT servers to chekc if the allowing commands worked.

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
            minimize TOTAL edges + payload bytes        # comms is linear in n_edges (Phase D)
            prefer intra- over cross-FEDERATION edges   # ~30%/edge; federation assignment
```
`placement MINIMIZES cross-machine data-exchange edges` was the original fourth
constraint. **Phase D removed it:** `κ_LAN ≈ 0.8 · κ_local`, so cross-machine edges
carry no penalty to minimise — coupling cost depends on how MANY edges and how wide
their payloads are, not which machine they cross. Machine placement is a
compute/contention-balance problem; edge locality matters only at the *federation*
level. Once `comms(...)` is fitted, "best config" stays a small search, not a brute force.

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

### Phase D — Data-exchange characterization  *(the core new work)* — **EXECUTED 2026-07-28**
**Goal:** fit the missing `comms(...)` term; show how coupling degrades throughput.

> **Done.** Results: `scripts/scaling_study/findings/phaseD_exchange.md` (canonical
> summary in `findings/README.md`). Delivered: `exchange_dummy` catalog model,
> cross-wiring in `gen_scenario.py`, fitted `comms` term in `cost_model.py`
> (+ `tests/test_cost_model_comms.py`), matrices `phaseD_local{,_wide}.yaml` and
> `phaseD_cross_machine.yaml`, figures 10–14, `stress_ramp.py`.
> Headline: `comms = per_edge[distance]·n_edges + per_byte·bytes`; **total edge
> count** is the regressor (not edge placement — a narrow matrix said the opposite
> and was wrong through collinearity); κ_LAN ≈ κ_local; publish cadence is the
> cheapest lever (−90% at every-10th-tick). New blockers found: **B10** (fixed) and
> **B12** (open, blocks Phase F).

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
Depends on Phase D's `comms` term — **and its objective is now the opposite of
what this section originally specified.**

- **Optimizer:** extend `cost_model.py recommend` to emit a federate→machine map.
  ~~graph-partition objective: minimize cross-machine data-exchange edges~~ —
  **superseded.** Phase D measured `κ_LAN ≈ 0.8 · κ_local`: a cross-machine edge is
  *not* more expensive than a local one, so an edge-cut objective optimises a cost
  that does not exist. The real objective is **balance compute + contention across
  machines** (the slowest machine gates every tick, and Phase D's own controls show
  distribution helping even with no wiring at all: 272 µs distributed vs 382 µs
  local at M=16). Edge placement enters only as the mild **intra- vs
  cross-federation** preference (~30% per edge), which is a *federation-assignment*
  decision, not a machine-assignment one.
  Scope limit to respect: measured on a same-campus LAN with an idle remote and
  ≤64 edges. Re-measure before trusting it on a WAN or a saturated link.
- **Blocked on B12.** `bottlenecks.md` B12 — the **teardown stall**: the simulation
  completes but the run never returns, one side of the federation having
  force-disconnected while the other blocks forever. It triggers at **≥256
  federates locally** (plain zmq) and at **≳1 kB/tick over distributed `_ss`**, and
  it strands processes that then poison subsequent runs. Both thresholds sit
  underneath the configurations Phase F exists to explore, so it must be diagnosed
  first.
- **Stress while placing:** for each candidate placement, push `M_total`/`N_total`
  until first failure; record the failure mode (comms `[-101]`, OOM, missed
  real-time deadline). Confirm memory (RSS × federates) is the true wall.
  Harness for this exists: `scripts/scaling_study/stress_ramp.py` (geometric
  ladder, RAM/load guards, stops at first failure).
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
Design note: [`phaseH_remote_brokers_design.md`](phaseH_remote_brokers_design.md).

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

---

## 8. Execution guide for a fresh session (READ BEFORE TOUCHING ANYTHING)

This section makes the plan hand-off-safe. A new session implementing Part B MUST
follow it exactly. It defines what may run autonomously, what must hard-stop, the
exact interfaces to extend (no inventing schemas), and the "done" bar per phase.

### 8.0 Read order (orient before coding)
1. `scripts/scaling_study/findings/README.md` — canonical state (what exists, gaps).
2. `scripts/scaling_study/CONTRACT.md` — the LOCKED knob/CSV/JSON schemas. Any new
   knob MUST be added here first, matching its naming convention.
3. `scripts/scaling_study/gen_scenario.py` — current generator (self-contained
   federations, **no subscriptions**). Phase D extends this file, does not replace it.
4. This §8. Then the relevant Part-B phase in §3.
5. Repo root `CLAUDE.md` (env, ports, "ask before massive runs") + graphify rule
   (`graphify query "<q>"` before reading source).

### 8.1 Environment preamble (every run)
```bash
docker compose -f src/docker-compose.yaml up -d      # redis, mqtt, minio — must be up
conda run -n cosim_gym python ...                     # `conda activate` fails from scripts here — use `conda run`
# run from repo root ALWAYS. perf log is opt-in:
COSIM_PERF_LOG=1 conda run -n cosim_gym python scripts/scaling_study/run_bench.py ...
```
If a `catalog.yaml` model is added/changed, reload Redis catalog:
`conda run -n cosim_gym python src/models/model_catalog/catalog_loader.py`.

### 8.2 Scope boundary — autonomous vs gated
| Phase | Fresh session may run autonomously? |
|-------|-------------------------------------|
| **D** (data-exchange, LOCAL: intra-fed + cross-fed only) | **YES** — dummy models, local, tiny→moderate. This is the turnkey work. |
| **D** cross-**machine** edges | **YES** — needs machines use cloud1 cloud5 and manager + go-ahead (see 8.7). |
| **H** (broker-remote design note) | **YES** — writing a plan only, no runs. |
| **E** (idle roofline) | **YES** — needs exclusive idle window + go-ahead. |
| **F** (placement + stress) | **NO** — depends on D's comms fit + large runs + go-ahead. |
| **G** (real case study) | **NO** — needs user model/data spec + go-ahead. |

**Default: implement Phase D local + draft Phase H. STOP before anything that has go ahead, show wplain simply what you are going to perform/run and ask for permission.**

### 8.3 Phase-D cross-wiring contract (extends CONTRACT.md — add these there too)
Today every federate is self-contained (`subscribes: []`). Phase D adds a wiring
layer to `gen_scenario.py`. New knobs (canonical names — add to CONTRACT.md D1 table
AND the locked CSV column list, both, before use):

| knob | type | meaning |
|------|------|---------|
| `exchange` | "none" \| "on" | wiring off (Part-A behavior, default) vs on |
| `distance` | "intra_fed" \| "cross_fed" \| "cross_machine" | where subscribers live relative to publishers |
| `fanout` | "1to1" \| "1toN" \| "Nto1" \| "all2all" | edge pattern between publisher/subscriber federates |
| `msg_width` | int | payload vector length per published value |
| `freq` | int | subscriber consumes every `freq`-th tick (1 = every tick) |
| `causality` | "same_step" \| "next_step" | on critical path vs offset-absorbed |

Wiring rules (deterministic, no cross-federation cycles for `intra_fed`):
- Publisher key format stays catalog-derived. Subscription target strings MUST use
  the repo's exact format: `<federate>.<instance>/<pub_key>` (same federation),
  `<federation>.<federate>.<instance>/<pub_key>` (cross-federation). See CLAUDE.md
  "Subscription target format".
- `intra_fed`: wire within each federation only. `cross_fed`: wire federation f→f+1.
  `cross_machine`: only valid under a distributed placement (gated).
- `fanout` maps publisher federate(s) → subscriber federate(s) by the pattern above.
- Backward-compat: `exchange: none` (default) MUST reproduce today's exact YAML —
  verify a diff of a generated `exchange=none` scenario against the current output
  is empty before landing the change.
- `<out>.spec.json` gains the new knobs; `run_bench.py` writes them as new CSV columns.

### 8.4 Dummy exchange model spec (new catalog model)
`heavy_compute_dummy` has NO inputs — it cannot receive data, so Phase D needs a
model that actually consumes subscriptions. Add `exchange_dummy` (near-zero compute,
the comms-isolating counterpart to the compute-isolating heavy/light dummies):
- Params: `msg_width` (int, output vector length), optional `iterations` (default 0,
  reuse the busy-loop only if a compute+comms combo cell is needed).
- Declares a vector output of length `msg_width` in `catalog.yaml` outputs.
- `step()`: MUST read every subscribed input value (touch it — e.g. sum into a field
  the output depends on) so HELICS actually transfers the payload and the optimizer
  can't elide it. Publishes its `msg_width`-wide output. Otherwise no work.
- Register in `catalog.yaml`, reload Redis catalog (8.1). Add a regression scenario
  per repo convention (CLAUDE.md pre-merge suite is the living contract).

### 8.5 `comms` fit signature (extends the locked fit_params.json)
Add to `cost_model.py fit` and the fitted-params JSON (CONTRACT.md D4 block):
```json
"comms": {
  "per_edge_s":  {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
  "per_byte_s":  0.0,           // marginal cost of msg_width payload
  "fixed_per_sub_s": 0.0        // per-subscription registration/poll cost
}
```
`predict()` adds `comms_m = Σ_edges(per_edge_s[distance] + per_byte_s·bytes) +
fixed_per_sub_s·n_subs` into `T_tick` (plan §2). `fit` must recover a synthetic
ground truth exactly (same discipline used for `c`/`s`/`O_par` in Part A).

### 8.6 Per-phase acceptance criteria (self-verify; don't declare done without these)
**Phase D (autonomous target):**
1. `gen_scenario.py --exchange on --distance intra_fed --fanout 1to1 --msg-width 1
   --F 1 --N 2 --M 1 --ticks 10 --placement local` emits valid YAML whose subscriber
   federate has a non-empty `subscribes:` list with a correctly-formatted target.
2. `--exchange none` output is byte-identical to the pre-change generator (empty diff).
3. Smoke run (10 ticks, local) of an `intra_fed 1to1` scenario completes with
   `failure_mode` empty AND the subscriber observably receives non-zero data.
4. A local `matrices/phaseD_local.yaml` (distance∈{intra,cross_fed} × fanout ×
   msg_width × freq × causality, tiny) runs green via `run_bench.py`; CSV has the new columns.
5. `cost_model.py fit` on that CSV emits a `comms` block; a synthetic round-trip
   recovers the injected coefficients within ~10%.
6. One plot: throughput vs coupling per distance (extend `make_report.py`).
7. `paper_ready_sentences.md` gains a data-exchange subsection; `findings/README.md`
   updated (it is canonical — record Phase D there).

**Phase H (autonomous):** a design note appended to §3 Phase H (or a leaf it links) —
no code. Covers `broker_config.host`, inverted uplink discovery, SSH broker spawn/monitor.

### 8.7 HARD STOP gates (do NOT cross without explicit user go-ahead)
Stop, report what's ready, and ask — never proceed past any of these autonomously:
- Any **cross-machine / distributed** run (Phase D cross_machine, E, F, G).
- Any run larger than **F=2, N=4, M=4, ticks=30** on the shared manager.
- Any run when `uptime` shows high load or `free -g` shows the 40%-free-RAM budget
  would be exceeded — check both immediately before scaling up.
- Phase G at all — it requires the user's model/data/topology spec first (§7).
- Config B — machine params are `TBD_*` placeholders in `gen_scenario.py`; do not
  invent them.

### 8.8 Anti-digression rules
- **Do not** create parallel docs/harness. Extend the existing files (§4.1 map). The
  ONLY canonical result doc is `findings/README.md`; the ONLY plan is this file.
- **Do not** relax `CONTRACT.md` schemas silently — extend them explicitly, in-place.
- **`run_bench.py` APPENDS** to its CSV — delete the target CSV before any rerun or a
  fit will mix work levels (this exact bug already bit Phase 1b).
- **Do not** change behavior when `COSIM_PERF_LOG` is unset (zero-overhead rule).
- Match existing code style; add a regression scenario for every new feature.
- Report back: files changed, how smoke-tested, any CONTRACT deviation — then STOP at
  the first §8.7 gate and hand back to the user.
