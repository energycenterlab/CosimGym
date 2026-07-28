# Phase D — Data-exchange characterisation (the `comms` term)

Closes gap #1 of `findings/README.md`: every Part-A sweep ran with **zero data
exchange**, so the cost framework `T_tick = compute + sync + comms` had no fitted
`comms` term at all. This phase measures it.

- **Executed:** 2026-07-28, manager (112c) idle, plus the Config-A NAT rig.
- **Data:** `phaseD_local.csv` (27 cells × 5 rep, gate-limited), 
  `phaseD_local_wide.csv` (59 cells × 3 rep, N≤32 / M≤64 / 300 ticks),
  `phaseD_cross_machine.csv` (two-arm LAN study).
- **Figures:** `10_exchange_cost_vs_load.png` … `14_causality.png`.
- **Fitted params:** `phaseD_wide_fit_params.json`.
- **Model:** `exchange_dummy` — near-zero compute, consumes every subscribed
  input, publishes an `msg_width`-wide vector. The comms-isolating counterpart of
  `heavy_compute_dummy`.

Every number below is a **paired delta**: a wired run minus the mean of
`exchange: none` control runs sharing all Part-A knobs. Absolute tick times are
never compared across cells.

> **New to this study?** `../EXPERIMENTS.md` narrates the whole campaign in plain
> language (every term defined, every file cross-referenced). `../RUNBOOK.md` has
> the commands to reproduce or modify any of it.

---

## 1. Headline law

```
comms_s  ≈  per_edge_s[distance] · n_edges  +  per_byte_s · (8 · msg_width · n_edges / freq)
```
Fitted over the wide matrix (141 wired rows, weighted least squares minimising
*relative* error since deltas span 50 µs → 22 ms): `per_edge_s` = **3.14 µs**
intra-federation / **4.36 µs** cross-federation, `per_byte_s` = **1.66 ns/byte**.
Weighted R² = 0.82, median relative error **31 %**.

> **A per-federate term was tried and does not survive.** Adding
> `in_per_link_s · max_fed_in` fits cleanly on the payload-free subset (+2.1 µs
> /link) but goes **negative** — and clamps to zero — once the payload cells are
> included. The concentration effect it was meant to capture is real and
> measured (§3, 3.1× at equal edge count) but is **not** an additive term. Left
> out of the shipped model rather than shipped as a term the data does not
> support. A physically-motivated simplification did survive: `in_per_link_s` is
> pooled across distance rather than split per distance, since `distance`
> describes an edge's *routing path* while per-federate polling is local CPU work
> that cannot know where a value came from — and the per-distance split was
> unidentifiable anyway (within-stratum `n_edges`/`max_fed_in` correlation
> ≈ 0.4–0.5 produced a 9.5×-inflated intra coefficient and a negative cross one).

Per-edge cost by distance, on one machine (fig. 10, relative-weighted fit
through the origin, edge counts 8 → 4096 — figure fits the payload-free slice,
hence marginally higher than the all-rows fit above):

| distance | µs per edge | note |
|---|---|---|
| intra-federation | **3.66** | traffic stays inside one federation broker |
| cross-federation | **4.90** | +34 % — routed via the hierarchy broker |

Coupling is **not** a rounding error. At 4096 edges the coupling term is
**22.3 ms/tick** against a 211 µs unwired baseline — a **>100×** inflation of
per-tick cost. Data exchange, not compute, is what makes a large federation slow.

## 2. Total edge count is the regressor — not edge placement

This is the phase's main methodological result, and it **reversed mid-study**.

A first, gate-limited matrix (27 cells, N pinned at 4) appeared to show that
*where* edges attach dominates: two cells with identical `n_edges = 16` and
identical `n_subs = 4` cost +87 µs (spread over 4 subscriber federates) vs
+157 µs (piled on 1). That reading gave a per-federate model R² = 0.73 against
0.53 for a totals model, and the cost model was rewritten around it.

The wide matrix refuted it. With N pinned at 4, `n_edges` and `max_fed_in` are
collinear, so the narrow fit could not tell them apart. Varying N **and** M
independently separates them, decisively:

| cell (cross_fed, N=16, M=4) | `max_fed_in` | `n_edges` | Δ tick |
|---|---|---|---|
| `Nto1` | 64 | 64 | **+234 µs** |
| `all2all` | 64 | 1024 | **+5383 µs** |

Same per-federate load, **23× the cost**, tracking edge count. On the wide data
the per-federate-only model scores **R² = 0.22** where the totals model scores
**0.97** — the exact opposite of the narrow-matrix verdict.

> **Lesson worth stating in the paper.** A cost model fitted on a narrow DoE can
> invert a causal claim. The collinearity was invisible in the fit statistics —
> only a design that moved N and M independently exposed it. This is the same
> hazard `README.md` already records for the pooled Part-A fit, hit again from a
> different direction.

Fan-out *shape* does still matter, but weakly: at a fixed 16 edges the four
patterns span 215–324 µs (≈ ±25 %), an order of magnitude less than the spread
edge count produces.

## 3. The residual: concentration (open gap)

Edge count alone does not explain everything. Holding `n_edges` fixed and moving
load between instances and federates changes cost up to 3×:

| intra_fed, equal `n_edges` | topology | `max_fed_in` | Δ tick |
|---|---|---|---|
| 512 edges | N=4, M=64 (concentrated) | 128 | **3541 µs** |
| 512 edges | N=16, M=4 (spread) | 32 | **1124 µs** |
| 128 edges | N=4, M=16 | 32 | 1023 µs |
| 128 edges | N=8, M=4 | 16 | 401 µs |

An additive `+2.1 µs · max_fed_in` term captures part of this on the payload-free
subset but does not survive the full dataset (it fits negative and clamps to
zero, §1), so the shipped model omits it and carries the effect as residual —
median relative error ≈31 %. **Open gap**, directly analogous to the documented
N×work interaction gap: the additive form has no *interaction* term between edge
count and edge concentration, and the effect is evidently multiplicative rather
than additive. Fixing it needs a matrix that crosses `n_edges` × `max_fed_in`
systematically (here they were only crossed incidentally, at 4 of 34 cells).

**Practical reading:** for a fixed amount of coupling, spread it over *more
federates with fewer instances each* rather than few fat federates.

## 4. Payload width — the dominant lever at scale

Δ tick vs `msg_width` (cross_fed, N=4, M=4, 64 edges; fig. 12):

| `msg_width` | bytes/msg | Δ tick |
|---|---|---|
| 1 | 8 | 263 µs |
| 16 | 128 | 357 µs |
| 64 | 512 | 372 µs |
| 256 | 2 048 | 529 µs |
| 1 024 | 8 192 | 1 423 µs |
| 4 096 | 32 768 | 3 539 µs |

Payload is **free up to ~512 B/message** (+41 % from 8 B to 512 B) and then goes
linear in bytes. Fitted marginal cost ≈ **5.6 ns/byte** (≈180 MB/s effective, a
Python-serialisation-bound figure, not a network one — these are local runs).
Practical rule: bundling state into wider vectors is nearly free until ~64
doubles per message; past that, payload dominates the whole comms term.

## 5. Publish cadence — the cheapest mitigation available

Δ tick vs `freq` (publisher emits every k-th tick; cross_fed, 64 edges; fig. 13):

| `freq` | Δ tick | reduction |
|---|---|---|
| 1 | 263 µs | — |
| 2 | 162 µs | 39 % |
| 5 | 94 µs | 64 % |
| 10 | 26 µs | **90 %** |
| 30 | 39 µs | 85 % (at the noise floor) |

Cost tracks **messages actually published**, not registered subscriptions — the
input handles still exist and are still polled every tick, yet cost falls ~linearly
with cadence. For any quantity that does not need tick-resolution coupling
(weather, prices, slow thermal state), decimating the publish rate is by far the
best return per unit of modelling effort.

## 6. Causality — no consistent effect (hypothesis not supported)

The plan predicted `next_step` would be cheaper by letting the auto-offset absorb
the edge off the tick's critical path. It does not, consistently (fig. 14):

| topology | `same_step` | `next_step` | Δ |
|---|---|---|---|
| cross_fed all2all, M=4 | 263 µs | 346 µs | **+83 µs** (worse) |
| cross_fed all2all, M=32 | 2089 µs | 1944 µs | −145 µs (better) |

Direction flips with load and the differences are within ~30 %. Mechanism: with
`exchange_dummy`'s near-zero compute there is nothing to overlap, while
`next_step` adds `_deferred_inputs` bookkeeping every tick. **The critical-path
benefit should only appear when per-instance compute is large enough to overlap
with the deferred read** — untested, since this phase deliberately used a
zero-compute model. Concrete follow-up: repeat the causality cells with
`exchange_dummy`'s `iterations` raised to a heavy-compute level.

## 7. Structural limits found (not performance — hard walls)

1. **`same_step` dependency cycles are forbidden.**
   `ScenarioManager._validate_causality_cycles()` raises `RuntimeError` on any
   `same_step` strongly-connected component (non-iterative HELICS time requests
   cannot resolve one). A fully-coupled all-to-all topology is therefore
   *unrepresentable* without marking at least one edge `next_step`. All Phase-D
   wiring is bipartite by construction to keep `causality` an independent knob.
2. **`multi_input_handling: sum` collapses vector payloads to a scalar.**
   A 4-way fan-in of `[0,1,2,3]` arrives as `[24.0]`, not `[0,4,8,12]` — HELICS
   sums across *elements*, not element-wise. Aggregation cannot preserve
   per-source payloads without `vectorize`. Wire traffic is unaffected, so the
   measurements stand, but it is a modelling trap worth documenting for users.
3. **Per-federation port blocks must be sized from N** — see `bottlenecks.md`
   **B10**. With plain `zmq`/`tcp`, every federate core binds its own inbound
   listener at `broker_port + 10 + n`, so a federation occupies `10 + N` ports,
   not a fixed 10. Every `N ≥ 8` cell failed (including unwired controls) until
   the generator's stride was resized to `N + 22`. Part A is unaffected: all its
   large-N runs used `zmq_ss`, whose cores bind no listener.

## 8. What this changes for `recommend()` / placement

- Coupling cost is dominated by a term the optimizer can compute statically from
  the scenario graph (`n_edges`), so placement search does not need to simulate.
- Cross-federation edges cost ~34 % more than intra-federation ones **locally**;
  a partitioner should therefore prefer cuts that keep dense sub-graphs inside a
  federation. Note this is a *federation-assignment* preference, not a
  machine-assignment one: there is no LAN penalty to trade it against (§9).
- Spreading equal coupling over more, smaller federates is measurably cheaper
  than concentrating it — the opposite of the "few fat federates" instinct that
  memory pressure encourages.

## 8.1 Stress to first failure — the instance axis has no wall

`stress_ramp.py --axis M`, F=2/N=4 (8 federates), `cross_fed all2all`, wiring ON,
100 ticks, geometric ladder, guards at 40% free RAM / load 112.

| M | model instances | n_edges | tick_mean | peak RSS | result |
|---|---|---|---|---|---|
| 64 | 512 | 1 024 | 4.7 ms | 1 363 MB | ok |
| 128 | 1 024 | 2 048 | 9.5 ms | 1 385 MB | ok |
| 256 | 2 048 | 4 096 | 17.5 ms | 1 439 MB | ok |
| 512 | 4 096 | 8 192 | 36.8 ms | 1 556 MB | ok |
| 1 024 | 8 192 | 16 384 | 76.5 ms | 1 847 MB | ok |
| 2 048 | 16 384 | 32 768 | 163 ms | 2 444 MB | ok |
| 4 096 | **32 768** | **65 536** | 329 ms | 3 637 MB | **ok — ladder exhausted** |

**No failure was reached.** The ladder ran out of rungs, not headroom: at the top
the host still had 97% RAM free. This is **20× Part A's previous maximum** (1 600
instances) and it is achieved *with data exchange on*, which Part A never had.

Two things this settles:
- **Instance count is not a scaling wall.** Cost is exactly linear —
  `tick_mean ≈ 0.080 ms · M` across six doublings (R² > 0.999 by inspection: 4.7,
  9.5, 17.5, 36.8, 76.5, 163, 329 ms) — and memory grows ~sub-linearly (1.36 → 3.64
  GB for a 64× increase in instances), because instances share one federate process.
- **The "300 MB/federate" rule is about federates, not instances.** 8 federates
  held 32 768 instances in 3.6 GB. For memory-bound planning, consolidate instances
  into fewer federates (raise M, not N) — which is exactly what B4 already advised
  for contention, and Phase D's concentration residual (§3) argues *against* for
  coupling cost. **That tension is the real placement trade-off:** raising M is
  cheap for memory and process contention, but concentrated coupling costs ~3×
  more per edge than the same edges spread across more federates.

Predictably, the run is comms-dominated at the top: 65 536 edges × ~4.4 µs/edge
≈ 288 ms of the measured 329 ms tick, i.e. **~88% of per-tick time is data
exchange**, not compute — the clearest possible statement of §1's point.

## 8.2 Stress to first failure — the federate axis DOES have a wall

Same driver, `--axis N`, F=2, M=4, `cross_fed all2all`, wiring ON, 100 ticks.
Note `n_edges = M·N²` for this topology, so edge count grows **quadratically** in N.

| N | federates | n_edges | tick_mean | peak RSS | result |
|---|---|---|---|---|---|
| 8 | 16 | 256 | 1.27 ms | 2.4 GB | ok |
| 16 | 32 | 1 024 | 6.10 ms | 4.7 GB | ok |
| 32 | 64 | 4 096 | 24.0 ms | 9.1 GB | ok |
| 64 | 128 | 16 384 | 101 ms | 18.3 GB | ok |
| 128 | **256** | 65 536 | — | — | **STALL** |

**Max stable: 128 federates** (F=2, N=64) under this coupling.

Two clean laws in the passing rungs:
- **Tick time is quadratic in N** — 1.27 → 6.1 → 24 → 101 ms, i.e. ×4 per doubling.
  That is not a federate-count effect: it is §1's linear-in-`n_edges` law showing
  through, because all-to-all coupling makes edges grow as N². **The federate axis
  is expensive only because of the topology it implies**, which is the single most
  actionable planning consequence of this phase.
- **Memory is linear in N** — 2.4 → 4.7 → 9.1 → 18.3 GB, ≈143 MB/federate. Contrast
  §8.1, where 64× more *instances* added only 2.3 GB. Federates cost memory;
  instances are nearly free.

### What actually broke — and it is not what the plan expected

Not OOM (96% RAM free), not lost comms, not a missed deadline. **The simulation
finished and the run refused to end.** Exactly 128 of the 256 federates reached
`t=100` and logged `disconnect Timer expired forcing disconnect`; the other
federation never completed, and the manager waited until the harness timeout.

This is the same signature as the distributed stalls in §9.1 — and it is **plain
zmq, local, no distribution**, which retires the earlier reading of that failure as
distribution- or `_ss`-specific. It is a **teardown/lifecycle** bug with (at least)
two triggers — high federate count, and high exchanged bytes per tick over `_ss` —
promoted to `bottlenecks.md` **B12** as the study's most serious open limit. It
also strands processes that poison later runs (19 spurious failures before this was
diagnosed), so `run_bench.py` now reaps orphaned brokers after a timeout.

**Answering the original question — "the real maximum number of instances".** On
this host, with data exchange on: **32 768 instances** (8 federates × 4 096) with no
failure and room to spare, but only **128 federates**, and the binding constraint on
the federate axis is a teardown bug rather than any resource. The two limits are
independent, and the instance limit was never reached.

## 9. Cross-machine (κ_LAN) — **no LAN penalty per edge**

Two-arm design (`matrices/phaseD_cross_machine.yaml`): every distributed cell has
a **local `zmq_ss` twin** — same core type, same wiring, no network — so the LAN
hop is not confounded with the `zmq → zmq_ss` core-type change. Config A rig,
manager (112c) + machine_a (32c, idle, same-campus LAN). Publisher side pinned to
machine_a, subscriber side to the manager, so every wired link is a real LAN hop.

| arm (all2all, width 1, every tick, 3 repeats) | M=1 (16 edges) | M=4 (64 edges) |
|---|---|---|
| local `cross_fed`, zmq_ss | 6.53 µs/edge | 6.97 µs/edge |
| **cross-machine, zmq_ss** | **6.46 µs/edge** | **4.60 µs/edge** |
| ratio κ_LAN / κ_local | **0.99** | **0.66** |

**κ_LAN ≈ κ_local — a LAN edge costs no more than a local one** (and at M=4,
measurably less).

Mechanism: HELICS is lockstep, so every tick already pays a barrier; the LAN
round-trip hides inside a barrier that has to happen anyway, while moving the
publisher side onto a second machine removes its federates from contention for
the manager's cores. The network cost is real but it is *overlapped*.

**Distribution itself is cost-NEUTRAL, not beneficial.** The unwired controls are
within noise of each other — 77 vs 79 µs (M=1), 131 vs 133 (M=4), 339 vs 327
(M=16), distributed vs local. (An earlier reading of this as "distribution helps"
came from the contaminated first attempt at this matrix — see §9.1 — and does not
survive clean data. The benefit is in what distribution *doesn't* cost, i.e. it
buys extra machines' capacity for free, not in making the same work faster.)

**This overturns the plan's placement premise.** `scaling_study_plan.md` §2 assumes
placement should "MINIMIZE cross-machine data-exchange edges" on the expectation
that `κ(cross_machine) ≫ κ(local)`. Measured, that expectation is wrong at this
scale on this LAN: a placement optimizer should balance **compute and contention**
across machines and treat cross-machine edges as ~free, rather than treating edge
cuts as the objective. (Scope: same-campus LAN, ≤64 edges, idle remote. A WAN
link, or a saturated one, is untested and would plausibly behave differently.)

### 9.1 The real LAN limit: distributed `zmq_ss` stalls above ~1 kB/tick

The LAN penalty is not in per-edge latency — it is a **capacity wall**, and it is
the most serious limit this phase found. Two symptoms that first looked like
separate bugs (a hang at M=16 with scalar payloads, and a hang at wide payloads
with M=4) are **one failure ordered by exchanged bytes per tick**:

| config (N=4, `zmq_ss`, distributed) | edges | bytes/tick | result |
|---|---|---|---|
| M=1, width 1 | 16 | 128 B | ok |
| M=4, width 1 | 64 | 512 B | ok |
| M=16, width 1 | 256 | 2 kB | **hang ×3** |
| M=4, width 64 | 64 | 32 kB | **hang ×3** |
| M=4, width 1024 | 64 | 512 kB | **hang ×3** |

Deterministic (3/3 at every failing point), so not a flake. It is **not bandwidth**
— 2 kB/tick is nothing. The remote side always finishes correctly (all remote
federates run their full 300 ticks and exit 0, one logging `disconnect Timer
expired forcing disconnect`); it is the manager that never completes.

It is **`_ss`-specific**: plain `zmq` locally carried 64 edges × 32 kB/msg without
trouble (`phaseD_local_wide.csv`). Local `zmq_ss` survives further than distributed
`zmq_ss` but still breaks — `width 1024` locally dies with
`HelicsException: [-101] lost connection with server`.

**Why this matters more than the κ result above.** `*_ss` is not optional on the
NAT rig — single-socket cores are *required* to reach machines behind NAT. So the
core type distribution depends on is the one that cannot carry payload. κ_LAN ≈
κ_local is a real and useful result, but it holds only in the narrow band where
distributed `zmq_ss` works at all (≲1 kB/tick per federation). Any distributed
district exchanging profiles, forecasts or state vectors hits this immediately.

Undiagnosed; **blocks Phase F**. Full mechanism notes, suspects and interim
workaround in `bottlenecks.md` **B12**.
