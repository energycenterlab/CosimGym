# Paper-ready sentences — seq/par crossover & real-model mapping

Reusable prose + figure/table assets for the sequential-vs-parallel
model-instance-execution results. All numbers from the clean M=16/W=8 slice
(`01_crossover.png`, `plot_crossover_clean.py`) unless noted. Drop into the
paper and trim to fit.

> **Provenance note.** The CosimGym-measured items (crossover, `O_par`, `c`
> slope for `heavy_compute_dummy`, RC-model cost) are from this study's runs.
> The **EnergyPlus per-step figures are literature/typical-run order-of-magnitude**,
> NOT measured in CosimGym — flag them as such in the paper (or replace with a
> measured FMU marker, see §Open anchor).

---

## Methods

> To characterize the trade-off between sequential and process-parallel
> execution of a federate's model instances, we used a synthetic model
> (`heavy_compute_dummy`) whose per-step computational cost is controlled by a
> single work parameter, allowing per-instance cost `c` to be tuned
> independently of the co-simulation topology. Fixing the federation at one
> federate hosting M=16 instances with W=8 worker processes, we swept the work
> parameter across five levels spanning two orders of magnitude and measured
> mean per-tick wall time (20 ticks, 3 repetitions) under both execution modes
> on a 112-core host.

## Results — the crossover law

> Sequential execution cost scales as `M·c`, whereas parallel execution scales
> as `⌈M/W⌉·c + O_par`, where `O_par` is a fixed per-tick worker-dispatch
> overhead. The two modes cross over when the sequential-instance saving offsets
> this overhead, i.e. `(M − ⌈M/W⌉)·c = O_par`. Empirically we measured
> `O_par ≈ 0.04 s` and `c = a + b·work` with `b = 1.34×10⁻⁷ s` per work unit
> (independent sweeps agreeing to <1%). The observed crossover (work ≈ 2.3×10⁴,
> `c ≈ 3.1 ms`) matched the analytical prediction (`c* ≈ 3.3 ms`) to within ~7%
> (Fig. X). Below the crossover, parallel execution was up to 17× slower than
> sequential owing to the fixed dispatch overhead; above it, parallel execution
> was up to 2.7× faster. This shows that process-parallel instance execution
> should be enabled only for models whose per-instance step cost exceeds a
> topology-dependent threshold, and is otherwise a pessimization.

## Decision rule (one line)

> The sequential-vs-parallel decision reduces to a single inequality,
> `c > O_par / (M − ⌈M/W⌉)`; the crossover per-instance cost falls
> hyperbolically with instance count, so parallel execution is beneficial only
> for expensive models or very large instance counts.

## Headline takeaway (one line)

> Process-parallel model execution is a *pessimization* for cheap models: it
> pays only once each instance's step is expensive enough that the
> sequential-instance saving exceeds the fixed per-tick worker-dispatch
> overhead; it should therefore not be enabled by default.

## Real-model mapping (EnergyPlus reference — see provenance note)

> The measured crossover per-instance cost (≈3 ms at M=16, W=8) corresponds to a
> medium-complexity EnergyPlus building model stepped once per co-simulation
> interval. Reduced-order (RC) building models, at ~0.06 ms/step, sit ~50× below
> this threshold and never benefit from process-parallel execution at practical
> instance counts; detailed physics FMUs (EnergyPlus-class, ~1–30 ms/step) meet
> or exceed it. Because the threshold scales as `O_par/(M − ⌈M/W⌉)`, even
> lightweight models cross over at district scale (≈100 instances).

## Figure X — caption

> **Fig. X.** Mean wall time per tick vs. per-instance model work for sequential
> and parallel execution of M=16 model instances (W=8 workers) in a single
> federate (`heavy_compute_dummy`, zmq, local, 20 ticks × 3 repeats; band = min–max
> over repeats). The upper axis maps the work parameter to per-instance step cost
> `c`. The dashed line marks the measured crossover (work ≈ 2.3×10⁴, `c ≈ 3.1 ms`),
> matching the analytical prediction `(M−⌈M/W⌉)·c = O_par` (`c* ≈ 3.3 ms`) to
> within ~7%. Asset: `01_crossover.png`.

## Table Y — measured seq vs par (M=16, W=8)

| work (iter/step) | per-instance `c` (ms) | seq per-tick (ms) | par per-tick (ms) | winner |
|---:|---:|---:|---:|:--|
| 1,000 | 0.16 | 2.4 | 43.1 | seq (17×) |
| 10,000 | 1.36 | 21.9 | 47.4 | seq |
| 25,000 | 3.37 | 54.4 | 51.6 | par (barely) |
| 50,000 | 6.71 | 106.1 | 60.0 | par (1.8×) |
| 100,000 | 13.4 | 216.9 | 76.5 | par (2.7×) |

## Open anchor (to make the EnergyPlus mapping measured, not cited)

> Wrap a real `EnergyPlusToFMU` (or an existing repo FMU) as one federate,
> measure its per-step `c` with the perf harness (`COSIM_PERF_LOG=1`), and place
> a real vertical marker on the crossover figure's `c`-axis. Converts the
> mapping from plausible to measured.

---

# Vertical scalability — the instance axis (M) (Phase 1b)

The dual view of the crossover above: model cost `c` is FIXED and the number of
model instances per federate M is swept, up to district scale (M=1024, ~a
thousand buildings in one federate process). Assets: `07_instance_crossover.png`,
`08_speedup_vs_M.png`, `09_par_staircase.png` (staircase noisy pending a clean
idle-machine rerun — treat 07/08 as the headline, 09 as supplementary).
Source: `matrices/phase1b_*.yaml`, `plot_instance_crossover.py`.

## Framing (see terminology note at end of file)

> We distinguish **vertical scalability** — increasing the number of model
> instances hosted within a single federate process on one machine, executed
> either sequentially or across local worker processes (`parallel_execution`) —
> from **horizontal scalability**, the sharding of a scenario across more
> federates, federations, and physical machines (Section [horizontal]). This
> section addresses the vertical axis.

## Methods

> Holding per-instance model cost fixed at two levels (a light, RC-building-like
> cost `c ≈ 0.29 ms` and a detailed, medium-EnergyPlus-like cost `c ≈ 3.37 ms`),
> we swept the number of model instances per federate M from 2 to 1024 and
> measured mean per-tick wall time under sequential and parallel execution
> (W = 8 workers, 20 ticks, 3 repetitions, single federate, 112-core host).

## Results — instance-crossover (dual of the cost-crossover)

> Solving the crossover condition for instance count rather than cost yields the
> instance-crossover M*: the smallest M for which parallel execution overtakes
> the sequential loop, `M* − ⌈M*/W⌉ = O_par / c`. For the detailed model
> (`c ≈ 3.37 ms`) the measured M* ≈ 15 matched the analytical value (16); for the
> light model (`c ≈ 0.29 ms`) M* ≈ 212 (analytical 181). Thus a federate must
> host on the order of hundreds of light-model instances — i.e. reach
> **district scale** — before process-parallel execution is worthwhile, whereas
> detailed models cross over at a few tens of instances. All configurations up to
> **1024 instances in a single federate** ran without failure.

## Results — speedup ceiling

> Parallel speedup (sequential ÷ parallel wall time) grows monotonically with M
> and is bounded above by the worker count: as M ≫ W the ratio approaches
> `M·c / (⌈M/W⌉·c) = W`. In the tested range the ceiling (W = 8) was approached
> but not reached — at M = 1024 the light model achieved ≈ 2.6× and the detailed
> model ≈ 2.5× at M = 64 — because the fixed per-tick overhead `O_par` remains a
> non-negligible fraction of parallel cost until `⌈M/W⌉·c ≫ O_par`. More workers
> or a heavier model raise the realized speedup toward W.

## Results — the ⌈M/W⌉ staircase (supplementary)

> Parallel cost is a step function of M: `⌈M/W⌉·c + O_par` jumps by one model-step
> cost `c` each time M crosses a multiple of the worker count W (a worker takes on
> one additional instance). The steps are measurable only when `c` exceeds the
> per-tick scheduling jitter (~10 ms on the shared host); at a heavy cost
> (`c ≈ 27 ms`) the jumps at M = W, 2W, … are resolved, confirming the model
> structure. A slight upward drift within each plateau reflects the marshalling
> cost of dispatching more instances to the same workers.

## Table Z — instance-crossover (measured vs law, W = 8)

| model cost `c` | regime | measured M* | law M* | interpretation |
|---:|:--|---:|---:|:--|
| 0.29 ms (work 2,000) | light / RC-building | ≈ 212 | 181 | parallel pays only at district scale (~hundreds) |
| 3.37 ms (work 25,000) | detailed / med-EnergyPlus | ≈ 15 | 16 | parallel pays at a few tens of instances |

> Provenance: measured on CosimGym (this study); `heavy_compute_dummy` synthetic
> model, cost tuned via its `iterations` parameter. The light↔RC and
> detailed↔EnergyPlus mappings are the order-of-magnitude correspondences from
> the real-model section (cited, not measured).

## Vertical vs. horizontal — one-line framing

> Vertical scaling packs more building models into one federate on one machine
> (this section: the M / sequential-vs-parallel axis); horizontal scaling shards
> them across federates, federations, and machines (Section [horizontal]).
> A real district-scale deployment combines both: many federates, each hosting
> many instances, distributed across a machine set — with per-machine **memory**
> (≈300 MB base + per-instance model footprint) as the ultimate ceiling on how
> many instances one host can hold, distinct from the compute crossover studied
> here.

---

## Terminology note — "vertical scalability" (is it correct?)

**Yes, correct.** Standard distributed-systems usage:
- **Vertical scaling (scale-up):** handle more load on a *single node/process* by
  using more of that node's resources (cores/RAM). Here = more model instances M
  per federate, absorbed either sequentially or via local worker processes
  (`parallel_execution`, W cores of the one host). Phase 1 (cost-crossover) and
  Phase 1b (instance-crossover, speedup ceiling, staircase) are all vertical.
- **Horizontal scaling (scale-out):** handle more load by adding *more
  nodes/processes* — more federates (N), federations (F), and physical machines.
  Phases 3 (federation sharding), 4 (distribution), 2/5 (multi-machine) are
  horizontal.

Precision caveat to keep the paper defensible: increasing M under *sequential*
execution is raising **load** on fixed resources, not scaling resources; the
vertical *scaling response* is `parallel_execution` recruiting more local cores.
Phrase it as "vertical scalability: how a single federate/host absorbs a growing
model-instance count, sequentially or by recruiting local cores." Reserve
"horizontal scalability" for the federate/federation/machine sharding axis.

---

# Data-exchange coupling — the `comms` term (Phase D)

All numbers from `phaseD_local_wide.csv` (59 cells × 3 repetitions, 300 ticks,
112-core host, plain zmq, local) unless noted. Figures `10`–`14`. Every value is
a **paired delta** against an identically-configured control run with the wiring
removed, so it is the marginal cost of coupling, not a raw tick time.

## Methods

> To isolate the cost of data exchange from computation, we introduced a second
> synthetic model (`exchange_dummy`) that performs negligible arithmetic but
> consumes every subscribed input and publishes a vector payload of configurable
> width. Federations were wired bipartitely — a disjoint publisher side and
> subscriber side — which keeps every dependency graph acyclic; this is required
> because CosimGym rejects `same_step` dependency cycles outright, as
> non-iterative HELICS time requests cannot resolve them. We swept topology
> distance (intra- vs cross-federation), fan-out pattern, payload width, publish
> cadence and subscription causality, varying the federate count N and the
> instance count M independently so that scenario-wide edge count and
> per-federate link count could be identified separately. Each configuration was
> compared against an otherwise identical control run with no subscriptions, and
> the coupling cost taken as the difference in mean per-tick wall time.

## Results — the coupling law

> Per-tick coupling cost grows linearly with the total number of HELICS
> input→target links in the scenario, at 3.66 µs per link for intra-federation
> edges and 4.90 µs per link for cross-federation edges (a 34% premium for
> traffic routed through the hierarchy broker). Coupling is not a second-order
> effect: at 4096 edges the coupling term reaches 22.3 ms per tick against a
> 211 µs baseline for the same federation with its subscriptions removed — a
> hundredfold inflation of per-tick cost. Adding a payload term of 1.66 ns per
> byte routed gives
> `comms = per_edge[distance]·n_edges + per_byte·(8·msg_width·n_edges/freq)`,
> which reproduces the measurements with a median relative error of 31% across
> three orders of magnitude of coupling. We also evaluated a term proportional to
> the busiest subscriber federate's inbound link count: it is well determined on
> payload-free configurations (≈2.1 µs per link) but fits negative once
> payload-bearing configurations are included, so we report the concentration
> effect it was intended to capture as an unmodelled residual rather than as a
> fitted coefficient.

## Results — placement implication (the useful one)

> Because the dominant term depends only on the total edge count, which is known
> statically from the scenario graph, coupling cost can be predicted without
> simulation — the property that makes placement optimisation tractable. The
> secondary term rewards spreading a given amount of coupling across many small
> federates rather than concentrating it: at a fixed 512 edges, hosting the
> coupling on 8 federates of 64 instances cost 3541 µs/tick, while the same 512
> edges spread over 32 federates of 4 instances cost 1124 µs/tick — a 3.1×
> penalty for concentration at identical total coupling.

## Results — payload and cadence

> Payload width is free until roughly 512 bytes per message: widening the
> published vector from 1 to 64 doubles raised coupling cost by only 41%. Beyond
> that the cost becomes linear in bytes at ≈5.6 ns/byte, and at 4096 doubles
> (32 kB/message) payload dominates every other term. Publish cadence is the
> single most effective mitigation available: emitting on every tenth tick rather
> than every tick removed 90% of the coupling cost, because cost tracks messages
> actually transferred rather than subscriptions registered — the input handles
> remain and are still polled each tick.

## Negative result worth reporting (causality)

> We found no consistent benefit from deferring subscriptions to the following
> step. `next_step` causality was 32% more expensive than `same_step` at low
> coupling and 7% cheaper at high coupling, with the direction of the effect
> reversing across the range. We attribute the absence of the expected
> critical-path saving to the near-zero computational cost of the probe model:
> deferring a read only pays when there is computation to overlap it with, while
> the deferred-input bookkeeping is paid every tick regardless.

## Methodological caution (recommended for the paper — it is a real finding)

> An intermediate fit performed on a narrower design, in which the federate count
> was held fixed, indicated that the *placement* of edges dominated their number
> (per-federate model R² = 0.73 against 0.53 for a total-count model). Widening
> the design reversed this conclusion decisively (R² = 0.22 against 0.97): with a
> fixed federate count the two regressors are collinear, and no fit statistic on
> that design revealed the problem. Two configurations with identical per-federate
> load but 64 versus 1024 total edges differ 23-fold in cost. We therefore report
> total edge count as the primary regressor and note that cost models for
> co-simulation coupling must vary topology breadth and depth independently to be
> identifiable.

## Figure captions

> **Figure 10.** Marginal per-tick cost of HELICS coupling against total edge
> count, for intra- and cross-federation topologies (log–log; 300 ticks, 3
> repetitions, payload width 1, every-tick publication, `same_step` causality).
> Dashed lines are relative-error-weighted fits through the origin. Vertical
> spread at fixed edge count reflects the secondary dependence on per-federate
> link concentration.

> **Figure 12.** Coupling cost against published payload width. Cost is
> essentially flat below ~512 bytes per message and linear in bytes above it.

> **Figure 13.** Coupling cost against publish cadence. Publishing every k-th
> tick reduces cost roughly in proportion to 1/k, reaching a 90% reduction at
> k = 10.

## Structural limits to state in the paper

> Two structural constraints emerged that are properties of the framework rather
> than of its performance. First, `same_step` dependency cycles are rejected at
> validation time, so fully-coupled topologies cannot be expressed without
> designating at least one edge as next-step; a bipartite publisher/subscriber
> decomposition is the natural way to keep a coupling study well-posed. Second,
> HELICS's summing multi-input handler reduces vector payloads across all
> elements, so an aggregating subscription yields a scalar rather than an
> element-wise sum — aggregation cannot preserve per-source payload structure
> without the vectorising handler.

## Results — stress to first failure (max scale)

> To locate the framework's actual limits we ramped each scaling axis
> geometrically until a run failed, re-checking host headroom before every step.
> Along the model-instance axis no failure was reached: eight federates hosted
> 32,768 instances exchanging 65,536 coupled values per tick at 329 ms per tick and
> 3.6 GB resident, with 97% of host memory still free when the ladder was
> exhausted. Per-tick cost was exactly linear in instance count (≈0.080 ms per
> instance) and memory grew sub-linearly, since all instances of a federate share
> one process. Along the federate axis the limit was reached at 128 federates;
> per-tick cost grew quadratically (1.27, 6.1, 24 and 101 ms as the federate count
> doubled) and memory linearly at ≈143 MB per federate. The quadratic growth is not
> a property of federates but of the topology their number implies: with
> all-to-all coupling the edge count grows as the square of the federate count, and
> cost is linear in edge count.

> The failure at 256 federates was not resource exhaustion. Memory was 96% free, no
> communication error was raised, and no deadline was missed: the simulation ran to
> completion and the *run* never terminated. Exactly half the federates reached the
> final tick and force-disconnected on their own timers while the remaining
> federation blocked indefinitely. The same signature appears in distributed runs
> once per-tick exchanged data exceeds roughly one kilobyte. We therefore report the
> binding limit on the federate axis as a shutdown-path defect rather than a
> scalability property, and note that the instance-axis limit was never reached.

## Takeaway sentence (one line)

> Coupling, not computation, sets the scale of a co-simulation: instance count is
> essentially free (32,768 instances on one host, linear cost, no failure), whereas
> coupled federate count is limited both by an edge count that grows quadratically
> with it and by a shutdown-path defect that binds well before any resource does.
