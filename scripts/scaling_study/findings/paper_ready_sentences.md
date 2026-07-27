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
