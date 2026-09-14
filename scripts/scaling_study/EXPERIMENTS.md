# What We Measured, In Plain Language

A guided tour of the Phase-D data-exchange campaign: what each experiment asked,
what we ran, what came out, and which file holds what. **No distributed-computing
or networking background assumed** — §0 defines every term used later.

- Want the *numbers and conclusions*? → `findings/phaseD_exchange.md`
- Want to *re-run or modify* an experiment? → `RUNBOOK.md`
- Want the *machine-readable schemas*? → `CONTRACT.md`
- Want the *canonical status of the whole study*? → `findings/README.md`

---

## 0. Vocabulary (read this once, everything else follows)

CosimGym runs **co-simulations**: several simulators run side by side and exchange
values as simulated time advances, so a building model and a grid model can affect
each other.

| Term | Plain meaning |
|---|---|
| **HELICS** | The library underneath CosimGym that actually moves values between simulators and keeps their clocks aligned. |
| **Federate** | One simulator process. Runs as its own OS process. In this study each federate hosts many copies of a small dummy model. |
| **Federation** | A group of federates that share one coordinator. |
| **Broker** | The coordinator process for a federation. All messages inside that federation pass through it. With more than one federation, an extra "hierarchy broker" sits above and links them. |
| **Tick** | One step of simulated time. Every federate must finish its tick before *anybody* moves to the next one — a lockstep barrier. So **the slowest federate sets the pace for everyone**. |
| **`tick_mean_s`** | Average wall-clock seconds per tick. The headline performance number throughout — **but only up to ~64 federates**. Above that the very first tick absorbs the whole spawn skew (at 176 federates one tick took 27 s of a 27.2 s run), and the mean stops describing steady-state cost. Use `tick_median_s` there. See Experiment 6. |
| **Publication / subscription** | A federate *publishes* a value under a name; other federates *subscribe* to that name to receive it. |
| **Edge** (`n_edges`) | One publisher→subscriber link. If 10 federates each subscribe to 1 value, that is 10 edges. This turned out to be **the** number that predicts cost. |
| **Payload / `msg_width`** | How much data one message carries, counted in doubles (8 bytes each). `msg_width=1` is a single number; `msg_width=1024` is an 8 kB vector. |
| **`freq`** | Publish cadence. `freq=10` means the publisher only sends every 10th tick. |
| **`causality`** | `same_step` = the subscriber needs the value *this* tick (it sits on the critical path). `next_step` = it can use last tick's value. |
| **Core type** (`zmq`, `zmq_ss`, …) | How HELICS moves bytes. `zmq` = normal. `zmq_ss` = "single socket", everything squeezed through one connection. |
| **NAT** | A network setup where a machine can dial *out* but cannot be dialled *into*. Our two cloud machines are behind NAT, which is **why** `zmq_ss` is mandatory for them — the plain `zmq` mode needs machines to be reachable from outside. |
| **Placement** | Which physical machine each federate runs on. `local` = all on one box; `distributed_nat` = spread across the manager + cloud machines over SSH. |

### The one method you must understand: the **paired delta**

We never compare raw tick times across different scenarios — too many things
differ. Instead every wired scenario has a **control twin**: the identical
scenario with the data exchange *switched off* (`exchange: none`). The number we
report is the difference:

```
Δ tick  =  tick_mean_s (wired)  −  tick_mean_s (identical control, no wiring)
```

That difference is the *cost of the coupling alone* — compute and clock-sync
cancel out because they are the same in both. Everything in §2–§6 below is a Δ.

### The two dummy models

Real models mix compute cost and communication cost together, which makes it
impossible to attribute a slowdown. So we used purpose-built dummies:

| Model | File | Purpose |
|---|---|---|
| `heavy_compute_dummy` | `src/models/model_catalog/physical_models/heavy_compute_dummy.py` | Burns CPU on demand, exchanges nothing. Isolates **compute** (used in Part A, before this campaign). |
| `exchange_dummy` | `src/models/model_catalog/physical_models/exchange_dummy.py` | **Built for this campaign.** Near-zero CPU, but reads every value it is sent and publishes a vector of configurable width. Isolates **communication**. |

`exchange_dummy` has one deliberate subtlety worth knowing: it builds its output
vector from a template computed once, and only overwrites element 0 each step. If
it rebuilt the whole vector every step, the *model's own CPU cost* would grow with
`msg_width` and we would have mistaken Python list-building for network cost.

---

## 1. Map of the campaign

Five experiment runs, in the order performed. Each row links its inputs to outputs.

| # | Experiment | Recipe file (input) | Raw results (output) | Runs |
|---|---|---|---|---|
| 1 | Local, gate-limited | `matrices/phaseD_local.yaml` | `findings/phaseD_local.csv` | 135 |
| 2 | Local, wide range | `matrices/phaseD_local_wide.yaml` | `findings/phaseD_local_wide.csv` | 177 |
| 3 | Across machines | `matrices/phaseD_cross_machine.yaml` | `findings/phaseD_cross_machine.csv` | 54 |
| 4 | Stress: instances | *(no matrix — CLI ladder)* | `findings/stress_M.csv` | 7 |
| 5 | Stress: federates | *(no matrix — CLI ladder)* | `findings/stress_N.csv` | 5 |
| 6 | Huge scale, one PC vs three | `matrices/test_matrix_hugescale_multipc.yaml` | `findings/large_test_Pietro_remote.csv` | 8 |

**Total: 456 scenario runs**, each in its own isolated subprocess.

Experiment 6 was run by hand at the end of the campaign and is **exploratory**
(one run per cell, no repeats). It is written up in
`findings/hugescale_multipc.md` and deliberately kept out of the paper prose.

### Tools used

| Script | Role |
|---|---|
| `gen_scenario.py` | Turns knobs (F, N, M, wiring, …) into a valid scenario YAML. Nothing is hand-written. |
| `run_bench.py` | Reads a matrix file, calls `gen_scenario.py` per cell, runs it isolated, samples CPU/RAM, appends one CSV row per run. |
| `stress_ramp.py` | **Built for this campaign.** Climbs one axis geometrically until something breaks, re-checking machine headroom before each step. Stops at first failure. |
| `cost_model.py` | `fit` turns a CSV into fitted coefficients; `predict` applies them. |
| `plot_exchange.py` | **Built for this campaign.** Produces figures 10–14. |

### Everything this campaign produced

**New code**
- `src/models/model_catalog/physical_models/exchange_dummy.py` + its entry in `src/models/model_catalog/catalog.yaml`
- `src/scenarios/exchange_dummy_test.yaml` (+ registered in `tests/regression_suite.py`)
- `scripts/scaling_study/stress_ramp.py`, `scripts/scaling_study/plot_exchange.py`
- `tests/test_cost_model_comms.py` (6 tests)

**Modified code**
- `gen_scenario.py` — added the whole wiring layer (previously every federate was isolated)
- `run_bench.py` — new CSV columns + orphan-process reaper
- `cost_model.py` — the fitted `comms` term

**Data / figures / prose**
- 5 CSVs and 2 fitted-parameter JSONs (table above, plus `phaseD_fit_params.json`, `phaseD_wide_fit_params.json`)
- `findings/10…14_*.png`
- `findings/phaseD_exchange.md` (results), `findings/paper_ready_sentences.md` (paper prose),
  `findings/bottlenecks.md` (what breaks), `findings/README.md` (canonical index)

---

## 2. Experiment 1 — first look, deliberately small

**Question.** Does coupling cost anything measurable, and which knobs matter?

**Why small.** The project's own safety rule (plan §8.7) caps unattended runs on
this shared machine at 2 federations × 4 federates × 4 instances × 30 ticks. This
experiment sits exactly at that cap.

**Design.** 27 configurations, 5 repeats each. Not every combination of every knob
(that would be hundreds of runs and, as Part A learned the hard way, produces a
fit where you cannot tell which knob caused what). Instead **one factor at a
time** around a fixed baseline: change distance, or fan-out, or payload, or
cadence, or causality — never several at once.

**Result and its fate.** It produced a clean-looking answer: that *where* edges
attach matters more than how many there are. **That answer was wrong** — see §3.
Kept in the repo as evidence for the methodological lesson.

---

## 3. Experiment 2 — widen the range, and overturn Experiment 1

**Question.** Does the law hold at district scale?

**Design.** After explicit go-ahead to exceed the safety cap: up to 32 federates
per federation (64 processes), up to 64 model instances each, 300 ticks. 59
configurations × 3 repeats. Edge counts span **8 → 4096**.

**The crucial design choice.** Experiment 1 held the federate count fixed at 4. If
you never change the federate count, then "total edges" and "edges landing on the
busiest federate" rise and fall *together*, and no statistic can separate them —
they are **collinear**. Experiment 2 varies federate count and instance count
*independently*, which breaks the tie.

**The decisive comparison.** Two configurations with the *same* busiest-federate
load (64) but very different total edges:

| shape | busiest federate | total edges | Δ tick |
|---|---|---|---|
| `Nto1` | 64 | 64 | +234 µs |
| `all2all` | 64 | 1024 | +5383 µs |

Same per-federate load, **23× the cost**, tracking total edges. Experiment 1's
conclusion inverted: on the wide data the "placement matters" model scores R²
0.22 where the "total count matters" model scores 0.97.

**Why this is a finding, not an embarrassment.** No warning sign appeared in
Experiment 1's statistics — its fit looked good. Only a wider design exposed the
problem. This is written up as a caution in `findings/paper_ready_sentences.md`
because it applies to anyone fitting a cost model to a co-simulation.

**What came out.**
```
coupling cost ≈ per_edge[distance] × (total edges)  +  per_byte × (bytes per tick)
```
with per-edge ≈ **3.14 µs** inside a federation, **4.36 µs** across federations,
and ≈ **1.66 ns per byte**. Fitted in `findings/phaseD_wide_fit_params.json`.

Two practical rules fell out:
- **Payload is free until ~512 bytes per message**, then cost grows with bytes.
- **Publishing every 10th tick removes 90% of the coupling cost** — cost follows
  messages actually sent, not subscriptions declared. Cheapest available lever.

One honest gap: at *equal* edge count, packing coupling into few large federates
costs ~3× more than spreading it. Real, measured, but not captured by the
formula — recorded as an open gap rather than papered over.

---

## 4. Experiment 3 — does crossing a network cost extra?

**Question.** How much more expensive is a link between two machines than one
inside a single machine? This is the number that decides where to put federates.

**The trap, and how it was avoided.** The distributed machines are behind NAT, so
they *must* use the `zmq_ss` core type; the local experiments used plain `zmq`.
Comparing them directly would blend two changes — the network hop *and* the core
type — and you could not tell which caused what. (Phase 4 of the earlier study was
ruined by exactly this class of mistake.)

So the recipe is **two-armed**: every distributed configuration has a *local twin*
using the same `zmq_ss` core type and the same wiring, with no network involved.
The difference between the arms is the network hop, and nothing else.

**Result — the opposite of what the plan assumed.**

| | M=1 | M=4 |
|---|---|---|
| local (`zmq_ss`) | 6.53 µs/edge | 6.97 µs/edge |
| across machines | 6.46 µs/edge | 4.60 µs/edge |

A network link costs **no more** than a local one. Why: HELICS makes every
federate wait at a barrier each tick anyway, so the network round-trip happens
*inside* waiting time that was already being spent, while moving federates to a
second machine relieves CPU competition on the first.

This invalidated a premise written into the study plan ("placement should minimise
cross-machine edges"). The plan has been corrected in both places it appeared.

**Scope limit, stated plainly:** same-campus network, idle remote machine, small
edge counts. A slow or busy link would likely behave differently.

**But wide payloads break it.** Every distributed configuration sending ≥512 bytes
per message hung. That is `bottleneck B12` — see §6.

---

## 5. Experiments 4 & 5 — how far can it actually go?

Rather than a fixed recipe, `stress_ramp.py` doubles one knob at a time and stops
at the first failure. Before every step it re-checks free memory and machine load
and aborts if the budget would be exceeded — an abort is recorded as "we chose to
stop", never as "the framework broke".

**Experiment 4 — more model instances (`--axis M`).** Never failed. The ladder ran
out of rungs at **32,768 model instances in 8 federates** (65,536 edges), 329 ms
per tick, 3.6 GB, with 97% of memory still free. Cost is a straight line in
instance count (≈0.080 ms each) and memory barely moves, because all instances of
a federate live inside that one process.

**Experiment 5 — more federates (`--axis N`).** Failed at **256 federates**; max
stable **128**. Two clean patterns before the wall:
- Tick time grows **quadratically** — 1.27 → 6.1 → 24 → 101 ms per doubling. Not
  because federates are expensive, but because this topology's edge count grows as
  the square of the federate count, and cost is linear in edges.
- Memory grows **linearly**, ≈143 MB per federate. Federates cost memory;
  instances are nearly free.

**Answering the original question.** With data exchange on: **32,768 instances is
not a limit** (never reached one), while **128 federates is**. Different axes,
different walls, and the instance wall was never found.

---

## 5b. Experiment 6 — the biggest thing we ran, on one PC and on three

Everything above stresses **one** axis at a time. This last run pushes two at once:
**176 federates** — exactly one per core across the whole rig (112 + 32 + 32) —
each holding 10, then 100, then 1 000, then 10 000 model instances. The same four
steps were run twice: all on the manager, then spread over the three machines. The
two runs differ in *nothing* except where the federates live, so subtracting them
isolates what distribution buys. No data exchange in this one (back to the Part-A
dummy model), so read the tick numbers as a floor.

**It held to 176 000 model instances.** N=176 × M=1 000 finished on both
placements — 432 ms/tick on one machine, 271 ms/tick on three. That is the largest
*federates × instances* product in the whole study; the stress ladders above each
pushed a single axis.

**Spreading over three machines bought almost exactly what the core count
predicts.** Three machines have 176 cores against the manager's 112 — a ceiling of
1.57×. Measured end-to-end speedup: **1.48–1.68×**. The network hop cost nothing
visible, which is the same conclusion Experiment 3 reached from a completely
different angle. Startup got faster too: spawning 176 federates over SSH beat
spawning them all locally by 1.7–2.8×.

**Parallel workers stopped paying off at high instance counts.** Compare each run
against a "perfect packing" bound — all the model work spread evenly over every
core. At 100 instances/federate on three machines the run sat at **1.3× the
bound**, near-ideal. At 1 000 instances/federate **both** placements degraded to
**~9.5× the bound**, identically — so it is not a network effect. The prime
suspect is worker oversubscription: 176 federates × 10 workers is 1 760 processes
competing for 112 cores. It stays a suspect, not a conclusion, because no
sequential control run was made (`bottlenecks.md` B13).

**The next rung broke.** 10 000 instances/federate — 1.76 million instances — timed
out on both placements. Whether that is the teardown stall (B12) or an honest
resource wall is **unknown**: the harness deleted the logs on cleanup. Redo it with
`--keep-scratch`.

**A measurement lesson worth more than any single number.** At 176 federates,
`tick_mean_s` became meaningless — one startup tick swallowed 27 of the run's 27.2
seconds while the typical tick took 2 ms. The average was describing how long
federates took to *start*, not how long a tick *costs*. Above ~64 federates, read
medians. The same distortion is baked into the CSV's `throughput_inst_steps_s`
column, which divides by total wall time: it reports 1 296 instance-steps/s where
the steady state is 834 000.

---

## 6. What broke, and why it matters

Three failures were found. Two are fixed; one is open and is the most important
result of the campaign.

**Port planning (fixed).** Every run with ≥8 federates per federation failed —
*including the unwired controls*, which is what proved it was not a data-exchange
problem. With plain `zmq`, each federate opens its own listening port numbered
from its federation's base port, so a federation actually occupies a *block* of
`10 + N` ports. The generator was leaving a gap of only 10, so one federation's
federates collided with the next federation's broker. Fixed by sizing the block
from N. Checked whether this had contaminated earlier published results: it had
not, because those runs used `zmq_ss`, which opens no such ports.

**Orphan cascade (fixed).** When a run is killed, its broker and federate
processes survive — they sit outside the process group that gets killed — and keep
holding their ports, so *every following run* fails with "port already in use".
One stall caused **19 spurious failures** before this was diagnosed. `run_bench.py`
now hunts down and kills leftover brokers after a timeout (narrowly: only
processes owned by this user, started after this run began).

**The teardown stall (OPEN — `bottlenecks.md` B12).** The most serious limit found.
**The simulation finishes and the run refuses to end.** At 256 federates, exactly
half of them reached the final tick and gave up waiting on a disconnect timer,
while the other half blocked forever. Not out of memory (96% free), not a
communication error, not a missed deadline — a **shutdown** problem. The identical
signature appears in distributed runs once traffic exceeds roughly 1 kB per tick.
It is undiagnosed, and it blocks the next phase of the study, because both
thresholds sit underneath the configurations that phase needs to explore.

---

## 7. Where each conclusion is written down

| You want | Read |
|---|---|
| Numbers, tables, caveats | `findings/phaseD_exchange.md` |
| Sentences to paste into the paper | `findings/paper_ready_sentences.md` |
| What breaks and how to avoid it | `findings/bottlenecks.md` (B10, B11, B12, B13) |
| The huge-scale one-PC-vs-three probe | `findings/hugescale_multipc.md` |
| Status of the whole study | `findings/README.md` ← wins any disagreement |
| Figures | `findings/10…14_*.png` |
| How to re-run or change any of this | `RUNBOOK.md` |
