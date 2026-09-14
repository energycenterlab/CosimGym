# Huge-scale multi-PC probe — N=176 federates × M up to 10 000 instances

**Run date:** 2026-07-28 (12:59–13:16). **Written up:** 2026-08-21.
**Raw data:** [`large_test_Pietro_remote.csv`](large_test_Pietro_remote.csv) ·
**matrix:** [`../matrices/test_matrix_hugescale_multipc.yaml`](../matrices/test_matrix_hugescale_multipc.yaml)
**Status:** ⚠️ exploratory — **1 repeat per cell, 20 ticks, no `seq` control arm**.
Treat every number below as a single observation, not a calibrated primitive. Nothing
here supersedes the fitted primitives in [`README.md`](README.md).

---

## 1. What was run

A hand-driven probe of the **largest configuration attempted so far on the combined
federate × instance axes**, run twice — once entirely on the manager, once spread over
three machines — so the two arms are a paired local-vs-distributed comparison.

| Knob | Value |
|---|---|
| `F` / `N` / `M` | 1 federation / **176 federates** / **10 → 100 → 1 000 → 10 000** instances per federate |
| `mode` / `W` | `par` / 10 workers per federate |
| `core_type` | `zmq_ss` (forced by `distributed_nat`; kept in the local arm too so the arms differ *only* in placement) |
| model / `work` | `heavy_compute_dummy`, `work=100` → `c = 2.85e-5` s/instance/tick |
| `exchange` | **`none`** — zero subscriptions, as in Part A |
| `ticks` / repeats | 20 / **1** |
| placement | `local` (manager only) vs `distributed_nat` (manager 112 fed + `machine_a` 32 + `machine_b` 32) |

`distributed_nat` splits federates core-proportionally (`gen_scenario.flatten_placement`).
With the rig's 112 + 32 + 32 = **176 cores** and N = 176, this lands **exactly one
federate per core** — which is why this cell is a useful roofline probe rather than an
arbitrary point.

> The committed matrix file holds the `distributed_nat` arm. The local arm is the same
> four rows with `placement: local`; both arms are in the CSV. The matrix has since been
> extended to carry both arms so a single `run_bench.py` invocation reproduces the whole
> probe.

## 2. Raw results

Steady-state cost is `tick_median_s` — **not** `tick_mean_s`, for the reason in §6.

| placement | M | instances | tick median | tick p95 | tick mean | sim_wall | setup | spawn | Σ RSS | CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| local | 10 | 1 760 | **2.11 ms** | 14.4 ms | 1358 ms | 27.2 s | 34.5 s | 8.30 s | 138 GB | 5682% |
| local | 100 | 17 600 | **13.27 ms** | 33.9 ms | 1301 ms | 26.0 s | 36.3 s | 7.73 s | 159 GB | 6354% |
| local | 1 000 | 176 000 | **432.3 ms** | 2310 ms | 726 ms | 14.5 s | 54.6 s | 5.14 s | 261 GB | 6549% |
| local | 10 000 | 1 760 000 | — | — | — | **timeout** | — | — | 283 GB | 9940% |
| distributed | 10 | 1 760 | **1.82 ms** | 29.9 ms | 807 ms | 16.1 s | 27.2 s | 3.25 s | 116 GB | 4910% |
| distributed | 100 | 17 600 | **3.81 ms** | 111.8 ms | 879 ms | 17.6 s | 26.7 s | 2.76 s | 128 GB | 4797% |
| distributed | 1 000 | 176 000 | **271.0 ms** | 708.8 ms | 488 ms | 9.8 s | 38.0 s | 3.09 s | 167 GB | 5883% |
| distributed | 10 000 | 1 760 000 | — | — | — | **timeout** | — | — | 178 GB | 10153% |

`Σ RSS` = `peak_rss_mb`, the **sum** of RSS over the local process tree — it double-counts
copy-on-write shared pages (261 GB "exceeds" the manager's 251 GB of physical RAM), and in
the distributed arm it covers **manager-side processes only**. Same for `cpu_util_pct`.

## 3. Result — 176 000 model instances ran clean; that is a new maximum

**N=176 × M=1000 = 176 000 model instances across 176 federates completed on both
placements.** Previous maxima: 32 768 instances over 8 federates (Phase D, *with* data
exchange) and 1 600 instances over 8 federates (Part A). This probe is the first to push
**both axes at once** — 176 federates *and* six figures of instances — and it held, at
407 k (local) / 649 k (distributed) instance-steps/s steady state.

Caveat that bounds the claim: `exchange: none`. Phase D showed coupling dominates at
scale (22.3 ms/tick at 4096 edges vs a 211 µs unwired baseline), so this is a
**compute/oversubscription ceiling, not a district-scale ceiling**. A wired 176 000-instance
run is untested and would likely hit B12 first.

## 4. Result — distribution speedup tracks the core-pool ratio (the roofline Phase 4 missed)

| M | tick median local/dist | sim_wall local/dist | spawn local/dist |
|---|---|---|---|
| 10 | 1.16× | 1.68× | 2.55× |
| 100 | 3.48× | 1.48× | 2.80× |
| 1 000 | 1.59× | 1.49× | 1.66× |

The roofline ceiling for this rig is `Σcores / local_cores = 176/112 = **1.57×**`.
Measured wall speedup is **1.48–1.68×, mean ≈1.55×** — i.e. distribution delivered
essentially the whole predicted core-pool gain, and the LAN hop cost nothing visible.
That is consistent with Phase D's `κ_LAN ≈ κ_local` and is the **first roofline
observation at this scale that is not confounded** the way Phase 4 was — though it is
n=1 and the manager's co-user load during the window was not recorded, so Phase E still
needs to do this properly with repeats on a verified-idle box.

Distribution also cut **spawn time 1.7–2.8×** (SSH-parallel spawning beats 176 local
`fork`+import storms) and manager-side memory 1.2–1.6×.

## 5. Result — both arms sit ~9.5× above the perfect-packing bound at M=1000

Perfect packing = total per-tick model work spread over every available core,
`N·M·c / cores`:

| placement | M | measured median | perfect-packing bound | ratio |
|---|---|---|---|---|
| local | 10 | 2.11 ms | 0.45 ms | 4.7× |
| local | 100 | 13.27 ms | 4.48 ms | 3.0× |
| local | 1 000 | 432.3 ms | 44.8 ms | **9.7×** |
| distributed | 10 | 1.82 ms | 0.29 ms | 6.4× |
| distributed | 100 | 3.81 ms | 2.85 ms | **1.3×** |
| distributed | 1 000 | 271.0 ms | 28.5 ms | **9.5×** |

Two readings:

- At **M=100 distributed** the federation ran at 1.3× the theoretical bound — near-ideal
  packing, the best efficiency seen anywhere in the study.
- At **M=1000 both arms degrade to ~9.5×** the bound, *identically*. The efficiency loss
  is therefore **not** a distribution effect; it scales with per-federate instance count.
  Candidate causes, none isolated here: `par` worker oversubscription (176 federates × 10
  workers = **1760 runnable processes on 112 cores, 15.7×**; distributed is a uniform 10×
  on every machine), memory pressure at 261 GB against 251 GB physical, and per-tick
  worker dispatch. Deciding between these needs a `seq` control arm at the same N and M —
  which this probe does not have. See B13 in [`bottlenecks.md`](bottlenecks.md).

## 6. Result (methodological) — at large N, `tick_mean_s` measures spawn skew, not tick cost

`tick_mean_s` is the mean of the *gating* federate's own tick times
(`ScenarioManager._write_perf_json`), and that federate's span covers the whole
simulation — so `tick_mean × n_ticks == sim_wall` exactly (27.153/20 = 1.3577 ✓).
With 20 ticks, p95 = 14 ms and mean = 1358 ms, **one single tick absorbed ~27 s**: the
first tick, where the gating federate waits at the barrier for all 176 federates to
finish spawning and initialising.

Consequences, and they reach beyond this probe:

- **Use `tick_median_s` for steady-state cost at large N.** `EXPERIMENTS.md` calls
  `tick_mean_s` "the headline performance number" — that holds at small N, where the
  startup tick is not 99% of the run. It does not hold here.
- **`throughput_inst_steps_s` in the CSV is computed from `sim_wall_s`** and is therefore
  startup-dominated too: it reports 1 296 inst-steps/s at M=10 local where the steady
  state is 834 000. The throughput column is only meaningful for runs long enough to
  amortise startup — 20-tick cells are not.
- Symptom to recognise: `sim_wall` **falls** as M rises (27.2 s at M=10, 14.5 s at
  M=1000) because the first-cell/cold-start penalty outweighs 100× more compute.

## 7. Result — the additive `par` cost model fails in both directions at N=176

Predicting with the trusted primitives (`⌈M/W⌉·c + O_par + s0 + s1·N`, `O_par = 0.044`):

| M | predicted | measured (local) | error |
|---|---|---|---|
| 10 | 44.7 ms | 2.11 ms | **over-predicts 21×** |
| 100 | 44.9 ms | 13.27 ms | over-predicts 3.4× |
| 1 000 | 47.5 ms | 432.3 ms | **under-predicts 9.1×** |

The model is dominated by the constant `O_par = 0.044 s/tick`, which at M=10/100 is
plainly not being paid — persistent workers are pre-warmed, so per-tick dispatch here is
milliseconds, not 44 ms — while at M=1000 the real cost overshoots it 9×. So `O_par` is
**not a scale-free constant**: it was fitted in a small-N, small-W regime and does not
transfer to N=176/W=10. This is the same failure family as the known N×work interaction
gap (README gap #3), extended to a third axis: **N × M × W**. The crossover law's
*structure* is untouched by this — no `seq` arm was run, so nothing here tests it.

## 8. Result — M=10 000 (1.76 M instances) times out on both placements, undiagnosed

Both arms failed with `failure_mode: timeout` (the harness wall-clock kill in
`run_bench.run_once`), at 9940% / 10153% CPU — ~100 cores saturated — and 283 GB / 178 GB
summed RSS. **It is not known whether this is B12 (teardown stall) or an honest
resource wall**: the default cleanup deleted the scratch scenario and every log, so
there is nothing to inspect. The two are distinguishable — B12's signature is
`disconnect Timer expired forcing disconnect` in half the federates *after* the physics
completed.

**Redo with `--keep-scratch` and a generous `--timeout` before drawing any conclusion.**
If it is B12, this is a third trigger for it (high instance count, unwired), which would
be new information: both known triggers so far involve federate count or exchanged bytes.

## 9. Limits of this probe

1. **1 repeat per cell** — no variance estimate; the M=100 distributed outlier (3.48×
   tick speedup, 1.3× packing) may not survive repetition.
2. **20 ticks** — startup dominates wall-clock; only medians are usable.
3. **No `seq` control arm** — the oversubscription hypothesis in §5 is unfalsified.
4. **`exchange: none`** — all tick numbers are lower bounds for a wired scenario (B11).
5. **Manager load during the window not recorded** (`uptime` not captured) — the Phase 4
   confounding lesson is not fully closed out here.
6. **Timeout value used is not recorded** in the CSV schema; the harness default is 180 s.
7. RSS/CPU are manager-local sums with shared pages double-counted (§2).

Because of 1–5, **none of these numbers went into `paper_ready_sentences.md`.** The
roofline result (§4) and the packing result (§5) are the two worth promoting after a
repeated, idle-host rerun — that rerun is naturally part of **Phase E**.

## 10. What this changes

- **Phase E has a concrete starting point:** rerun this exact 2×3 grid with ≥3 repeats,
  ≥60 ticks, both `seq` and `par`, on a verified-idle manager. If the 1.55× wall speedup
  holds, the roofline `Σcores/local_cores` is validated and README gap #4 closes.
- **B13 added** to `bottlenecks.md` (par-worker oversubscription at large N).
- **`O_par` is now known to be regime-bound** — flagged in README gap #3.
- **Reporting convention:** medians, not means, above ~64 federates.
