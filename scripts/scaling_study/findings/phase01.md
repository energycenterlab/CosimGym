# Scaling Study — Phase 0 (calibrate primitives) + Phase 1 (crossover)

Local-only (`placement: local`), manager machine (112 cores), conda env
`cosim_gym`. Harness used as-is (`run_bench.py`, `cost_model.py`,
`make_report.py`) — no harness code changes; no bugs found (see caveat below,
which is a fitting/design-of-experiment gotcha, not a code defect).

Matrices: `scripts/scaling_study/matrices/phase0{a,b,c,d}_*.yaml`,
`phase1_crossover_v2.yaml`. Raw per-phase CSVs + combined
`bench_all.csv` + `fit_params.json` are in this directory. Plots:
`01_crossover.png` (from `bench_all.csv`, clean — see note), `02_sync_curve.png`
(regenerated from `bench_phase0a.csv` alone — the report's default sync-curve
pools all matrices' seq/zmq rows regardless of M/work, which for the
concatenated file visually contaminates the N=1 point with heavy-work rows
from other matrices; the version here isolates the intended N-sweep at
fixed M=1, work=1).

Total wall-clock: **~5.5 min** of actual scenario runs (0a 77s, 0b 52s,
0c 32s, 0d 27s, phase1 140s) — well inside the ~10 min/matrix bound. No
timeouts, no crashes; all 61 successful rows + 3 expected `pv_dest` failures.

## Fitted primitives

| primitive | value | source |
|---|---|---|
| `c(heavy_compute_dummy)` intercept `a` | ≈ 3.0–3.9e-5 s/instance/tick | phase0b (a=3.87e-5) and phase1 (a=2.65e-5) self-consistent fits agree to within the same order of magnitude |
| `c(heavy_compute_dummy)` slope `b` | **1.34e-7 s/iteration** | phase0b fit b=1.3437e-7, phase1 fit b=1.3371e-7 — the two independent sweeps agree to <1% |
| `c(simple_building)` | ≈ 6.4e-5 s/instance/tick (M=1, defaults) | phase0c, 3 reps |
| `c(simple_heatpump)` | ≈ 6.2e-5 s/instance/tick (M=1, defaults) | phase0c, 3 reps |
| `c(pv_dest)` | **N/A — errors standalone** (`KeyError: 'GHI'`) | needs a wired weather federate; gen_scenario federates are self-contained/no-subscription by design (documented limitation, not fixed here) |
| `s(N)` zmq, `s0` | ≈ 1.30e-4 s | phase0a-only clean fit (N∈{2..64}, M=1, work=1) |
| `s(N)` zmq, `s1` | ≈ 2.65e-6 s/federate | same; curve is nearly flat 2≤N≤16, rises modestly and plateaus ~2.65e-4 s/tick by N=32–64 |
| `O_par` | ≈ 0.033–0.046 s/tick | 0.0329 from the global `cost_model.py fit` over `bench_all.csv`; 0.0462 from a phase1-only self-consistent fit (see caveat) — both close to the prior estimate (~0.039 s/tick) quoted in the task |
| `rss_per_instance(heavy_compute_dummy)` | **~0 MB/instance** | phase0d: peak_rss_mb flat at ~304–305 MB for M∈{1,4,16,64} — dominated entirely by the base Python/HELICS process footprint, no measurable per-instance memory growth for this pure-float-arithmetic model |
| `rss(simple_building/simple_heatpump)` | ~304–305 MB (M=1 only, not separable from baseline) | phase0c |

Global `fit_params.json` (written by `cost_model.py fit` over the concatenated
`bench_all.csv`) is in this directory. **Caveat**: that single joint fit
regresses sync `s0+s1*N` and compute `c=a+b*work` using `tick_mean_s ~ 1 + N + M`
across *all five matrices pooled together*. Because no single matrix varies
both N and M together (0a varies N at fixed M=1; 0d varies M at fixed N=1; 0b/1
vary work at fixed N,M), the joint regression is close to collinear across the
pooled dataset and produces a slightly negative `s1` (-3.5e-4) and an inflated
`c` intercept (a=2.65e-4, vs. ~3-4e-5 from the matrix-isolated fits above). The
per-matrix isolated fits (0a alone for s(N); 0b/1 alone for c(work), which
agree with each other to <1% on the slope) are the trustworthy primitives;
the pooled global fit is included per contract but should be treated with
caution when matrices don't share a common varied-axis design. This is a
design-of-experiment artifact of concatenating single-axis sweeps into one
regression, not a harness bug.

## Phase 1 — crossover result

Config: F=1, N=1, M=16, W=8, zmq, `heavy_compute_dummy`, ticks=20, work ∈
{1000, 10000, 25000, 50000, 100000}, 3 repeats.

| work | seq tick_mean_s | par tick_mean_s | par − seq |
|---|---|---|---|
| 1,000 | 0.00245 | 0.04307 | +0.0406 (seq wins) |
| 10,000 | 0.02207 | 0.04713 | +0.0251 (seq wins) |
| 25,000 | 0.05398 | 0.05351 | **−0.00047 (par wins, barely)** |
| 50,000 | 0.10704 | 0.05942 | −0.0476 (par wins) |
| 100,000 | 0.21449 | 0.07801 | −0.1365 (par wins) |

Crossover brackets tightly between work=10,000 (seq still faster) and
work=25,000 (par just overtakes) — three independent estimates agree:

- **make_report.py's own crossover detector** (linear interp on the sampled
  grid): **≈23,500**
- **Interpolated zero-crossing** of (par−seq) between the two adjacent
  samples: **≈24,700**
- **Predicted from the law** `(M − ⌈M/W⌉)·c = O_par` using the phase1-local
  self-consistent fit (a=2.65e-5, b=1.337e-7, O_par=0.0462): solving
  `14·c* = O_par → c* = 0.0033 → work* = (c*−a)/b`: **≈24,500**

All three land in the **23.5k–24.7k** band — the measured crossover matches
the predicted crossover from the law to within ~1000 work units (~4%). The
law `(M−⌈M/W⌉)·c > O_par` is confirmed: parallel execution only pays off once
the *sequential-instance savings* (14 of the 16 instances skip the sequential
loop, in this W=8 config) exceed the fixed per-tick dispatch/IPC overhead.

Notably, `par`'s own tick time is *not* flat vs. work (0.043s → 0.078s across
the sweep) — consistent with the framework's `ceil(M/W)·c + O_par` term: with
M=16, W=8, each of the 8 workers still steps `ceil(16/8)=2` instances, so
`par` scales with `2·c(work)`, just at 1/8th the slope of `seq`'s `16·c(work)`.

## Real models tried standalone (Phase 0c)

- `simple_building` — **ran cleanly** on catalog defaults, c≈6.4e-5 s/tick.
- `simple_heatpump` — **ran cleanly** on catalog defaults, c≈6.2e-5 s/tick.
- `pv_dest` — **errored** (`KeyError: 'GHI'`) every rep. Expected: gen_scenario
  federates have no subscriptions, and `pv_dest` requires an external
  irradiance (`GHI`) input normally supplied by a weather federate. Not a
  harness bug — noted per task instructions and not worked around.

## Files in this directory

- `phase01.md` — this report
- `bench_all.csv` — concatenated phase0{a,b,c,d}+phase1 rows (80 data rows)
- `fit_params.json` — output of `cost_model.py fit results/scaling/bench_all.csv`
- `01_crossover.png` — seq vs par crossover curve (from `bench_all.csv`)
- `02_sync_curve.png` — sync cost vs N (regenerated from `bench_phase0a.csv`
  alone for a clean, non-contaminated curve — see caveat above)
