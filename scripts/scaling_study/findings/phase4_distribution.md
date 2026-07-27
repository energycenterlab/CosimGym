# Scaling Study — Phase 4 (distribution roofline)

Question: on this hardware (manager 112c + cloud1 32c + cloud5 32c), does
distributing federates ever pay off, and how close does it get to the
naive roofline ceiling `total_cores/local_cores = 176/112 = 1.571x`
(i.e. a distributed/local `sim_wall_s` ratio floor of `1/1.571 = 0.637`)?
Harness used as-is (`run_bench.py`, `cost_model.py`, `make_report.py`); no
harness code changes. **F=1 throughout** (avoids the Phase-3 zmq
port-spacing bug — irrelevant anyway since `distributed_nat` forces
`zmq_ss`, which that bug doesn't touch). The **local twin also uses
`core_type=zmq_ss`** (not plain `zmq`) so local and distributed are
compared on the identical wire protocol, per `generate_scale_benchmark.py`'s
own fairness rule ("both scenarios must use the SAME core_type to stay
comparable").

## Headline result — the opposite of the expectation in the task brief

The task brief (and `generate_scale_benchmark.py`'s docstring, measured
previously with the analytic building/PV/heatpump models) expected
distribution to be **slower** at low compute intensity and only approach
break-even as compute dominates sync. That is **not** what was measured
here: `distributed_nat` was **faster than local at every single work/N
level tested**, and at the higher end (N=88, work=50000) it beat even the
theoretical 0.637 roofline floor (measured ratio **0.548**, i.e. distributed
ran **1.82x faster** than local — more than the idealized 1.571x cores-only
ceiling predicts).

**Root cause, confirmed directly**: this is a **shared machine**, not an
idle 112-core box. At measurement time:

```
manager (ipazia):  nproc=112   load average: 67.33, 61.64, 42.81   (2 other users' tmux sessions running since Jun/Jul)
cloud1:             nproc=32   load average: 1.53, 0.99, 0.63       (idle)
cloud5:             nproc=32   load average: 1.73, 2.04, 1.01       (idle)
```

The "local" baseline places **all** N federates on a manager that is
already carrying a load average of ~67 from other tenants (`bottaccioli`,
`montaldo`, both with long-running tmux sessions per `who`), even though
N ≤ 112 so our own federates never oversubscribe the box on paper. The
`distributed_nat` placement moves 36% of the federates onto two essentially
idle 32-core remotes, which — despite adding LAN/SSH overhead — removes
them from contention with the manager's *background* load. This is an
honest, reproducible confound of testing on a shared server, not a
fabricated result: every one of the 26 runs below passed cleanly (no
`failure_mode` set), and the direction and magnitude of the effect is
monotonic and consistent across both sub-phases.

## 4a — distribution overhead vs compute intensity (N=48, sweep work)

Matrix `scripts/scaling_study/matrices/phase4a_distribution_overhead.yaml`,
CSV `phase4a_distribution_overhead.csv`. `F=1, N=48, M=1, mode=seq,
core_type=zmq_ss, model=heavy_compute_dummy, ticks=30`, work ∈ {1, 5000,
50000}, placement ∈ {local, distributed_nat}, 3 repeats/cell. **18/18
PASS.**

| work | local sim_wall_s (median) | distributed sim_wall_s (median) | ratio (dist/local) |
|---|---|---|---|
| 1 | 0.00784 | 0.00780 | 0.995 |
| 5,000 | 0.07995 | 0.07214 | 0.902 |
| 50,000 | 0.69770 | 0.64547 | 0.925 |

Per-tick deltas tell the same story more precisely (`tick_mean_s`,
distributed − local): −0.000002s (work=1), −0.000260s (work=5,000),
−0.001741s (work=50,000) — distributed is never slower, and the gap widens
(in absolute terms) as compute intensity rises, consistent with more
per-tick compute giving the contended manager more opportunity to fall
behind the idle remotes. In relative terms the ratio does **not** move
monotonically toward 1.0 as the plan hypothesized (it dips to 0.90 at
work=5,000, then partially recovers to 0.925 at work=50,000) — at N=48 the
manager's own background contention dominates the story more than the
work-driven compute-vs-sync balance does.

## 4b — approaching the roofline (N ∈ {64, 88}, work=50000, compute-bound)

Matrix `scripts/scaling_study/matrices/phase4b_roofline_approach.yaml`, CSV
`phase4b_roofline_approach.csv`. Same config, N ∈ {64, 88} (near, but under,
the Phase-2 SSH-ControlPath N≥112 cap), work=50000 fixed (compute-bound
region per 4a / Phase 1's crossover), 2 repeats/cell. **8/8 PASS.**

| N | local sim_wall_s (median) | distributed sim_wall_s (median) | ratio (dist/local) |
|---|---|---|---|
| 64 | 0.98984 | 0.69332 | **0.700** |
| 88 | 1.45811 | 0.79965 | **0.548** |

**Best ratio observed: 0.548 at N=88** — this is *better* (lower) than the
1/1.571 = 0.637 theoretical roofline floor for a cores-only,
contention-free comparison. As N grows from 64→88, more federates pile
onto the already-contended manager under `local`, while `distributed_nat`
keeps proportionally shedding load onto the idle remotes — so the
ratio keeps improving with N *specifically because of the contention
confound*, not because the job is becoming "more compute-bound" (work is
fixed at 50000 across both rows). A clean, contention-free environment
would be expected to plateau near 0.637, not blow through it.

See `phase4_ratio_plot.png` (custom 2-panel plot: ratio vs work at N=48;
ratio vs N at work=50000, both with the break-even line at 1.0 and the
roofline ceiling at 0.637 marked) — the N=88 point visibly sits below the
ceiling line. The generic D5 `03_roofline.png` (throughput vs total
instances by placement, from `make_report.py`) is also copied here per
task instructions, but — like the pooled-fit issue noted in Phase 1/3 — it
plots all three `work` levels of 4a on one throughput axis, which is
dominated visually by the work=1 points; `phase4_ratio_plot.png` is the
more legible artifact for this phase's actual question.

## Cost-model fit

`cost_model.py fit scripts/scaling_study/findings/phase4_combined.csv -> phase4_fit_params.json`
(26 rows = 4a's 18 + 4b's 8). Notable outputs:

- `s[zmq_ss]`: s0=0.0, s1=7.759e-4 s/federate (regressed over N ∈
  {48,64,88}) — steeper than Phase 2's isolated s(N) fit (s1≈2.65e-6 for
  plain `zmq` at N≤89, s1≈2.76e-6 for `zmq_ss` in Phase 3's local-only
  fit), because this fit's rows mix work levels and — for the
  `distributed_nat` rows — bake the contention/RTT effect into what should
  be a pure sync term. Read with the same caution flagged in Phase 1's
  "pooled fit across single-axis sweeps" caveat.
- `c[heavy_compute_dummy]`: fit lstsq to a=0.0, b=0.0 — the regression
  degenerates because `tick_mean_s ~ 1 + N + M` doesn't isolate `work` as a
  regressor at all (M is fixed at 1 throughout Phase 4, and the design
  varies `work` only within N=48 and `N` only within work=50000, so the two
  single-axis sweeps are as collinear here as they were in Phase 1's pooled
  fit) — **not a harness bug**, just this phase's design not being shaped
  for that particular term; Phase 1's isolated fit (a≈2.65–3.9e-5,
  b≈1.34e-7) remains the trustworthy `c(work)` primitive.
- **`rtt_s`: fit to 0.0.** `cost_model.py`'s fitter computes
  `rtt_s = max(0.0, mean(distributed_tick − matched_local_tick))` over 13
  matched (N,work) pairs — floored at zero because RTT/comms latency is
  defined as non-negative in the plan's cost model (`T_tick = compute +
  sync + comms`, comms ≥ 0). Since **every** matched pair here has
  distributed *faster* than local (see the per-tick deltas in §4a), the raw
  mean delta is negative and gets clipped to the floor. This is correct
  behavior for the model's semantics, not a bug — but it means the fitted
  `rtt_s=0.0` should be read as **"no measurable positive LAN/SSH latency
  is discernible from this dataset, because it is masked by a larger
  negative contention effect"**, not as "RTT is actually zero." The
  manually-computed raw deltas (not clipped) are: N=48 work=1:
  −0.000002s, work=5,000: −0.000260s, work=50,000: −0.001741s; N=64
  work=50,000: −0.009884s; N=88 work=50,000: −0.021948s — all negative,
  growing in magnitude with N. A genuine positive LAN RTT almost certainly
  exists (this is a real network hop) but this dataset cannot isolate it
  from the much larger manager-contention effect running the opposite
  direction.

`make_report.py --bench phase4_combined.csv --params phase4_fit_params.json`
ran cleanly (4/5 plots; crossover skipped, no `mode=par` rows in this
phase by design — see Optional section below).

## Optional parallel_execution variant

Skipped. Per the task's own escape hatch ("only if time permits, skip if
unsure") — Phase 1 already isolated the seq/par crossover cleanly on this
same hardware, and adding a `mode=par` cell here would mix a second
new axis into a phase whose actual finding (the contention confound) was
already unambiguous from the `seq` sweeps; not needed to answer Phase 4's
question and would cost another matrix + fit + report round-trip for a
result Phase 1 already covers.

## Verdict

- **Does distribution ever pay off on this hardware? Yes — but not for the
  reason the task hypothesis (or `generate_scale_benchmark.py`'s prior
  note) expected.** Every tested (N, work) cell had `distributed_nat`
  faster than `local`, including at trivial work=1 (sync-bound, ratio
  ≈0.995 — essentially a wash, as expected there). The effect strengthens
  with both higher work (4a) and higher N (4b), reaching a best observed
  ratio of **0.548 at N=88, work=50,000** — beating even the idealized
  1.571x/0.637 cores-only roofline.
- **This is a contention artifact of a shared, non-idle manager, not
  evidence that the theoretical roofline is wrong or beatable in
  general.** `uptime`/`who` confirm the manager was carrying a load
  average of ~67 from two other users' long-running sessions during this
  entire phase, while both remotes sat idle (load average ~1.5–2.0). The
  1.571x/0.637 ceiling is only a hard ceiling when the local machine is
  otherwise idle and every core is available to the job — that premise
  was silently false for every run in this phase.
- **On a truly idle 112-core manager, the honest expectation (stated in
  the task brief) still holds**: with N ≤ 112, none of these federate
  counts would oversubscribe the manager's own cores, so a contention-free
  local run should track close to the sync-bound/compute-bound curve
  `generate_scale_benchmark.py` originally measured (~1.0 at low work,
  approaching but not beating 0.637 as work grows) — this phase cannot
  distinguish "distribution genuinely beats local due to compute-bound
  work" from "distribution wins because it dodges contention," because on
  this specific machine, right now, both are true at once and the
  contention term dominates.
- **What would have to change to see the *intended* experiment (a clean
  compute-bound roofline approach, not a contention artifact)?**
  1. Run at a time (or on a machine) with no other tenants — re-run 4a/4b
     when `uptime`'s manager load average is ≈0, to isolate the pure
     compute/sync/comms terms the plan's cost model assumes.
  2. More cores off-manager relative to the manager itself would raise the
     *theoretical* ceiling above 1.571x, but does nothing to fix the
     contention confound seen here (which flows from *background* load on
     the manager, unrelated to our own federate count).
  3. Higher federate counts once the Phase-2 SSH-ControlPath 108-byte bug
     is fixed (N≥112) would let this phase's design push N past the SSH
     cap and see whether the ratio keeps improving with N indefinitely (as
     the contention story predicts, since more of our own federates start
     competing with the *background* load too) or plateaus once the
     manager's *own* core budget is exceeded by our federates specifically
     (a genuinely different, architectural roofline effect the current
     N<112 design cannot reach).
  4. Heavier per-step compute (work well past 50,000) would still be
     useful to separate "compute-bound distribution" from "contention
     dodge" — if the ratio keeps tracking with N rather than saturating
     near 0.637 once work is pushed very high at fixed N, that is further
     evidence the effect here is contention-driven, not roofline-driven.

## Total wall-clock

4a: run_bench.py console timestamps ≈35s for all 18 runs (setup dominates —
sim_wall_s per run is sub-second). 4b: ≈25s for 8 runs. Fit + report
generation: a few seconds. **Total this phase: well under 2 minutes of
actual scenario execution**, plus investigation/write-up time.

## No harness bugs found this phase

`run_bench.py`, `gen_scenario.py`, `cost_model.py`, `make_report.py` all
behaved as documented. The `rtt_s` floor-at-zero behavior (above) is
correct model semantics, not a bug, and is called out for interpretive
clarity rather than as a defect report.

## Files in this directory

- `phase4_distribution.md` — this report
- `phase4a_distribution_overhead.csv`, `phase4b_roofline_approach.csv` — the
  two designed sweeps
- `phase4_combined.csv` — concatenation of the two, feeds the fit/report
- `phase4_fit_params.json` — `cost_model.py fit` output (see rtt_s caveat
  above)
- `phase4_ratio_plot.png` — custom 2-panel plot: dist/local ratio vs work
  (4a) and vs N (4b), both with break-even (1.0) and roofline-ceiling
  (0.637) reference lines — the primary visual for this phase
- `03_roofline.png` — the generic D5 roofline plot (throughput vs total
  instances by placement), copied per task instructions; see the caveat
  above on why `phase4_ratio_plot.png` is more legible for this phase's
  question
