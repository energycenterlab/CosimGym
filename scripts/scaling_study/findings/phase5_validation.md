# Scaling Study — Phase 5 (framework validation + bounded max-stable scan)

Manager = ipazia (112c), shared with co-users `bottaccioli`/`montaldo`. Harness
used as-is (`gen_scenario.py`, `run_bench.py`, `cost_model.py`,
`make_report.py`) — no harness code changes made in this phase. Both
previously-reported harness bugs (SSH-ControlPath 108-byte overflow at
N≥112, `gen_scenario.py` zmq port-1-apart stride collision at F≥2) are
confirmed **fixed** in the current tree (`ScenarioManager._setup_remote_execution`
now builds `ControlPath` under a short `tempfile.gettempdir()/cosim-ssh/<8-char
hash>` path instead of the long `logs/<scenario_name>/...` tree;
`gen_scenario.py:282` strides federation broker ports by 10, not 1) — see §B
below, which pushes distributed federate count to 200 with zero SSH failures.

## Safety budget used (read before any run)

`free -g` on the manager at the start of this phase:

```
total 251G / used 6G / free 152G / available 242G
```

`uptime` at the same moment: `load average: 0.45, 7.39, 20.95` (light —
well below Phase 4's ~67, decaying from an earlier spike).

**Hard budget = 40% of free RAM = 0.4 × 152 GB = 60.8 GB.** At ≈300 MB/federate
(Phase 0d rss), that is a ceiling of **≈207 federate processes resident on
the manager at once**, which this phase never approached (max manager-local
peak_rss measured: 17.9 GB at N=200 distributed, §B — 29% of the 60.8 GB
budget, 71% headroom unused). cloud1/cloud5 were confirmed idle
(`load average` ≈0.01–1.7, ≈108–111 GB free of 125 GB) before every
distributed run.

**Observed but not a violation**: `uptime`'s 1-minute load average spiked
transiently to 80–116 during/just after the heavier runs (Task A's N=96×2
sweep, Task B's N-sweep), then decayed back down within 1–2 minutes with
`ps` showing no lingering high-CPU or leaked federate/broker processes.
This is consistent with Linux's load-average counting many short-lived
fork/exec bursts (our own rapid spawn-and-teardown of dozens to hundreds of
federate/broker/ssh-control-master processes per run), not a sustained
CPU/memory squeeze — `free -g`'s **free/available columns never moved**
(152 GB / 242 GB, constant through the whole phase) in any measurement
taken during this phase. No run was throttled or capped because of this.

---

## Task A — Framework validation

### A.1 Fitted params used

Two params files were produced:

- `phase5_pooled_fit_params.json` — `cost_model.py fit` over a straight
  concatenation of `bench_all.csv` (phase0+1) + `phase3_combined.csv` +
  `phase4_combined.csv` (130 rows). Reproduces the same DoE-collinearity
  artifact flagged in `phase01.md`/`phase4_distribution.md`: pooling
  single-axis sweeps together gives a slightly negative `s1[zmq]`
  (-3.48e-4) and an inflated `c` intercept — **not used for prediction**,
  kept only as the contractual "fit over everything" artifact.
- **`phase5_clean_fit_params.json` (used for all predictions below)** —
  hand-assembled from the *isolated* single-axis fits, matching the
  methodology phase0/1/2 already used: `c(heavy_compute_dummy)` from
  `bench_phase0b.csv` (work-only sweep, N=1,M=1): `a=1.5103e-5,
  b=1.3437e-7`; `s(zmq)` from `bench_phase0a.csv` (N-only sweep):
  `s0=6.4762e-5, s1=2.6471e-6`; `s(zmq_ss)` from
  `phase2b_distributed_nat_zmqss.csv` (N-only sweep, distributed_nat — so
  its s0/s1 already include typical LAN/SSH cost for that placement, hence
  `rtt_s=0` to avoid double-counting); `O_par=0.04393` from
  `bench_phase1.csv`'s isolated fit.

**Sanity check vs. the task-quoted primitives**
(`c=1.6e-6+1.34e-7·work`, `s0=1.3e-4/s1=2.65e-6`, `O_par≈0.039`): `b` and
`s1` reproduce almost exactly (1.3437e-7 vs 1.34e-7; 2.6471e-6 vs 2.65e-6);
`O_par` is 13% high (0.0439 vs 0.039, inside phase01's own 0.033–0.046
spread from two independent estimates); `s0` is ~2× lower here (6.48e-5 vs
1.3e-4 quoted) and `c`'s intercept `a` is ~10× higher (1.51e-5 vs 1.6e-6
quoted) — both flagged, and both negligible in practice: `a << b·work` for
any `work` above a few hundred, and `s0`'s absolute difference (~6.5e-5 s)
is far smaller than the errors reported below, so neither discrepancy
changes any conclusion.

### A.2 Predicted vs measured — 4 held-out configs

None of these (F,N,M,mode,W,core_type,work,placement) combinations appear
in any prior phase's training sweeps. Ticks=30, repeats=3,
`scripts/scaling_study/matrices/phase5a_validation_holdout.yaml` →
`phase5a_validation_holdout.csv`. **12/12 PASS**, no failures.

Load at test time (checked immediately before the run): `load average: 4.80,
4.98, 17.36` — light, comparable to the calibration runs, i.e. this is not
a repeat of Phase 4's ~67-load confound; the manager was reasonably free of
other-user contention during this specific sweep (though see the
self-inflicted transient noted above, which straddles this run and the
next one).

| # | config | predicted T_sim (s) | measured sim_wall_s median (s) | rel. error | region |
|---|---|---|---|---|---|
| 1 | local seq, F1 N24 M1 work=12000 | 0.0527 | 0.1396 | **+165%** | mid-crossover (N moderate + nontrivial work) |
| 2 | local par, F1 N1 M12 W4 work=30000 | 1.6841 | 1.2514 | **−25.7%** | above crossover, par should (and does) win |
| 3 | local seq, F1 N56 M1 work=1 | 0.00685 | 0.00800 | **+16.9%** | sync-dominated (pure N, near-zero compute) |
| 4 | distributed_nat, F1 N64 M1 work=50000 | 0.2074 | 0.6783 | **+227%** | compute-heavy + N=64 spread over 3 machines |

### A.3 Verdict — does the framework predict?

**Mixed, and the pattern is informative, not just noisy.** The two configs
that vary **only one axis away from the training region** (config 3: pure
N at trivial work, matches how `s(N)` was fit; config 2: an M/W crossover
point close in kind to Phase 1's own crossover sweep) predict well —
+16.9% and −25.7%, both within the kind of spread seen inside individual
phases' own repeat-to-repeat variance. The two configs that combine a
**nontrivial per-instance `work` together with a nontrivial `N`** (configs
1 and 4) are wrong by **+165% and +227%** — the model systematically
*under*-predicts, never over-predicts, in this region.

This is exactly the gap the DoE caveat in phase01.md/phase4_distribution.md
predicted would eventually bite: every training sweep in Phases 0–4 varied
`N` **or** `work` **or** `M`, never two of them together in the same
matrix. The additive model `T_tick = compute(work,M) + sync(N)` has no
cross term, so it has no way to represent whatever actually happens when
many federates are simultaneously doing substantial per-tick compute —
plausibly increased scheduling/GC jitter across dozens of concurrently
CPU-busy federate processes, and/or HELICS's time-advance barrier being
gated by the *slowest* of N busy federates rather than a fixed per-tick
constant (a tail-latency effect that grows with N once each federate's own
per-tick compute has enough variance to matter). The recommend() check in
A.4 below independently reproduces the same signature at N=96,
work=20000 (+242–250%), confirming this is a systematic region gap in the
model, not a fluke of these 4 specific cells.

**Practical implication for `recommend()`**: predictions are trustworthy
near the conditions they were fit under (pure-N sweeps, pure-work sweeps,
crossover regions at fixed N), but should be treated as a **lower bound**,
not a point estimate, once a config combines high N with high per-instance
work — by a factor that grew to ~2.3–3.3× in the cells tested here. A
follow-up fix (out of scope for this "run and interpret" phase, reported
per instructions) would add an `N × c(work)` interaction term, fit from a
matrix that varies N and work together (which no existing matrix in
`scripts/scaling_study/matrices/` does).

### A.4 `recommend()` validation

Target: 96 instances of `heavy_compute_dummy`, work=20000, n_ticks=30,
tick_budget_s=2.0 (`scripts/scaling_study/machines.example.json`).

```
recommended: F=1, N=96, M=1, mode=seq, core_type=tcp, placement=local
predicted:   compute_s=0.002703 sync_s=0.0 T_tick_s=0.002703 T_sim_s=0.081075
verdict:     MEETS budget (2.0s)
```

**Ran it** (`phase5a_recommend_check.csv`, 3 repeats, plus a `core_type=zmq`
twin for comparison, 6/6 PASS):

| core_type | predicted T_sim | measured sim_wall_s median | rel. error |
|---|---|---|---|
| tcp (recommended) | 0.0811 | 0.2840 | **+250%** |
| zmq (comparison) | 0.0811 | 0.2771 | **+242%** |

**Two findings here.** (1) The absolute prediction is off by the same
~2.4–2.5× under-prediction seen in A.2/A.3 (N=96 + work=20000 is exactly
the "high N + high work" region the additive model misses) — but the
**budget check itself was not misled**: 2.0s of headroom over a true
~0.28s run is wide enough that "MEETS budget" is still the right call even
with a 2.5x model error. A tighter budget (anywhere under ~0.3s) would have
flipped this incorrectly, since the recommender's own confidence interval
implicitly assumes the additive model is accurate. (2) `recommend()`
**picked `tcp` over `zmq` for a reason that isn't real**: `s(tcp)` has *no
fitted data at all* in `phase5_clean_fit_params.json` (no tcp rows exist in
any training sweep) and defaults to `s0=s1=0`, making tcp look
(marginally) cheaper than zmq's small nonzero sync term purely because of
a data gap, not a measured performance difference. The actual run
confirms this: tcp (0.2840s) and zmq (0.2771s) measured **statistically
indistinguishable** (zmq if anything measured *faster*, well within
repeat-to-repeat noise) — `recommend()`'s tcp-over-zmq choice here is an
artifact of an unmeasured core_type defaulting to zero cost, not a real
recommendation to prefer tcp. Flagged as a gap in the *params data*, not a
bug in `recommend()`'s search logic (which did exactly what the contract
says: minimize predicted T_sim over the candidates it evaluated, 1504
of them).

---

## Task B — Bounded max-stable scan

### B.1 Federate-count axis (distributed_nat, the two now-fixed bugs)

`scripts/scaling_study/matrices/phase5b_max_stable_N.yaml` →
`phase5b_max_stable_N.csv`. `F=1, M=1, mode=seq, core_type=zmq_ss` (forced),
`heavy_compute_dummy`, work=1, ticks=30, N ∈ {112, 150, 200}, 2
repeats/cell, split core-proportionally 112:32:32 across
manager/cloud1/cloud5 by `gen_scenario.py`. **6/6 PASS — no failures at any
N, including N=200** (past the old N≥112 SSH-ControlPath ceiling that
Phase 2 diagnosed and this session confirms is now fixed).

| N (total feds) | ~feds on manager | manager peak_rss_mb (local process-tree) | setup_s (median) | sim_wall_s (median) |
|---|---|---|---|---|
| 112 | ~71 | 10,108 MB | 6.08 s | 0.0104 s |
| 150 | ~96 | 13,464 MB | 7.25 s | 0.0177 s |
| **200** | **~127** | **17,903 MB** | 9.16 s | 0.0209 s |

(`peak_rss_mb` here is `run_bench.py`'s psutil sample of the **local**
process tree only — brokers + manager-resident federates + the
ScenarioManager parent; it does not see the remote cloud1/cloud5 federate
processes, which is why the per-federate implied cost, ~140 MB, reads
lower than Phase 0d's ~304 MB/idle-federate baseline: this column
undercounts a distributed run's true total RSS by design, not a
regression.)

**Max stable total federate count reached: 200** (N=200 distributed_nat).
Resource headroom at that point: manager peak local RSS 17.9 GB vs. the
60.8 GB safety budget (**71% of budget unused**); manager `free`/`available`
RAM was unchanged (152 GB / 242 GB) before and after; cloud1/cloud5 stayed
at ≈108–111 GB free throughout. Not pushed further per the task's explicit
`N∈{112,150,200}` bound and the "bounded, not to-failure" safety directive
— **200 is where this scan stopped by design, not where the framework
failed.**

**Theoretical memory ceiling** (separate from what was actually run): at
the ≈300 MB/federate Phase-0d rate and the 60.8 GB budget, the manager
alone could in principle host ≈207 federates before crossing the 40%-of-free
guardrail; distributing 60.8 GB of budget core-proportionally across all
three machines (manager 112c + cloud1 32c + cloud5 32c = 176c, each
machine's own RAM budget is far larger — 125 GB machines vs. the 251 GB
manager) would allow on the order of several hundred to ~1,000+ federates
total before any single machine's *own* 40%-of-its-free-RAM budget was
exceeded — this was not tested (out of the task's explicit N-range) and
should be read as a back-of-envelope ceiling, not a validated one.

### B.2 Instances-per-federate axis (local, M sweep)

`scripts/scaling_study/matrices/phase5b_max_stable_M.yaml` →
`phase5b_max_stable_M.csv`. `F=1, N=8, mode=seq, core_type=zmq`,
`heavy_compute_dummy`, work=1, ticks=30, M ∈ {50, 100, 200}, 2
repeats/cell, local (manager only, 8 federate processes total). **6/6
PASS.**

| M | total instances (N×M) | sim_wall_s (median) |
|---|---|---|
| 50 | 400 | 0.0348 s |
| 100 | 800 | 0.0502 s |
| 200 | **1,600** | 0.0885 s |

Confirms phase0d's finding that `heavy_compute_dummy`'s RSS-per-instance is
~0 (base process footprint dominates) — 8 federate processes regardless of
M means this axis never touches the memory budget at all; the constraint
here is purely "does the sequential per-federate stepping loop still work
correctly at M=200", which it does, cheaply (<0.1s sim time for 30 ticks).

### B.3 Combined max-scale headline

- **Max total federate processes reached, stably: 200** (distributed_nat,
  §B.1) — 71% of the memory safety budget unused at that point.
- **Max total model instances reached, stably: 1,600** (N=8×M=200 local,
  §B.2) — instances scale independently of federate count and are far
  cheaper per-unit for this trivial-RSS model.
- No run in either sweep failed, timed out, or showed `failure_mode` set.
- Did not have to cap below the task's stated plan (N up to 200, M up to
  200 were both explicitly in-scope and both completed) — the co-user load
  present at session start (~20 avg, decaying) never approached a level
  that required capping scale below what was asked for.

---

## `make_report.py`

`results/scaling/phase5_combined_all.csv` (161 rows: all of phase0/1/3/4's
pooled data + this phase's A/B CSVs) + `phase5_clean_fit_params.json` →
5/5 plots rendered. `05_predicted_vs_measured.png` (copied to this
directory) plots all 157 rows with both a valid prediction and a measured
`sim_wall_s`; the bulk of points (mid-to-high T_sim, from the original
per-axis-clean training sweeps) sit close to the y=x line, while the
under-prediction signature from A.2–A.4 is visually present but compressed
near the plot's origin at this pooled dataset's scale — the per-config
table in A.2 is the more legible read of that specific effect.

## Files in this directory (this phase)

- `phase5_validation.md` — this report
- `phase5_clean_fit_params.json` — the per-axis-clean params used for every
  prediction above (see A.1)
- `phase5_pooled_fit_params.json` — the contractual `cost_model.py fit`
  pooled-CSV output (not used for predictions; DoE-collinear, see A.1)
- `phase5a_validation_holdout.csv` — the 4 held-out configs, 3 reps each
- `phase5a_recommend_check.csv` — the recommend()-returned config + zmq twin
- `phase5b_max_stable_N.csv`, `phase5b_max_stable_M.csv` — the two Task-B
  sweeps
- `05_predicted_vs_measured.png` — D5 plot over the full pooled dataset
- `scripts/scaling_study/matrices/phase5a_validation_holdout.yaml`,
  `phase5a_recommend_check.yaml`, `phase5b_max_stable_N.yaml`,
  `phase5b_max_stable_M.yaml` — the matrices used (kept for reproducibility,
  matching prior phases' convention)

## Total wall-clock

Sum of `setup_s + sim_wall_s` across all 30 runs this phase: **≈123 s**
(~2 minutes of actual scenario execution). Elapsed wall-clock for the whole
phase (reading prior findings, fitting, running, reporting, cleanup):
**≈15 minutes**.

## No new harness bugs found this phase

Both previously-reported bugs (SSH ControlPath, zmq port stride) are
confirmed fixed and were exercised directly (N=200 distributed_nat; no F≥2
zmq cell was re-tested here since Phase 3 already isolated that one to
`core_type=zmq_ss` locally). The `recommend()` tcp-vs-zmq artifact in A.4
is a **params-data gap** (no tcp training rows), not a code defect in
`cost_model.py`'s search logic.
