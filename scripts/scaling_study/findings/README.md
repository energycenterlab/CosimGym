# Scaling Study — canonical index (single source of truth)

This file is **the** map of the CosimGym scaling study. Every result lives in a
per-phase leaf under this directory; this index says what's done, what's trusted,
and what's still open. If another doc disagrees with this one, this one wins —
report the discrepancy rather than trusting the other.

- **⭐ Plain-language tour of every experiment (start here if new):** `../EXPERIMENTS.md`
- **⭐ Step-by-step: how to re-run or modify an experiment:** `../RUNBOOK.md`
- **Plan / forward-looking:** `docs/future_and_TODOs/scaling_study_plan.md`
- **Harness schemas (knobs, CSV columns):** `../CONTRACT.md` + `gen_scenario.py`,
  `run_bench.py`, `cost_model.py`, `stress_ramp.py`, `plot_exchange.py`,
  `make_report.py`, `plot_crossover_clean.py`, `plot_instance_crossover.py`
- **Paper prose:** `paper_ready_sentences.md`
- **Bottleneck catalog:** `bottlenecks.md`
- **Cross-phase roll-up (narrative):** `all_phases_synthesis.md`

> **Scope caveat that colours every PART-A result below:** all Part-A sweeps used
> `heavy_compute_dummy` with **zero data exchange** (self-contained federates, no
> subscriptions). The multi-entity axes were characterised for *empty* federates —
> broker/sync/compute overhead only. **Phase D (2026-07-28) closes that gap** and
> shows coupling dominates at scale: at 4096 edges the comms term is 22.3 ms/tick
> against a 211 µs unwired baseline. Read every Part-A tick number as a *lower
> bound* for a wired scenario. Distribution numbers (Phase 4) remain
> **confounded by shared-host contention** and need an idle-machine redo.

## Phases done

| Phase | Axis | Leaf | Headline | Trust |
|---|---|---|---|---|
| 0 | primitives | `phase01.md` | fitted c, s(N), O_par, RSS | ✅ |
| 1 | instances — cost crossover (fix M, sweep cost) | `phase01.md`, `01_crossover.png` | par beats seq only above `c* ≈ 3.3 ms` at M16/W8 (work≈24k); law matches to ~4% | ✅ |
| 1b | instances — instance crossover (fix cost, sweep M) | `phase1b_*.csv`, `07/08/09_*.png` | instance-crossover M\*: light model ≈212 (district scale), detailed ≈15; speedup → ceiling W; staircase visible (09 noisy, pending idle rerun) | ✅ (09 ⚠️) |
| 2 | federates (N) + zmq_ss ceiling | `phase2_ceiling.md` | no comms ceiling reproduces (see below); found+fixed SSH-ControlPath + zmq-port bugs | ✅ |
| 3 | federations (F) sharding | `phase3_federations.md`, `06_federation_sharding.png` | hierarchy-broker cost is setup-only (~0.37 s/fed); tick-flat; 256 feds local | ✅ |
| 4 | machines (distribution) | `phase4_distribution.md`, `phase4_ratio_plot.png` | distribution "won" but via shared-host contention — **CONFOUNDED**, needs idle redo | ⚠️ |
| 5 | max scale + framework validation | `phase5_validation.md` | 200 feds / 1600 instances, 0 fail; predictor under-predicts 2.3–3.3× when N and work both high | ✅ / ⚠️ gap |
| **D** | **data exchange (`comms`)** | `phaseD_exchange.md`, figs `10–14` | `comms = per_edge[dist]·n_edges + per_byte·bytes`; **total edge count**, not edge placement, is the regressor (a narrow matrix said the opposite — collinearity); payload free to ~512 B then linear; publishing every 10th tick cuts 90% | ✅ (concentration residual open) |
| **D-x** | **cross-machine κ(LAN)** | `phaseD_exchange.md` §9 | **κ_LAN ≈ κ_local** (ratio 0.99 @M=1, 0.66 @M=4) — a LAN edge costs no more than a local one; distribution is cost-**neutral**. But distributed `zmq_ss` **stalls above ~1 kB/tick** (B12) | ✅ / ⚠️ B12 blocks Phase F |

## Trusted primitives (use these; ignore the pooled fits)

Per-axis isolated fits only (pooled `fit_params.json` is DoE-collinear — do not use):
- `c(heavy_compute_dummy) = 1.51e-5 + 1.34e-7 · work` s/instance/tick (slope agreed <1% across two sweeps)
- `c(simple_building) ≈ 6.4e-5`, `c(simple_heatpump) ≈ 6.2e-5` s/tick
- `s(zmq): s0 ≈ 6.5e-5–1.3e-4, s1 ≈ 2.65e-6` s/federate/tick
- `s(zmq_ss): s0 ≈ 1.76e-4, s1 ≈ 2.76e-6` s/federate/tick
- `O_par ≈ 0.044` s/tick (0.033–0.046 across estimates)
- RSS ≈ **300 MB/federate** base; ~0 extra per instance for `heavy_compute_dummy`
- **Crossover law:** parallel wins when `(M − ⌈M/W⌉)·c > O_par`. Cost-crossover `c* ≈ 3.3 ms` (M16/W8). Instance-crossover `M* = O_par/c → ⌈⌉` (≈212 light, ≈15 detailed).
- **`comms` (Phase D):** `per_edge_s[distance]·n_edges + per_byte_s·(8·msg_width·n_edges/freq)`.
  Fitted (all wired rows, relative-weighted): per-edge `3.14 µs` intra-federation,
  `4.36 µs` cross-federation (local, zmq); `per_byte_s ≈ 1.66 ns/B`; median
  relative error 31%. Payload-free slice alone gives `3.66`/`4.90 µs` per edge
  (fig. 10). A `max_fed_in` term is NOT included — it fits negative on the full
  dataset; the concentration effect is a documented residual (gap #1).
  Params: `phaseD_wide_fit_params.json`.
- Canonical params file: `phase5_clean_fit_params.json` (compute/sync/O_par);
  `phaseD_wide_fit_params.json` (comms).

## Max scale reached

**Phase D (2026-07-28), stressed to first failure, WITH data exchange on** —
supersedes the bounded Part-A figures below:

- **32,768 model instances** (F2 × N4 × M4096 = 8 federates), 65,536 edges,
  329 ms/tick, 3.6 GB, **no failure** — the ladder exhausted its rungs with 97% RAM
  free. The **instance axis has no wall**; cost is linear (`≈0.080 ms·M`) and
  instances are nearly free in memory.
- **128 federates max** (F2 × N64), 16,384 edges, 101 ms/tick, 18.3 GB. At 256
  federates the run **stalls in teardown** (gap #2 / B12) — not OOM, not comms:
  the simulation completes and the run never returns. Memory ≈143 MB/federate.
- Tick time is **quadratic in N** under all-to-all coupling — because `n_edges =
  M·N²`, not because federates are inherently expensive.

Part-A figures (bounded, not to failure, **no data exchange**): 200 federates
(distributed, 1 broker) · 1,600 model instances (N8×M200 local) · 256 total
federates (F8×N32 multifed local) · 1,024 instances in ONE federate (Phase 1b).

## The zmq_ss "ceiling" — resolved

The ~33-federate zmq_ss ceiling in `generate_scale_benchmark.py`'s docstring and
in earlier notes is **not a reliable architectural limit**. It has been observed
as flaky on some days (fails ~70–80/broker) and absent on others (Phase 2: passes
to N=89 distributed; Phase 5: N=200 on one broker) — consistent with a
**transient LAN condition**, not zmq_ss/HELICS itself. The only *deterministic*
failure found was the **SSH-ControlPath AF_UNIX 108-byte overflow at N≥112** — a
harness path-length bug, now **fixed** (`ScenarioManager` builds ControlPath under
a short hashed `/tmp` path). Also fixed: `gen_scenario.py` zmq broker-port stride
(was 1, collided with zmq's `port+1`; now 10).

## Known gaps / open

1. ~~**Data exchange never studied**~~ — **CLOSED by Phase D** (`phaseD_exchange.md`).
   Residual open piece: the **edge-count × concentration interaction**. At equal
   `n_edges = 512`, a concentrated topology (N=4/M=64) costs 3541 µs vs 1124 µs
   spread (N=16/M=4). An additive `max_fed_in` term does not capture it (fits
   negative on the full dataset and is excluded from the shipped model), leaving
   ~31% median relative error — the effect is evidently *multiplicative*. Same
   shape of gap as #3 below; fix with a matrix that crosses `n_edges` ×
   `max_fed_in` systematically (here they crossed only incidentally, 4 of 34 cells).
2. **Teardown stall — simulation completes, run never returns** (`bottlenecks.md`
   B12). Two triggers for one signature: **≥256 federates locally** (plain zmq) and
   **≳1 kB/tick over distributed `_ss`**. One side of the federation reaches the
   final tick and force-disconnects; the other blocks forever. Strands processes
   that then poison subsequent runs with port conflicts. Deterministic,
   undiagnosed, **blocks Phase F** — highest-priority open item.
3. **N×work interaction gap** — additive model under-predicts 2.3–3.3× when both
   are high; fix = a joint N×work calibration matrix (none exists).
4. **Distribution confounded** (Phase 4) — redo on an idle manager. Partly
   addressed by Phase D's cross-machine arm (idle both ends, paired controls):
   distribution measured **cost-neutral**, κ_LAN ≈ κ_local.
5. **Cost model has no F term** — Phase 3 derived `broker_setup_s ≈ -0.070 + 0.369·F`
   by hand; add it as a one-time setup term.
6. **tcp / tcp_ss unmeasured** — `recommend()` can pick them by data-gap artifact.
7. **Fig 09 (staircase) noisy** — clean rerun (60 ticks × 5 reps, idle host) pending.

## Reproduce

```bash
conda run -n cosim_gym python scripts/scaling_study/run_bench.py --matrix <m.yaml> --repeats 3 --out <csv>
conda run -n cosim_gym python scripts/scaling_study/cost_model.py fit <csv> --out params.json
conda run -n cosim_gym python scripts/scaling_study/make_report.py --bench <csv> --params params.json --outdir <dir>
```
Matrices in `../matrices/`. **`run_bench.py` APPENDS to its output CSV** — delete
the target CSV before re-running a matrix, or rows from different runs mix.
