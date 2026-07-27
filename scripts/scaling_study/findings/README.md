# Scaling Study — canonical index (single source of truth)

This file is **the** map of the CosimGym scaling study. Every result lives in a
per-phase leaf under this directory; this index says what's done, what's trusted,
and what's still open. If another doc disagrees with this one, this one wins —
report the discrepancy rather than trusting the other.

- **Plan / forward-looking:** `docs/future_and_TODOs/scaling_study_plan.md`
- **Harness (how to run):** `../CONTRACT.md` + `gen_scenario.py`, `run_bench.py`,
  `cost_model.py`, `make_report.py`, `plot_crossover_clean.py`,
  `plot_instance_crossover.py`
- **Paper prose:** `paper_ready_sentences.md`
- **Bottleneck catalog:** `bottlenecks.md`
- **Cross-phase roll-up (narrative):** `all_phases_synthesis.md`

> **Scope caveat that colours every result below:** all controlled sweeps used
> `heavy_compute_dummy` with **zero data exchange** (self-contained federates, no
> subscriptions). The multi-entity axes were characterised for *empty* federates —
> broker/sync/compute overhead only. **Data-exchange coupling has not been studied
> yet** (it is the next study). Distribution numbers (Phase 4) are additionally
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

## Trusted primitives (use these; ignore the pooled fits)

Per-axis isolated fits only (pooled `fit_params.json` is DoE-collinear — do not use):
- `c(heavy_compute_dummy) = 1.51e-5 + 1.34e-7 · work` s/instance/tick (slope agreed <1% across two sweeps)
- `c(simple_building) ≈ 6.4e-5`, `c(simple_heatpump) ≈ 6.2e-5` s/tick
- `s(zmq): s0 ≈ 6.5e-5–1.3e-4, s1 ≈ 2.65e-6` s/federate/tick
- `s(zmq_ss): s0 ≈ 1.76e-4, s1 ≈ 2.76e-6` s/federate/tick
- `O_par ≈ 0.044` s/tick (0.033–0.046 across estimates)
- RSS ≈ **300 MB/federate** base; ~0 extra per instance for `heavy_compute_dummy`
- **Crossover law:** parallel wins when `(M − ⌈M/W⌉)·c > O_par`. Cost-crossover `c* ≈ 3.3 ms` (M16/W8). Instance-crossover `M* = O_par/c → ⌈⌉` (≈212 light, ≈15 detailed).
- Canonical params file: `phase5_clean_fit_params.json`.

## Max scale reached (bounded, not to failure)

200 federates (distributed, 1 broker) · 1,600 model instances (N8×M200 local) ·
256 total federates (F8×N32 multifed local) · 1,024 instances in ONE federate
(Phase 1b). Memory (~300 MB/federate) is the true ceiling for real models; the
dummy's ~0/instance RSS means instance count is nearly free.

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

1. **Data exchange never studied** — the next study (taxonomy: topology distance ×
   fan-out × size × frequency; needs a `comms` cost term the model lacks).
2. **Distribution confounded** (Phase 4) — redo on an idle manager.
3. **N×work interaction gap** — additive model under-predicts 2.3–3.3× when both
   are high; fix = a joint N×work calibration matrix (none exists).
4. **Cost model has no F term** — Phase 3 derived `broker_setup_s ≈ -0.070 + 0.369·F`
   by hand; add it as a one-time setup term.
5. **tcp / tcp_ss unmeasured** — `recommend()` can pick them by data-gap artifact.
6. **Fig 09 (staircase) noisy** — clean rerun (60 ticks × 5 reps, idle host) pending.

## Reproduce

```bash
conda run -n cosim_gym python scripts/scaling_study/run_bench.py --matrix <m.yaml> --repeats 3 --out <csv>
conda run -n cosim_gym python scripts/scaling_study/cost_model.py fit <csv> --out params.json
conda run -n cosim_gym python scripts/scaling_study/make_report.py --bench <csv> --params params.json --outdir <dir>
```
Matrices in `../matrices/`. **`run_bench.py` APPENDS to its output CSV** — delete
the target CSV before re-running a matrix, or rows from different runs mix.
