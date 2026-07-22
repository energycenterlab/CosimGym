# Paper Case-Study — Run & Reporting Strategy (proposal)

Scope of THIS phase: scenarios written + smoke-tested. Full runs happen only on
explicit go-ahead from the author. This file is the proposed plan for that run.

## Machine
- Host: shared server, **112 cores**. RAM: TBD (record in MANIFEST at run time).
- Docker services (redis, mosquitto, minio) must be up. Single-machine for
  S1–S4b + S5; S4c distributed needs remote hosts (blocked on human input).

## Execution harness
One driver `scripts/paper_case_study/run_all.sh` + a Python profiling wrapper
`scripts/paper_case_study/run_profiled.py` that, for each scenario:
1. records git commit, seed(s), start wall-clock, scenario file;
2. launches `main('<name>')` in an isolated subprocess (same pattern as
   `tests/regression_suite.py`), captures stdout to `results/paper_case_study/logs/`;
3. reads `logs/<name>/execution_metrics.json` if present, else times externally;
4. samples peak RSS (via `resource`/`psutil` on the child process group);
5. appends a row to `results/paper_case_study/exec_metrics.csv`
   (scenario, config-axis, wall_s median±IQR over repeats, peak_rss_mb, exit, marker).

Repeats: timing-sensitive scenarios (S4a, S4c) → **3 reps, report median**.
RL scenarios → **3 seeds (42,43,44)**, mean ± std on metrics.

## Per-stage run plan & est. cost (full horizons)
| stage | scenarios | reps/seeds | rough cost | deliverable |
| --- | --- | --- | --- | --- |
| S1 | cs_s1_baseline | 1 | seconds–min | fig_s1_traces, tab_s1 row |
| S2A | cs_s2_sac, cs_s2_dqn | 3 seeds each | **hours** (100 episodes × 2880) | tab_s2_metrics |
| S2B | cs_s2_reset_{full,rolling,none} | 3 seeds each | **hours** | fig_s2_learning_curves, tab_s2_sample_eff |
| S3 | cs_s3_fmu | — | BLOCKED (see Appendix A.5) | tab_s3_metrics |
| S4a | cs_s4_vert_{seq,par}_N{1,5,10,20,40} | 3 reps | min–tens of min | fig_s4_throughput |
| S4b | cs_s4_topo | 1 | min | log evidence (hierarchy broker) |
| S4c | cs_s4_dist_{1,2,3}m | 3 reps | BLOCKED on remote hosts | fig_s4_machines |
| S5 | cs_s5_dt + feeder | 1 | min (short) | fig_s5_dashboard (human screenshot) |
| S6 | LOC audit script | — | seconds | tab_s6_loc |

**S2 dominates runtime** (SAC/DQN training × 3 seeds). Recommend running S1, S4a,
S4b, S5, S6 first (fast, high-confidence), then launch the long S2 sweep overnight.

## Reporting / analysis scripts (one per deliverable, built after smoke OK)
- `metrics.py` — comfort degree-hours (deadband [19.5,20.5]°C) + energy kWh (∫P_elec). Shared.
- `fig_s1_traces.py`, `tab_s2_metrics.py`, `fig_s2_learning_curves.py`,
  `tab_s2_sample_eff.py`, `tab_s3_metrics.py`, `fig_s4_throughput.py`,
  `fig_s4_machines.py`, `s5_external_feeder.py`, `s6_loc_audit.py`.
- Figures: PDF+PNG 300 dpi. Tables: CSV + rendered .md. All under
  `results/paper_case_study/`. `MANIFEST.md` records provenance per deliverable.

## Open decisions for the author (before full run)
1. **S3 FMU** — pick option (a) restore MinIO FMU object / (b) accept
   variable-names-only diff on `bui0_building_fmu` / (c) defer. See Appendix A.5.
2. **S4c distributed** — provide remote host IPs, SSH access, remote workdir +
   `cosim_gym` env; also decide the CPU-bound scenario design (instr. §S4.9).
3. **S2 training budget** — confirm 100 episodes × 2880 steps is the intended
   convergence budget, or set a smaller paper budget to bound runtime.
