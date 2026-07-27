# Scaling-Study Harness — Interface Contract (agents MUST obey)

Shared schemas so independently-built pieces interlock. Plan: `docs/future_and_TODOs/scaling_study_plan.md`.
All new harness code lives under `scripts/scaling_study/`. Instrumentation lives in `src/core/`.

## Config knobs (the sweep dimensions) — canonical names & types

| knob        | type            | meaning |
|-------------|-----------------|---------|
| `F`         | int             | number of federations |
| `N`         | int             | federates per federation |
| `M`         | int             | model instances per federate |
| `mode`      | "seq" \| "par"  | sequential vs parallel_execution |
| `W`         | int \| null     | max_parallel_workers (null when mode=seq) |
| `core_type` | str             | zmq \| tcp \| zmq_ss \| tcp_ss |
| `model`     | str             | catalog model_name (e.g. heavy_compute_dummy, simple_building) |
| `work`      | number \| null  | model work/cost param (heavy_compute_dummy only) |
| `placement` | str             | "local" \| "distributed_nat" \| "distributed_direct" |
| `n_machines`| int             | machines used |
| `n_ticks`   | int             | horizon in ticks |

## D3 — perf log (ScenarioManager/BaseFederate emit; D2/D4 consume)

ScenarioManager writes ONE JSON file per run at:
`results/<scenario_name>/<sim_id>/perf.json`

Schema (all times seconds, floats):
```json
{
  "sim_id": "str",
  "scenario_name": "str",
  "setup_s": 0.0,            // scenario start -> all federates ready
  "broker_setup_s": 0.0,     // broker(s) launch cost
  "federate_spawn_s": 0.0,   // spawning all federate subprocesses
  "sim_wall_s": 0.0,         // first tick -> last tick (excludes setup)
  "n_ticks": 0,
  "tick_mean_s": 0.0,
  "tick_median_s": 0.0,
  "tick_p95_s": 0.0,
  "failure_mode": null       // null on success; else short string e.g. "lost_comms_-101","oom","timeout"
}
```
Instrumentation must be OFF by default and gated by env var `COSIM_PERF_LOG=1` (zero overhead when unset). Never change existing behavior when unset.

## D1 — parametric generator

CLI: `python scripts/scaling_study/gen_scenario.py --F .. --N .. --M .. --mode .. --W .. --core-type .. --model .. --work .. --placement .. --ticks .. --out <path.yaml>`
- Emits a valid CosimGym scenario YAML honoring all knobs.
- Also emits a sidecar `<path>.spec.json` = the exact knob dict used (canonical names above).
- Reuse patterns from `src/scenarios/generate_scale_benchmark.py` (placement, sync, wiring, causality loop-break). Support all 3 axes + both distributed placements + local.
- `placement=distributed_nat` -> zmq_ss + the 3-machine NAT deployment block; `distributed_direct` -> zmq/tcp + a 2-machine direct deployment block (machine B host is a param `--machine-b-host` etc., default TBD placeholder). `local` -> no deployment block.

## D2 — bench driver

CLI: `python scripts/scaling_study/run_bench.py --matrix <matrix.yaml> [--repeats 3] [--out results/scaling/bench.csv]`
- For each matrix cell: call gen_scenario.py, run scenario in isolated subprocess with `COSIM_PERF_LOG=1`, sample per-process/per-machine RSS+CPU via psutil while running, read `perf.json`, merge → append one CSV row.
- Row schema = ALL knobs (above) + `repeat` + perf.json fields + `peak_rss_mb` (max across sampled procs) + `cpu_util_pct` + `throughput_inst_steps_s` (= F*N*M*n_ticks / sim_wall_s).
- Isolated subprocess per run (like `tests/regression_suite.py`). Timeout per run; on timeout/crash still write a row with `failure_mode` set.
- Matrix file = list of knob-dicts or axis-sweeps expanded to the cartesian product.

## D4 — cost-model fitter + recommender

Module `scripts/scaling_study/cost_model.py`:
- `fit(bench_csv) -> params` : fit c(model,work), s(N,core_type), O_par, RTT from bench rows.
- `predict(config, params, machines) -> T_tick, T_sim` : uses framework in plan §1.
- `recommend(scenario_spec, machines, params) -> best_config` : searches decision vars, returns config minimizing T_sim under constraints (ceiling, RAM).
- CLI: `python scripts/scaling_study/cost_model.py fit|predict|recommend ...`

### Locked CSV schema (D2 emits — D4/D5 consume; canonical column order)
`F,N,M,mode,W,core_type,model,work,placement,n_machines,n_ticks,repeat,scenario_name,sim_id,setup_s,broker_setup_s,federate_spawn_s,sim_wall_s,perf_n_ticks,tick_mean_s,tick_median_s,tick_p95_s,failure_mode,peak_rss_mb,cpu_util_pct,throughput_inst_steps_s`
Note: `perf_n_ticks` = ticks actually run (perf.json); `n_ticks` = ticks configured (knob). `failure_mode` empty on success.

### Locked fitted-params JSON (D4 `fit` writes → D4 `predict`/`recommend` + D5 consume)
Path default `results/scaling/fit_params.json`:
```json
{
  "c": {"<model>": {"a": 0.0, "b": 0.0}},   // per-instance step cost model, e.g. c = a + b*work (heavy_compute_dummy); constant a for others (b=0)
  "s": {"<core_type>": {"s0": 0.0, "s1": 0.0}}, // per-tick sync cost s(N) = s0 + s1*N
  "O_par": 0.0,          // parallel dispatch/IPC overhead per tick (s)
  "rss_per_instance_mb": {"<model>": 0.0},
  "rtt_s": 0.0,          // LAN round-trip added per tick per remote (0 for local-only fit)
  "notes": "str"
}
```

## D5 — report/plots

`scripts/scaling_study/make_report.py` reads bench.csv + fitted params → PNGs + a markdown/HTML report: crossover curve (seq vs par), sync curve s(N), roofline, ceiling-vs-network, predicted-vs-measured.

## Rules for all agents
- Conda env `cosim_gym`. Run from repo root.
- Do NOT run large/real-hardware experiments — build + smoke-test on tiny configs (F=1,N=2,M=2,ticks=10) locally only.
- Match existing code style. Keep instrumentation zero-overhead when disabled.
- Report back: files created/changed + how you smoke-tested + any deviation from this contract.
