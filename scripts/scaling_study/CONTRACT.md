# Scaling-Study Harness — Interface Contract (agents MUST obey)

> This file is the **machine-facing** spec: knob names, CSV columns, JSON shapes.
> For prose you can actually read, see:
> - **`EXPERIMENTS.md`** — what each experiment asked, what ran, what came out,
>   which files it used and produced. No networking background assumed.
> - **`RUNBOOK.md`** — copy-pasteable commands to re-run or modify anything,
>   in order, with cleanup and troubleshooting.

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

### Part-B / Phase-D additions — data-exchange wiring knobs

Added 2026-07-28 for the data-exchange study (plan §3 Phase D, §8.3). Every knob
below is OPTIONAL; omitting all of them (or `exchange=none`) MUST reproduce the
Part-A behaviour byte-for-byte (self-contained federations, `subscribes: []`).

| knob        | type            | meaning |
|-------------|-----------------|---------|
| `exchange`  | "none" \| "on"  | wiring off (default, Part-A) vs on |
| `distance`  | "intra_fed" \| "cross_fed" \| "cross_machine" | where subscribers live relative to publishers (empty when `exchange=none`) |
| `fanout`    | "1to1" \| "1toN" \| "Nto1" \| "all2all" | edge pattern between publisher/subscriber federates |
| `msg_width` | int             | published payload vector length (doubles); 1 = scalar-equivalent |
| `freq`      | int             | publisher emits every `freq`-th tick (1 = every tick) |
| `causality` | "same_step" \| "next_step" | subscription on the tick's critical path vs offset-absorbed |
| `n_edges`   | int (DERIVED)   | total HELICS input→target links in the scenario; computed by the generator, not passed by the user |
| `n_subs`    | int (DERIVED)   | total HELICS input HANDLES registered (`M` × number of subscriber federates). Differs from `n_edges` whenever an input carries >1 target (`Nto1`, `all2all`) — the pair separates per-subscription polling cost from per-edge transfer cost |
| `max_fed_in`  | int (DERIVED) | `M` × targets on the BUSIEST subscriber federate — inbound links the gating federate polls per tick |
| `max_fed_out` | int (DERIVED) | `M` × subscriber federates pointing at the BUSIEST publisher federate — outbound fan-out its core services per tick |

**Why BOTH totals and per-federate counts are needed (measured, Phase D 2026-07-28).**
Two cost mechanisms coexist and a fit needs a regressor for each:

- **`n_edges` — shared routing cost, dominant.** Every wired link costs ~3.9 µs/tick
  regardless of where it attaches. Decisive evidence: at N=16/M=4 cross_fed, `Nto1`
  and `all2all` have the *same* `max_fed_in` = 64 but 64 vs 1024 edges, and cost
  **+234 µs vs +5383 µs** — 23×, tracking edge count, not per-federate load.
- **`max_fed_in` — the gating federate's own polling loop, ~2.1 µs/link.** Visible
  only once edge count is held fixed: at *equal* 512 edges, N=4/M=64 (concentrated,
  `max_fed_in` 128) costs **3541 µs** while N=16/M=4 (spread, `max_fed_in` 32) costs
  **1124 µs**.

> **Do not fit on a narrow matrix.** An earlier 27-cell matrix with N pinned at 4
> made `n_edges` and `max_fed_in` collinear and produced the opposite conclusion
> (per-federate placement dominant, R² 0.73 vs 0.53). The wide matrix reversed it:
> per-federate-only scores R² 0.22 where totals score 0.97. Vary N **and** M
> independently or the two regressors cannot be told apart.

`max_fed_out` is retained as a diagnostic but added nothing beyond `max_fed_in` in
the wide fit.

**Wiring rules (locked — the generator must produce exactly this):**
- Target strings use the **flat global** HELICS namespace `<federate_name>.<instance_id>/<pub_key>`
  for BOTH intra- and cross-federation edges. CosimGym registers every publication
  with `register_global_publication` as `<federate>.<instance>/<key>` (no federation
  prefix; see CLAUDE.md "Multi-Federation Scenarios" and `simple_test_multifederations.yaml`).
- **Bipartite by construction.** Every wiring splits federates into a disjoint
  publisher side `P` and subscriber side `S`; a federate is never both. This is not
  cosmetic: `ScenarioManager._validate_causality_cycles()` **raises RuntimeError** on any
  `same_step` dependency cycle (non-iterative HELICS time requests cannot resolve one),
  so a naive all-to-all with `causality: same_step` would abort before tick 1. Bipartite
  wiring is acyclic for every pattern, which keeps `causality` an independently
  sweepable knob instead of one forced by the topology.
  - `intra_fed` : within each federation, `P` = federates `[0, N//2)`, `S` = `[N//2, N)`.
                  Requires `N >= 2` (and even `N` for a balanced split).
  - `cross_fed` : for `f = 1..F-1`, `S` = all federates of federation `f`,
                  `P` = all federates of federation `f+1` (no wrap → no cycle). Requires `F >= 2`.
  - `cross_machine` : same federate-level patterns as `cross_fed`, but placement puts
                  `P` and `S` on different machines (distributed placements only —
                  HARD-GATED, plan §8.7).
- **Instance-paired**: subscriber instance `j` targets publisher instance `j` only.
  Edge count therefore scales linearly in M, not M². With several publisher federates
  per subscriber (`Nto1`, `all2all`), the per-instance target list holds one entry per
  publisher federate and `multi_input_handling: sum` is set.
- Per-instance targets are emitted with the dict form `targets: {"0": [...], "1": [...]}`
  keyed by instance index (the form `BaseFederate._register_subs` resolves per instance).
- Federate-level patterns over `(P, S)`, `p = len(P)`, `s = len(S)`:
  - `1to1`   : `S[k] <- P[k mod p]` — federate-edges = `s`.
  - `1toN`   : every `S[k] <- P[0]` (broadcast) — federate-edges = `s`.
  - `Nto1`   : `S[0] <- every P` (aggregation, `multi_input_handling: sum`) — federate-edges = `p`.
  - `all2all`: every `S[k] <- every P` — federate-edges = `s·p`.
  With the balanced `intra_fed` split (`s == p == N/2`), `1to1`/`1toN`/`Nto1` all yield
  the SAME federate-edge count — deliberate, so the sweep isolates fan-out SHAPE from
  edge COUNT; `all2all` is the edge-count regressor.
- `n_edges` = `M` × Σ over subscriber federates of (number of targets per instance),
  written into `<out>.spec.json` and the bench CSV.

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

**Phase-D extension (2026-07-28):** seven exchange columns are APPENDED, in this
order, after `throughput_inst_steps_s` — appended, never inserted, so every Part-A
CSV stays readable and old readers (which key by header) are unaffected:
`exchange,distance,fanout,msg_width,freq,causality,n_edges,n_subs,max_fed_in,max_fed_out`
For a Part-A-style run they are `none,,,1,1,,0,0,0,0`.

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
  "comms": {             // Phase-D (2026-07-28): data-exchange cost term
    "per_edge_s":    {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
    "in_per_link_s": {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
    "per_byte_s": 0.0          // marginal cost per byte routed per tick
  },
  "notes": "str"
}
```
`predict()` adds, for a wired config:
```
comms = per_edge_s[distance]    · n_edges           # shared routing, dominant
      + in_per_link_s[distance] · max_fed_in        # gating federate's poll loop
      + per_byte_s · (8 · msg_width · n_edges / freq)
```
Fitted by **weighted** least squares (weight `1/max(delta, 20 µs)`): deltas span
50 µs → 22 ms, so an unweighted fit would model only the largest cells and ignore
everything below ~1 ms.

> **Two superseded shapes** — recorded because the sequence is the lesson.
> (1) `per_edge_s·n_edges + per_byte_s·bytes + fixed_per_sub_s·n_subs`.
> (2) `fixed_per_tick_s + in_per_link_s·max_fed_in + out_per_link_s·max_fed_out + per_byte_s·…`,
> adopted after a narrow (N=4-only) matrix appeared to show placement dominating —
> a collinearity artifact, reversed by the wide matrix (R² 0.22 vs 0.97).
> The current shape keeps a term for each mechanism. Params files in either older
> shape still load; missing keys contribute zero.

The `comms` block is OPTIONAL — a params file fitted before Phase D (no `comms` key)
MUST still load, with the term treated as zero.

## D6 — stress driver (added Phase D, 2026-07-28)

`scripts/scaling_study/stress_ramp.py` — climbs ONE axis until CosimGym actually
breaks, then stops. Distinct from `run_bench.py`, which runs a fixed matrix and
keeps going after a failure; that is wrong for a stress test, because every cell
above the ceiling is both uninformative and the heaviest run in the file.

CLI: `--axis M|N --start N --factor F --steps K [--exchange on|none] [...knobs] --out <csv>`
- Reuses `run_bench.bench_one`, so a stress row and a matrix row share the D2 CSV schema.
- **Guards, re-checked immediately before every rung** (plan §8.7): `--guard-free-pct`
  (default 40, abort if free RAM drops below it) and `--guard-load` (default = core
  count). A guard abort is recorded as such and is NOT reported as a framework
  failure — conflating "out of permission to continue" with "CosimGym broke" is how
  Phase 4 produced a confounded result.
- Stops at the FIRST row with a non-empty `failure_mode`, and prints the max stable
  value for the ramped axis.

## D5 — report/plots

`scripts/scaling_study/make_report.py` reads bench.csv + fitted params → PNGs + a markdown/HTML report: crossover curve (seq vs par), sync curve s(N), roofline, ceiling-vs-network, predicted-vs-measured.

## Rules for all agents
- Conda env `cosim_gym`. Run from repo root.
- Do NOT run large/real-hardware experiments — build + smoke-test on tiny configs (F=1,N=2,M=2,ticks=10) locally only.
- Match existing code style. Keep instrumentation zero-overhead when disabled.
- Report back: files created/changed + how you smoke-tested + any deviation from this contract.
