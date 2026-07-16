# Case-Study Experiment Implementation Instructions (Paper: CosimGym SIMPAT)

You are an implementation agent working inside the CosimGym code repository.
Your job: implement and run the six-stage case study (S1–S6) that produces every
figure and table the paper needs. Work stage by stage, in order. Each stage builds
on the previous one by a small config diff — preserving that diff structure is
itself a paper claim, so keep changes minimal and diffable.

## 0. Ground rules (read fully before touching anything)

1. **Verify before writing.** Before creating any YAML, verify every field name,
   model name, and default against this repo's `CLAUDE.md`, `docs/user_guide/`,
   and `src/models/model_catalog/catalog.yaml`. If a graphify knowledge graph
   exists (`graphify-out/`), use `graphify query "<question>"` first. The RL and
   interface config schemas use Pydantic `extra='forbid'` — a single misspelled
   key aborts the run, so exact names matter.
2. **Environment.** All runs: from repo root, conda env `cosim_gym` active,
   Docker services up first:
   ```bash
   docker compose -f src/docker-compose.yaml up -d
   conda activate cosim_gym
   ```
3. **Smoke test first.** For every new scenario, first run a shortened version
   (e.g. 1–2 simulated hours / a few episodes) to confirm it executes cleanly,
   then run the full configuration. Never launch a long run on an untested YAML.
4. **Never fabricate results.** Every number in the deliverables must come from
   an actual run. If a run fails or a metric cannot be computed, record the
   failure in the manifest instead of substituting a plausible value.
5. **Naming and locations.**
   - New scenarios: `src/scenarios/cs_s1_baseline.yaml`, `cs_s2_sac.yaml`,
     `cs_s2_dqn.yaml`, `cs_s2_reset_{full,rolling,none}.yaml`, `cs_s3_fmu.yaml`,
     `cs_s4_vert_{seq,par}_N<k>.yaml`, `cs_s4_topo.yaml`, `cs_s4_dist_{1,2,3}m.yaml`,
     `cs_s5_dt.yaml`.
   - Analysis/plot scripts: `scripts/paper_case_study/` (create it). One script
     per figure/table, plus `run_all.sh` that reproduces everything in order.
   - Outputs: figures (PDF + PNG, 300 dpi) and metric tables (CSV + a rendered
     markdown copy) into `results/paper_case_study/`.
   - `results/paper_case_study/MANIFEST.md`: for each deliverable, record the
     scenario file, git commit hash of the repo at run time, seed(s), run
     command, raw-results path, and generating script. Also record installed
     versions: `python -c "import helics, gymnasium, stable_baselines3; ..."`.
6. **Seeds.** Fixed seeds everywhere. Find where the framework accepts a seed
   (check RL config schema and agent hyperparameters; verify — do not guess).
   If no seed field exists, set library-level seeds inside the run script and
   document it in the manifest. RL comparisons: ≥3 seeds per configuration,
   report mean ± std.
7. **Known hard constraints (source-verified — respect them, do not "fix" them):**
   - RL federates support only `memory_config.sink: json` or `none`;
     `parquet` raises `NotImplementedError`.
   - `parallel_execution` is rejected on RL federates and on federates with
     `override_enabled` — vertical-scaling runs (S4a) must use plain `base`
     federates.
   - `host:` (remote placement) is rejected on `type: rl` federates; the RL
     federate always stays on the manager machine.
   - Implemented RL reset modes are exactly `full`, `rolling`, `none`. There is
     no `soft` or `random` mode — do not attempt them.
   - RL observation/action keys use dot notation
     `<federation>.<federate>.<instance>.<variable>`.
8. **Comfort/energy metrics definition (used in S1–S3):**
   - Comfort violation: degree-hours outside the comfort deadband, computed as
     `sum(max(0, T_lower - T_zone) + max(0, T_zone - T_upper)) * dt_hours` over
     the evaluation horizon. Use the same deadband the reward function uses
     (read it from the reward implementation; document the bounds used).
   - Energy: integral of heat-pump electrical power over the horizon (kWh).
   - Implement once in `scripts/paper_case_study/metrics.py`, reuse everywhere.

---

## S1 — Baseline co-simulation (PID)

**System:** one building zone + heat pump + hourly weather feed + PID controller.

1. Start from the closest existing scenario (`src/scenarios/bui_hp_test_base.yaml`
   or similar — inspect what ships and reuse its wiring). Save as
   `cs_s1_baseline.yaml`.
2. Requirements:
   - Models from the catalog: building (`simple_building` — verify exact key),
     heat pump (`simple_heatpump`), weather feeder (`weather_csv_reader`),
     PID (`simple_pid_controller`). Verify each key in `catalog.yaml` first.
   - Multi-rate on purpose: weather `real_period: 3600`, building / heat pump /
     PID `real_period: 60`. This is a paper claim (multi-rate normalization).
   - Horizon: 48 simulated hours minimum, winter period with meaningful heating
     load (pick a weather CSV shipped in the repo; document which).
   - `memory_config: {attrs: all, sink: json}`.
3. **Deliverables:**
   - `fig_s1_traces`: two stacked panels over time — (top) zone temperature with
     comfort band shaded, (bottom) heat-pump electrical power. X axis in hours.
   - `tab_s1_metrics` row: comfort degree-hours + energy kWh for the PID run
     (this is the baseline row reused in S2/S3 tables).
4. **Acceptance:** run completes; results JSON present under
   `results/<scenario_name>/...`; temperature stays within physically plausible
   range (roughly 0–40 °C); no HELICS deadlock/timeout in logs.

## S2 — RL control + reset-strategy benchmark (LONGEST STAGE)

**Part A — PID → RL swap.**
1. Copy `cs_s1_baseline.yaml` → `cs_s2_sac.yaml`. Remove the PID federate block.
   Add a top-level `reinforcement_learning_config` with the four axes
   (`environment` / `agent` / `run` / `experiment`). Follow
   `docs/user_guide/rl_integration.md` and an existing working RL scenario
   (e.g. `bui_hp_SAC.yaml`) for exact field shapes — copy a known-good structure,
   then adapt keys to the S1 federate/instance names.
   - Observations: zone temperature, outdoor temperature (+ whatever the existing
     SAC scenario observes). Actions: heat-pump modulation/setpoint (whatever
     variable the PID actuated in S1 — keep the same variable).
   - Reward: the shipped comfort/energy reward (verify its dotted path in
     `src/models/model_catalog/RL_agents/` — likely
     `models.model_catalog.RL_agents.reward_functions.comfort_energy_reward`).
   - Agent: `rl_simple_SACsb3`, `backend: stable_baselines3`. Keep
     hyperparameters at backend defaults unless the existing scenario sets them.
   - `run`: online training; choose episodes × episode_length so total steps give
     SAC a fair chance to converge on this task (inspect what existing RL
     scenarios use as a reference; document the choice). Then a deterministic
     test phase on the same horizon as S1's evaluation.
2. Copy → `cs_s2_dqn.yaml`: same everything, agent `rl_simple_DQN` (discrete —
   check the ActionSpec needs `bins` for a discrete action space; follow the
   schema).
3. **Deliverables:**
   - `tab_s2_metrics`: rows PID (from S1), SAC, DQN × columns comfort
     degree-hours, energy kWh, evaluated deterministically on the identical
     horizon and weather. ≥3 seeds for RL rows, mean ± std.

**Part B — reset-strategy benchmark.** Uses `cs_s2_sac.yaml` as base.
4. Three variants differing ONLY in `environment.reset.mode`: `full`, `rolling`,
   `none` (`cs_s2_reset_full.yaml`, etc.). Same seeds, same total training steps.
5. **Deliverables:**
   - `fig_s2_learning_curves`: episode return vs training step, one curve per
     reset mode, mean ± shaded std over seeds.
   - `tab_s2_sample_eff`: steps (or episodes) to first reach a fixed return
     threshold per mode (define threshold as e.g. 90% of the best final return
     across modes; document the definition).
6. **Acceptance:** learning curves show learning (return trend up); the three
   modes produce distinguishable curves; training logs confirm the configured
   reset mode was actually exercised (episode boundaries visible).

## S3 — Model-formalism swap (EnergyPlus FMU)

1. Copy `cs_s2_sac.yaml` → `cs_s3_fmu.yaml`. Change ONLY the building federate's
   model instantiation to the EnergyPlus-derived FMU model (`adelaide_test` —
   verify key and its I/O variable names in `catalog.yaml`).
   - Keep federate and instance names identical so the RL observation/action
     dot-keys stay unchanged — this "agent untouched" property is the paper claim.
   - If the FMU exposes different variable names than `simple_building`, the
     minimal permitted change is the variable segment of the dotted keys;
     record exactly what had to change (the paper reports this diff honestly).
   - Check FMU timing constraints (`docs/user_guide/fmu_models.md`) — EnergyPlus
     FMUs may need a specific `real_period`; adapt and document.
2. **Deliverable:** `tab_s3_metrics`: same columns as `tab_s2_metrics`, SAC on the
   FMU building (+ a PID-or-baseline row on the FMU building if a controller can
   drive it — if not feasible, report SAC only and note why in the manifest).
3. **Acceptance:** run completes with the FMU actually loaded (log line), agent
   trains, `reinforcement_learning_config` diff vs S2 is empty or
   variable-names-only.

## S4 — Scalability (three sub-experiments)

**S4a — vertical (parallel model execution).**
1. Base on the shipped pair `benchmark_parallel_seq.yaml` /
   `benchmark_parallel_par.yaml` (CPU-heavy `heavy_compute_dummy` model).
   **Known bug to fix in your copies:** the shipped pair has mismatched
   `n_instances` (8 vs 20) — copies must be identical except for
   `parallel_execution: true/false`.
2. Sweep `n_instances` ∈ {1, 5, 10, 20, 40} × {sequential, parallel}
   (`cs_s4_vert_{seq,par}_N<k>.yaml`), default `max_parallel_workers`.
   Capture wall-clock per run (the framework writes `execution_metrics.json`
   under `logs/` — verify and use it; otherwise time the run yourself).
   3 repetitions per point, report median.
3. **Deliverable:** `fig_s4_throughput`: wall-clock vs N, two curves (seq/par),
   annotate worker count and CPU core count of the machine.

**S4b — horizontal topology (multi-federation).**
4. `cs_s4_topo.yaml`: extend the S1 system into TWO federations — federation
   `buildings` (≥2 building+HP pairs) and federation `generation` (PV + battery;
   verify PV/battery catalog keys, e.g. in `pv_batt_test_base.yaml`), with at
   least one cross-federation subscription
   (`<federation_name>.<federate>.<instance>/<pub_key>` format).
5. **Deliverable:** no figure — acceptance evidence only: log lines showing the
   auto-inserted hierarchy broker (`--sub_brokers=2`) and dynamically assigned
   ports; note them in the manifest (the paper cites this mechanism).

**S4c — horizontal machines (distributed deployment).**
6. `cs_s4_dist_{1,2,3}m.yaml`: same physical scenario (use the S4b topology or
   an N-building version big enough that distribution matters), plus a
   `deployment:` block per `docs/user_guide/distributed_deployment.md`:
   `manager_address` (LAN IP of this machine — REQUIRED once any federate sets
   `host:`), `machines.<alias>: {host, user, workdir, conda_env}`. Assign
   building federates `host: <alias>` across 1, 2, 3 machines. Brokers, Redis,
   MQTT, and any RL federate stay on the manager — do not try to move them.
7. **Human prerequisites — STOP and ask the user before this sub-stage:** remote
   hostnames/IPs, SSH key access, remote workdir paths, remote `cosim_gym` env
   present. Follow `docs/user_guide/multi_machine_test_walkthrough.md`.
8. **Deliverable:** `fig_s4_machines`: wall-clock vs number of machines (1/2/3),
   3 repetitions per point, median; annotate what was placed where.

## S5 — Digital-twin swap (BRIEF)

1. `cs_s5_dt.yaml`: copy the S1 baseline and replace the building federate with
   `type: interface` following the shipped BK4 demo pair
   (`src/scenarios/m5_bk4_demo_a_full_sim.yaml` vs `m5_bk4_demo_b_digital_twin.yaml`)
   as the exact template: same HELICS key names as the replaced federate,
   `interface_config` with the MQTT adapter, bridges targeting the keys the
   heat pump subscribes to. Also set `streaming: {stream: true}` on the heat-pump
   federate.
2. Feed the interface externally: small helper script
   `scripts/paper_case_study/s5_external_feeder.py` publishing plausible zone
   temperatures to the bridge's MQTT topic (mirror what the BK4 demo docs do).
3. **Deliverable:** run the combined dashboard (`src/dashboard/run_dashboard.sh`),
   Live page, during the run. **Screenshot is a human step** — get the run and
   dashboard working, then ask the user to capture `fig_s5_dashboard.png`.
4. **Acceptance:** subscribers (heat pump) consume externally fed values —
   verify in stored results that the zone-temperature input matches the feeder's
   published values, not a physics model.

## S6 — Engineering-effort audit (no simulation)

1. Script `scripts/paper_case_study/s6_loc_audit.py`:
   - For each stage S1→S5: YAML LOC of the stage's scenario file, the DIFF LOC
     vs the previous stage's file (added+changed lines), and new Python LOC
     written for that stage (reward function if custom, feeder script, etc. —
     exclude analysis/plotting scripts).
   - Count with blank/comment lines excluded; state the rule in the output.
2. **Deliverable:** `tab_s6_loc.csv` with columns: stage, capability added,
   YAML LOC, diff LOC vs previous, new Python LOC.
3. The "hand-built HELICS-to-Gym wrapper" comparison estimate is the human
   author's judgment call — output the framework-side proxy (LOC of
   `src/core/RL_Federate.py` + the RL config schema code) and flag it for the
   author to interpret.

---

## Final checklist

- [ ] `run_all.sh` reproduces every deliverable from a clean `results/` dir
- [ ] `MANIFEST.md` complete (scenario, commit, seeds, command, versions per deliverable)
- [ ] Figures as PDF+PNG, tables as CSV+MD, all under `results/paper_case_study/`
- [ ] Every deliverable maps to a paper slot:
  `fig_s1_traces`, `tab_s2_metrics`, `fig_s2_learning_curves`,
  `tab_s2_sample_eff`, `tab_s3_metrics`, `fig_s4_throughput` (+S4b log evidence),
  `fig_s4_machines`, `fig_s5_dashboard` (human screenshot), `tab_s6_loc`
- [ ] Nothing invented; failed/skipped items documented in MANIFEST with reason
