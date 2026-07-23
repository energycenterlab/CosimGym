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
1.1 **creation of scnearios.** the approach is that a scenario needs a declarative file the .yaml scenario and the models to be declared inside this file, check if you can use existing ones, do not modify existing ones, if no existing model is what it is needed create and register a new one.track tha creation of everything you did from scratch.
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
6. **Seeds.** Fixed seeds everywhere. The seed lives at
   `reinforcement_learning_config.seed` (top-level of that block — VERIFIED in
   `bui_hp_SAC.yaml`, `seed: 42`). RL comparisons: ≥3 seeds per configuration
   (e.g. 42, 43, 44), report mean ± std. For non-RL runs there is no seed field;
   the framework is deterministic given fixed inputs — note this in the manifest.
7. **Known hard constraints (source-verified — respect them, do not "fix" them):**
   - RL federates support only `memory_config.sink: json` use json;
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
   - The shipped reward `building_heatpump_comfort` is **comfort-only**
     (setpoint 20 °C, σ=0.5 °C, quadratic penalty — NO hard deadband; energy is
     not in the reward). VERIFIED at
     `src/models/model_catalog/RL_agents/reward_functions.py:93`.
   - Comfort violation: degree-hours outside an explicit comfort deadband
     `[19.5, 20.5] °C` (= setpoint 20 ± σ 0.5), computed as
     `sum(max(0, T_lower - T_zone) + max(0, T_zone - T_upper)) * dt_hours` over
     the evaluation horizon. Document these bounds in the metric output.
   - Energy: integral of heat-pump electrical power over the horizon (kWh).
   - Implement once in `scripts/paper_case_study/metrics.py`, reuse everywhere.
9. **execution metrics.** profile every run, keep track of timings and if possible memory usage, for each scenario (make more runs to add statistical validity) keep trakc of these metrics in a unique comparative table that i can use late inside the paper.
10. **energy scenarios** even if we are just designings cenarios for demonstration purposes i would like them to have a sort of energy meaningfulness, so most probably scenarios from S1-S4 will be focused on single building high granular control while all the scalability scenarios will focus on distrcit and multi building scenarios, make them energetically representative as much as possible. 

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

**S4a — vertical (parallel model execution within federate).**
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
9. **design** the distributed approach does not always brings benefits so we need to understand the kind of scenario that could benefit from this and the one that couldn't using dummy CPU-bounded models could be the way to make this an enhancer. also the machines specs make the difference, when you are at this point iterate with me on the best way to design this scenario

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

---

## APPENDIX A — VERIFIED IMPLEMENTATION REFERENCE (source-checked 2026-07-22)

Everything below was read directly from this repo. Trust it over guessing.
Do NOT re-verify these unless an error suggests they changed.

### A.1 How to launch / smoke-test ONE scenario

A scenario file `src/scenarios/<name>.yaml` is run by its `name` (filename stem).
From the **repo root**, conda env `cosim_gym`:

```bash
conda run -n cosim_gym python -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('<name>')"
```

- **SUCCESS marker:** the string `completed successfully` appears in stdout.
  Anything else (traceback, `Error`, `address already in use`, timeout) = FAIL.
- Results land in `results/<name>/<sim_id>/...`; logs in `logs/<name>/`.
- **Smoke first, always.** Copy the full scenario, cut horizon to ~1–2 sim-hours
  (RL: `episodes: 2`, `episode_length: 20`), run it, confirm the marker, THEN
  keep the full-horizon file. Delete the throwaway `results/logs` of smoke runs.
- **Concurrency safety:** OMIT `broker_config.port` (or give each scenario a
  UNIQUE port) so the framework auto-assigns — lets scenarios run back-to-back
  without `address already in use`. Redis is shared but keyed per scenario name.
- Docker services must be up first:
  `docker compose -f src/docker-compose.yaml up -d` (redis, mosquitto, minio).

### A.2 Exact model catalog keys (VERIFIED in catalog.yaml)

| role | catalog key | key I/O (name : dir) |
| --- | --- | --- |
| building 1R1C | `simple_building` | in `T_ext`,`Q_heat` · out `T_indoor` · param `thermal_capacitance`,`thermal_resistance`,`T_initial` |
| building 5R1C (richer) | `rc_building` | in `T_ext`,`solar_gains`,`internal_gains`,`t_set_heating`,`t_set_cooling` · out `T_indoor`,`Q_heating`,`P_elec`,`P_elec_mw`,… |
| heat pump | `simple_heatpump` | in `T_ext`,`modulation` · out `Q_heat`,`P_elec`,`COP` · param `P_rated`,`eta_carnot`,`T_supply`,`COP_min`,`COP_max` |
| PID controller | `simple_pid_controller` | in `T_indoor` · out `modulation` · param `T_setpoint`,`Kp`,`Ki`,`Kd` |
| weather (temp only) | `weather_csv_reader` | out `T_ext` · param `csv_path` (rel to model dir),`column`,`skip_rows` |
| generic CSV feeder | `base_csv_reader` | out = whatever the federate `publishes` (col names must match CSV headers) · param `csv_path`,`skip_rows` |
| PV | `pv_dest` | in `GHI`,`DHI`,`T_ext` · out `PV_power` · params are LISTS (e.g. `lat: [39.8]`) |
| battery | `battery_dest` | in `Battery_power` · out `SOC`,`energy`,`P_net`,`P_clipped` |
| rule-based BEMS | `rb_bems` | in `SOC`,`P_gen`,`P_load` · out `Battery_power`,`Grid_power` · param `SOC_min`,`SOC_max` |
| pandapower grid | `pandapower_grid` | dot-notation I/O `{comp}.{idx}.{col}` / `res.{comp}.{idx}.{col}` |
| CPU-heavy dummy | `heavy_compute_dummy` | out `result` · param `iterations` · no inputs (for S4a) |
| EnergyPlus FMU (local) | `bui0_building_fmu` | in 6 schedules (`PeopleNumber`,`LightsWatt`,`EEquipWatt`,`OthEquRadWatt`,`OthEquFCWatt`,`ZoneSetPoint`) · out `TBuilding`,`HeatingLoadTarget` · **local FMU, no MinIO** |
| EnergyPlus FMU (MinIO) | `adelaide_test` | in `SAT_SP` · out `Indoor Temp.`,`HVAC Power`,… · **needs MinIO object — currently MISSING (KNOWN_FAIL)** |
| FMU schedule feeder | `bui0_input_feeder` | out the 6 schedules above (pairs with `bui0_building_fmu`) |

RL agents: `rl_simple_SACsb3` (backend `stable_baselines3`, algo `SAC`, continuous),
`rl_simple_DQN` (backend `custom_torch`, algo `DQN`, discrete — needs `bins`),
`rl_simple_rllib` (RLlib PPO). Reward:
`models.model_catalog.RL_agents.reward_functions.building_heatpump_comfort`.

Weather CSVs shipped: `resources/weather_TO.csv` (**Turin — cold winter, use for
heating scenarios**), `resources/weather_data_bj.csv` (Beijing, has irradiance
`GloHorzRad`/`DifHorzRad` cols for PV).

### A.3 Canonical templates (copy these, do not reinvent)

- **S1 PID base** → copy `src/scenarios/bui_hp_test_base.yaml`. It IS the target
  topology (weather+PID+HP+building, wiring in its header comment). For the paper
  claim set weather `real_period: 3600`, the other three `real_period: 60`
  (multi-rate). Horizon ≥48 h → `start 2024-01-01T00:00:00`, `end 2024-01-03`.
- **S2 SAC** → copy `bui_hp_SAC.yaml` (PID federate REMOVED, RL block added). Obs
  keys `federation_1.weather.0.T_ext`, `federation_1.building.0.T_indoor`
  (dot-notation `<federation>.<federateName>.<instanceIdx>.<var>`); action
  `federation_1.heatpump.0.modulation: null`.
- **S2 DQN** → copy `bui_hp_DQN.yaml`. Action gains `space: discrete`, `bins: 10`;
  agent `rl_simple_DQN`, backend `custom_torch`.
- **S2 reset variants** → copy `bui_hp_SAC.yaml`, add
  `environment.reset: {mode: full|rolling|none}` (rolling also takes
  `rolling_window: 10`, see `bui_hp_SAC_rollingreset.yaml`). Nothing else differs.
- **S4a parallel** → copy `benchmark_parallel_par.yaml` / `_seq.yaml`. **Fix the
  shipped mismatch**: seq has `n_instances: 8`, par has `20`. Your copies for a
  given N must be BYTE-IDENTICAL except `parallel_execution: true/false`.
- **S4b multifed** → PV/battery wiring from `pv_batt_test_base.yaml`; district
  energy realism from `dh_district_jan_base.yaml` (10 `rc_building` + weather).
  Cross-federation subscribe target = **BARE, no federation prefix**:
  `<federate>.<instance>/<pubkey>` (flat GLOBAL namespace; EMPIRICALLY VERIFIED —
  the fed-prefixed form silently fails to bind. CLAUDE.md's config-reference line
  claiming a `<federation_name>.` prefix is WRONG; its architecture section is right).
  Multi-fed gotchas (both hit & fixed while building `cs_s4_topo.yaml`):
  (a) use `core_type: tcp` NOT `zmq` — zmq's `port+1` reply socket collides with
  sequential broker port auto-assignment; (b) federate `name:` must be unique across
  the WHOLE broker hierarchy, not just per-federation (else `duplicate federate name`).
- **S5 DT interface** → copy the BK4 pair `m5_bk4_demo_a_full_sim.yaml` (all-sim)
  / `m5_bk4_demo_b_digital_twin.yaml` (interface federate + MQTT bridges,
  `scope: input`, `mode: replace`). External feeder pattern:
  `src/scenarios/bk4_demo_external_sensor.py`.

### A.4 Hard gotchas (will silently break a run)

- RL config is Pydantic `extra='forbid'` — one misspelled key aborts. Copy a
  working block; change only values/keys you understand.
- RL federate: only `sink: json`; `parallel_execution` and `host:` rejected on RL.
- Cross-federation keys use GLOBAL flat namespace; federation broker uplink is a
  bare `host:port` (no `zmq_ss://` scheme — that hangs the sub-broker).
- FMU horizons must be whole-day multiples; do NOT auto-shorten FMU scenarios to 1 h.
- `pv_dest` parameters are single-element LISTS in YAML, not scalars.

### A.5b S3 — RESOLVED (supersedes A.5 below)

Investigated and settled 2026-07-22:
- The MinIO object WAS genuinely missing (bucket `fmus` is created empty by
  `minio-init`; nothing ever uploaded it). It has now been **restored** by
  uploading the local `resources/PCMA_1_0_control_2022.fmu` to
  `fmus/adelaide_test/1.0.0/PCMA_1_0_control_2022.fmu`. The download path works.
- **But `adelaide_test` still cannot run on this machine**: its native binary needs
  `GLIBC_2.33`, the host has **glibc 2.31** →
  `Failed to load shared library …PCMA_1_0_new_control.so … version 'GLIBC_2.33' not found`.
  This is a hard binary-compatibility wall, not a config problem. Do not retry it
  here (it would need a newer-glibc container or a recompiled FMU).
- **S3 therefore uses the local BUI0 EnergyPlus FMU** (`bui0_building_fmu`), which
  runs fine. `cs_s3_fmu.yaml` is derived from the working `bui0_setpoint_SAC.yaml`.
- The swap is **NOT variable-names-only** — be honest in the paper. It also drops the
  weather + heat-pump federates, moves the action to a zone set-point, changes the
  reward, and forces `real_period` 600 s + `reset.mode: none`. The exact diff table is
  emitted by `scripts/paper_case_study/tab_s3_metrics.py`.

### A.5 S3 (FMU) design note — SUPERSEDED by A.5b, kept for history

The "agent untouched" claim wants the FMU building to expose the SAME dot-keys as
`simple_building` (`T_indoor` out, driven by `Q_heat`). Neither shipped FMU does:
`adelaide_test` needs a MISSING MinIO object (blocked), and `bui0_building_fmu` is
schedule-driven (`ZoneSetPoint` in, `TBuilding` out — no `Q_heat`/`T_indoor`).
Therefore S3 as literally specified is NOT drop-in. **Do not fake it.** The S3
agent must STOP after S1/S2/S4/S5 are done and report options to the human:
(a) restore the MinIO FMU object then use `adelaide_test`; (b) accept a
variable-names-only diff against `bui0_building_fmu` and report that diff honestly;
(c) defer S3. This is a genuine decision for the author — surface it, don't guess.
