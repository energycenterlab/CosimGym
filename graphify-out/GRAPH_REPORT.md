# Graph Report - CosimGym  (2026-11-19)

## Corpus Check
- 62 files · ~87,503 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 13 file(s) not represented in the graph (top: (none) 6, .pth 3, .drawio 1)

## Summary
- 431 nodes · 624 edges · 25 communities
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.87)
- Token cost: 724,631 input · 0 output

## Community Hubs (Navigation)
- Huge-Scale Distribution Probes
- SSH Spawning Alternatives
- Examples & Docs Home
- Project Rules & Setup
- FMU Restart & Reset
- Phase-D Exchange Experiments
- Algebraic Loop Federate
- Horizon vs Episode Reset
- FMU Models & RL Example
- Synchronization & Causality
- RL Configuration Axes
- Scaling Study Harness
- Pandapower Extension
- Core Terminology
- Scaling Bottlenecks B1-B10
- Federate Configuration Keys
- Huge-Scale Index & Gaps
- Storage & Dashboard
- Scaling Synthesis & Rules
- Federation & Scenario Config
- Coupling Cost Model
- Known Issues Register
- Models Reference
- Connections & Key Naming
- Doc Authority Rules

## God Nodes (most connected - your core abstractions)
1. `CLAUDE.md project guide` - 15 edges
2. `Scaling Bottleneck Catalog` - 15 edges
3. `Federate Configuration` - 14 edges
4. `Phase D report (data-exchange comms term)` - 14 edges
5. `Installation & Setup` - 13 edges
6. `Digital-Twin Interfaces & Live Streaming` - 13 edges
7. `Reinforcement Learning Configuration` - 13 edges
8. `Scaling-Study Harness Interface Contract` - 13 edges
9. `Model reset redesign plan` - 12 edges
10. `What We Measured, In Plain Language (Phase-D tour)` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Introduction` --semantically_similar_to--> `README`  [INFERRED] [semantically similar]
  docs/overview/introduction.md → README.md
- `Docs Home (index)` --semantically_similar_to--> `README`  [INFERRED] [semantically similar]
  docs/index.md → README.md
- `Installation & Setup` --references--> `Docker Compose v2 requirement`  [EXTRACTED]
  docs/Installation_Setup.md → CLAUDE.md
- `Models Reference` --references--> `Model catalog (catalog.yaml + RedisCatalog)`  [INFERRED]
  docs/models_reference.md → CLAUDE.md
- `Live page (live_dashboard.py MQTT)` --conceptually_related_to--> `streaming (outbound MQTT mirror)`  [INFERRED]
  src/dashboard/README.md → docs/user_guide/scenario_configuration/federate.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CosimGym execution pipeline** — claude_scenariomanager, claude_federate_launcher, claude_basefederate, claude_rlfederate, claude_redis [EXTRACTED 1.00]
- **Digital-twin mechanisms over MQTT** — docs_user_guide_digital_twin_interfaces_streaming, docs_user_guide_digital_twin_interfaces_interface_config, docs_user_guide_digital_twin_interfaces_override_registry, docs_user_guide_digital_twin_interfaces_mqtt_adapter, docs_user_guide_digital_twin_interfaces_bk4_pattern [EXTRACTED 1.00]
- **Distributed run networking guidance** — docs_user_guide_distributed_deployment_single_socket_cores, docs_user_guide_multi_machine_test_walkthrough_nat_registration_timeout, docs_user_guide_multi_machine_test_walkthrough_firewall_range, docs_user_guide_multi_machine_test_walkthrough_docker_dnat_bypass, docs_user_guide_distributed_deployment_brokers_on_manager [INFERRED 0.85]
- **auto_offset sequencing pipeline** — docs_user_guide_scenario_configuration_synchronization_build_federate_dependency_graph, docs_user_guide_scenario_configuration_synchronization_compute_sccs, docs_user_guide_scenario_configuration_synchronization_kahn_topological_sort, docs_user_guide_scenario_configuration_synchronization_auto_offset [EXTRACTED 1.00]
- **Reset redesign mechanisms** — docs_changes_reports_model_reset_redesign_plan_relative_clock, docs_changes_reports_model_reset_redesign_plan_unified_reset, docs_changes_reports_model_reset_redesign_plan_reset_in_federate_ticks, docs_changes_reports_model_reset_redesign_plan_span_ticks, docs_changes_reports_model_reset_redesign_plan_fmu_single_restart_path, docs_changes_reports_model_reset_redesign_plan_csv_row_from_rel_ts [EXTRACTED 1.00]
- **RL config four orthogonal axes** — docs_user_guide_scenario_configuration_rl_environment_axis, docs_user_guide_scenario_configuration_rl_agent_axis, docs_user_guide_scenario_configuration_rl_run_axis, docs_user_guide_scenario_configuration_rl_experiment_axis [EXTRACTED 1.00]
- **Same-step SCC resolution policies (error / relax / iterate)** — docs_future_and_todos_synchronization_causality_plan_same_step_cycle_validation, docs_future_and_todos_synchronization_causality_plan_cycle_policy_relax, docs_future_and_todos_algebraic_loop_federate_plan_cycle_policy_iterate, docs_future_and_todos_synchronization_causality_plan_next_step_causality [EXTRACTED 1.00]
- **Distributed SSH execution design choices** — docs_future_and_todos_distributed_ssh_spawning_alternatives_openssh_controlmaster, docs_future_and_todos_distributed_ssh_spawning_alternatives_conda_env_plus_rsync, docs_future_and_todos_distributed_ssh_spawning_alternatives_rsync_back_results, docs_future_and_todos_distributed_ssh_spawning_alternatives_per_federate_host_placement, docs_future_and_todos_distributed_ssh_spawning_alternatives_brokers_on_manager, docs_future_and_todos_distributed_ssh_spawning_alternatives_redis_config_fetch, docs_future_and_todos_distributed_ssh_spawning_alternatives_tt_pty_orphan_prevention [EXTRACTED 1.00]
- **Scaling study Part B phases D-H** — docs_future_and_todos_scaling_study_plan_phase_d_data_exchange, docs_future_and_todos_scaling_study_plan_phase_e_idle_roofline, docs_future_and_todos_scaling_study_plan_phase_f_optimal_placement, docs_future_and_todos_scaling_study_plan_phase_g_real_case_study, docs_future_and_todos_scaling_study_plan_phase_h_remote_brokers [EXTRACTED 1.00]
- **Phase-D data-exchange campaign experiments** — scripts_scaling_study_experiments_experiment1_local_gated, scripts_scaling_study_experiments_experiment2_local_wide, scripts_scaling_study_experiments_experiment3_cross_machine, scripts_scaling_study_experiments_experiment4_stress_m, scripts_scaling_study_experiments_experiment5_stress_n, scripts_scaling_study_experiments_experiment6_hugescale [EXTRACTED 1.00]
- **Scaling harness D1-D6 pipeline** — scripts_scaling_study_contract_d1_generator, scripts_scaling_study_contract_d2_bench_driver, scripts_scaling_study_contract_d3_perf_log, scripts_scaling_study_contract_d4_cost_model, scripts_scaling_study_contract_d5_report, scripts_scaling_study_contract_d6_stress_driver [EXTRACTED 1.00]
- **Fixed harness port/path bugs** — scripts_scaling_study_findings_bottlenecks_b8, scripts_scaling_study_findings_bottlenecks_b9, scripts_scaling_study_findings_bottlenecks_b10 [EXTRACTED 1.00]
- **Additive cost-model failure modes (missing interaction terms)** — scripts_scaling_study_findings_phase5_validation_n_work_gap, scripts_scaling_study_findings_hugescale_multipc_o_par_regime_bound, scripts_scaling_study_findings_phaseD_exchange_concentration_residual, scripts_scaling_study_findings_phase01_pooled_fit_collinearity [INFERRED 0.85]
- **Coupling cost levers (edges, payload, cadence, concentration)** — scripts_scaling_study_findings_phaseD_exchange_comms_law, scripts_scaling_study_findings_phaseD_exchange_payload_width, scripts_scaling_study_findings_phaseD_exchange_publish_cadence, scripts_scaling_study_findings_phaseD_exchange_concentration_residual [EXTRACTED 1.00]
- **Harness bugs found and later fixed (SSH ControlPath, port spacing, port block)** — scripts_scaling_study_findings_phase2_ceiling_ssh_controlpath_bug, scripts_scaling_study_findings_phase3_federations_port_spacing_bug, scripts_scaling_study_findings_phaseD_exchange_port_block_b10 [EXTRACTED 1.00]

## Communities (25 total, 0 thin omitted)

### Community 0 - "Huge-Scale Distribution Probes"
Cohesion: 0.06
Nodes (56): Distribution wall speedup tracks core-pool ratio (1.48-1.68x vs 1.57x), Huge-scale multi-PC probe report (N=176, M to 10000), 176 000 model instances run clean (unwired), M=10000 (1.76M instances) timeout, undiagnosed, O_par is regime-bound (N x M x W failure of additive par model), B13 par worker oversubscription at large N, Phase E idle-host rerun plan, Coupling law (paper prose: per-link + per-byte) (+48 more)

### Community 1 - "SSH Spawning Alternatives"
Cohesion: 0.07
Nodes (19): MinIO/S3 live results sink (future), DeploymentConfig / MachineConfig, distributed_demo localhost-as-remote E2E, Hierarchy-broker uplink bare host:port fix, RemoteExecutor, Per-machine port allocation / remote bind probe, Single-socket cores zmq_ss/tcp_ss, T_tick = max(compute+sync+comms) (+11 more)

### Community 2 - "Examples & Docs Home"
Cohesion: 0.07
Nodes (31): Example 1: Base Simulation, simple_test scenario (spring-mass-damper), simple_DQN_test scenario, Provided Examples List, Distributed Execution feature, Docs Home (index), Dev Container + VS Code, Installation & Setup (+23 more)

### Community 3 - "Project Rules & Setup"
Cohesion: 0.08
Nodes (24): BaseFederate, code-review-graph (code only), CosimGym framework, CLAUDE.md project guide, federate_launcher.py, graphify (docs only), Gymnasium, HELICS co-simulation middleware (+16 more)

### Community 4 - "FMU Restart & Reset"
Cohesion: 0.07
Nodes (35): FMU restart & reset changes report, Correction: EnergyPlus FMU can start/restart at any date within range (whole-day duration, 60 s min step), scripts/fmu_warmstart_validation harness, Horizon restart (max_sim_time), Model clock local_ts / ts_shift / reposition (superseded), noSetFMUStatePriorToCurrentPoint fix, FMU restart and replay (superseded), restart_strategy runperiod_shift (rejected) (+27 more)

### Community 5 - "Phase-D Exchange Experiments"
Cohesion: 0.15
Nodes (15): What We Measured, In Plain Language (Phase-D tour), exchange_dummy model, Experiment 1: local gate-limited, Experiment 2: local wide range, Experiment 3: cross-machine two-armed, Experiment 5: stress federates, Publish every 10th tick cuts 90% coupling cost, LAN edge costs no more than local (+7 more)

### Community 6 - "Algebraic Loop Federate"
Cohesion: 0.20
Nodes (8): HELICS iterative time request, IterativeFederate (type: iterative), Relaxation / damping factor, Auto time offsets (topological staging), default_subscription_causality ignored bug, ObservationSpec phase: pre_action / post_action, same_step causality (instantaneous), Same-step cycle validation (Tarjan SCC)

### Community 7 - "Horizon vs Episode Reset"
Cohesion: 0.16
Nodes (13): E+ FMU engine auto-discovery (generationTool), Model horizon restart (max_sim_time), parallel_execution reset propagation gap, Restart discontinuity not signalled (truncated), RL episode reset (full/rolling/none), Rolling reset publishes initial state, Bug A simple_DQN_test missing experiment block, Bug B setpoint scenarios silent zero reward (+5 more)

### Community 8 - "FMU Models & RL Example"
Cohesion: 0.16
Nodes (13): Example 2: RL Training, reinforcement_learning_config (environment/agent/run/experiment), bui0_fmu_test worked example, FMU Models, EnergyPlus FMU constraints (midnight start, whole days, defined stop time), fmu_catalog_register.py, fmu_source (local|minio|http), RL Integration (+5 more)

### Community 9 - "Synchronization & Causality"
Cohesion: 0.19
Nodes (13): ScenarioManager._resolve_target_federate_node, synchronization block, auto_offset: automatic time-offset sequencing, ScenarioManager._build_federate_dependency_graph, Subscription causality same_step | next_step, ScenarioManager._compute_sccs (Tarjan SCC), Synchronization & Causality, BaseFederate._enforce_startup_input_sync (+5 more)

### Community 10 - "RL Configuration Axes"
Cohesion: 0.17
Nodes (12): reinforcement_learning_config, agent axis (model_name/backend/hyperparameters/params), Available agents rl_simple_SACsb3 / rl_simple_DQN / rl_simple_rllib, Reinforcement Learning Configuration, environment axis (MDP: observations/actions/reward/reset), experiment axis (name/checkpoint/logging/offline), Pydantic extra='forbid' on RL config, Hyperparameters universal core (all Optional) (+4 more)

### Community 11 - "Scaling Study Harness"
Cohesion: 0.26
Nodes (12): Sweep knobs F/N/M/mode/W/core_type/placement, D1 parametric generator gen_scenario.py, D2 bench driver run_bench.py, D3 perf.json log (COSIM_PERF_LOG), D4 cost-model fitter + recommender, D5 report/plots make_report.py, D6 stress driver stress_ramp.py, Scaling-Study Harness Interface Contract (+4 more)

### Community 12 - "Pandapower Extension"
Cohesion: 0.18
Nodes (9): Case file format auto-detection, DC approximation fast-path, Diagnostics outputs, Multi-area grids via tie-line flows, Network metadata pushed to Redis, OPF cost functions, output_all_res_tables mode, State-estimation integration (+1 more)

### Community 13 - "Core Terminology"
Cohesion: 0.27
Nodes (11): Adapter (InterfaceAdapter), Framework Terminology, Federate, Federation, Model (BaseModel), Scenario, BaseModel lifecycle (initialize/step/finalize/reset), Custom Models & Catalog (+3 more)

### Community 14 - "Scaling Bottlenecks B1-B10"
Cohesion: 0.24
Nodes (11): B1 HELICS lockstep sync s(N), B10 per-federation port block sized from N (fixed), B4 host process contention, B5 shared-machine co-user contention, B6 distribution/LAN overhead, B7 ~300 MB RSS per federate, B8 SSH ControlPath 108-byte limit (fixed), B9 zmq port+1 collision (fixed) (+3 more)

### Community 15 - "Federate Configuration Keys"
Cohesion: 0.22
Nodes (10): interface_config bridges (scope input/output/param, mode replace/passthrough), Federate Configuration, flags (FedFlags), type interface: interface federate (digital-twin bridge), Per-federate memory_config override, override_enabled flag, type rl: RL agent federate, streaming (outbound MQTT mirror) (+2 more)

### Community 16 - "Huge-Scale Index & Gaps"
Cohesion: 0.22
Nodes (9): Experiment 6: huge scale one PC vs three, heavy_compute_dummy model, B13 par worker oversubscription, Scaling Study Canonical Index, Known gaps / open list, Max scale reached, O_par regime-bound, Phases done table (0-5, D, D-x, HS) (+1 more)

### Community 17 - "Storage & Dashboard"
Cohesion: 0.33
Nodes (9): Parquet sink native SIGSEGV (KI 1), Storage sink json | parquet | none, dashboard_data.py JSON/Parquet parsing, Dashboard Notes, Live page (live_dashboard.py MQTT), Results page (dashboard_app.py), streamlit_dashboard.py single entrypoint (st.navigation), Streamlit Dashboard Requirements (+1 more)

### Community 18 - "Scaling Synthesis & Rules"
Cohesion: 0.22
Nodes (8): Best-config rules (M/seq-par, F, placement, recommend), Scaling Study Synthesis (Phases 0-5), F broker-startup setup term, N x work interaction term needed, B2 parallel dispatch O_par, B3 federation broker startup tax, N x work modeling gap, Crossover law (M-ceil(M/W))*c > O_par

### Community 19 - "Federation & Scenario Config"
Cohesion: 0.29
Nodes (7): Federation Configuration, Hierarchy broker (multi-federation), General Scenario Configuration, multi_computer / multi_computer_config, Scenario name key (renamed from scenario_name), Scenario config hierarchy (ScenarioConfig > FederationConfig > FederateConfig), Scenario Configuration Overview

### Community 20 - "Coupling Cost Model"
Cohesion: 0.32
Nodes (6): comms cost term (per-edge + per-link + per-byte), Phase-D data-exchange wiring knobs, Fan-out patterns 1to1/1toN/Nto1/all2all, max_fed_in (busiest subscriber inbound links), n_edges (total HELICS links), B11 data-exchange coupling comms

### Community 21 - "Known Issues Register"
Cohesion: 0.29
Nodes (7): Adelaide_test missing MinIO object (KI 4), Known Issues & Limitations, regression_suite KNOWN_FAIL dict, test_scenario_manager_remote 2 tests broken (KI 6), RL_Simple_Agent non-functional skeleton (KI 3), zmq auto-port allocation ignores port+1 (KI 2), broker_config (BrokerConfig)

### Community 22 - "Models Reference"
Cohesion: 0.43
Nodes (6): Models Reference, heavy_compute_dummy benchmark model, pandapipes_grid model, pandapower_grid model, rc_building (5R1C ISO 13790), Subscription key mapping via targets

### Community 23 - "Connections & Key Naming"
Cohesion: 0.29
Nodes (7): connections (publishes/subscribes), Subscription targets format <federate>.<instance>/<key>, Cross-federation subscriptions: flat global HELICS key namespace, BaseFederate._register_pubs, BaseFederate._register_subs, simple_test_multifederations scenario, RL key naming <federation>.<federate>.<instance>.<variable>

### Community 24 - "Doc Authority Rules"
Cohesion: 0.40
Nodes (5): Authority order: code graph > code > docs, Cross-federation target doc fix, Session rules & doc audit changes report, KNOWN_ISSUES as single defect register (not a changelog), CLAUDE.md working rules

## Ambiguous Edges - Review These
- `M=10000 (1.76M instances) timeout, undiagnosed` → `Teardown stall B12 (sim finishes, run hangs)`  [AMBIGUOUS]
  scripts/scaling_study/findings/hugescale_multipc.md · relation: conceptually_related_to

## Knowledge Gaps
- **93 isolated node(s):** `Example 1: Base Simulation`, `Gymnasium`, `Claude Code setup rebuild on eclab-cloud1`, `cosim_gym conda env missing on new machine`, `Main features list (declarative YAML, orchestration, plug-and-play, Gym, dashboard, DT)` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 120 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Digital-Twin Interfaces & Live Streaming` connect `Examples & Docs Home` to `FMU Models & RL Example`, `Project Rules & Setup`, `Core Terminology`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **What connects `Example 1: Base Simulation`, `Gymnasium`, `Claude Code setup rebuild on eclab-cloud1` to the rest of the system?**
  _93 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Huge-Scale Distribution Probes` be split into smaller, more focused modules?**
  _Cohesion score 0.055191256830601096 - nodes in this community are weakly interconnected._
- **Why does `Known Issues & Limitations` connect `Known Issues Register` to `FMU Restart & Reset`, `RL Configuration Axes`, `Storage & Dashboard`, `Federation & Scenario Config`, `Doc Authority Rules`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `SSH Spawning Alternatives` be split into smaller, more focused modules?**
  _Cohesion score 0.06533776301218161 - nodes in this community are weakly interconnected._
- **Should `Examples & Docs Home` be split into smaller, more focused modules?**
  _Cohesion score 0.06829268292682927 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `M=10000 (1.76M instances) timeout, undiagnosed` and `Teardown stall B12 (sim finishes, run hangs)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._