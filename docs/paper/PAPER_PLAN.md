# CosimGym — Scientific Paper Plan

> Working document. Planning only — no code changes implied. Author target: P. Rando Mazzarino et al., EC-Lab / Politecnico di Torino.
> Date drafted: 2026-06-15.

This document turns the CosimGym repository into a publishable research artifact. It contains:
1. Technical anatomy of the repo (what it actually does, grounded in the code).
2. Novelty + gap analysis vs. the literature.
3. LaTeX paper outline.
4. Ranked journal targets.
5. Storytelling spine.
6. Case studies to prove the claims.
7. Future enhancements that raise significance.
8. Seed bibliography.

---

## 0. One-paragraph pitch (the abstract seed)

> Data-driven control of cyber-physical energy systems requires agents that learn against *realistic, multi-domain, multi-rate* physics. Today these two worlds are bridged ad hoc: co-simulation middleware (HELICS, FMI) speaks publish/subscribe and federated time, while reinforcement-learning (RL) tooling (Gymnasium, Stable-Baselines3, RLlib) speaks `reset()`/`step()`. Existing RL testbeds either hard-wire a single simulator (Sinergym→EnergyPlus, CityLearn, BOPTEST) or expose a lightweight, *synchronous fixed-frequency* power-systems gym (PowerGridworld). We present **CosimGym**, a declarative, model-agnostic orchestration framework that compiles a single YAML scenario into a running HELICS federation and *automatically* presents it as a standard Gymnasium environment. Its contributions are: (i) a **declarative scenario compiler** that resolves brokers, ports, multi-rate timing, causality, and hierarchical multi-federation topologies with no orchestration code; (ii) a **catalog-driven model abstraction** that unifies native Python models, FMI 2.0/3.0 FMUs, CSV/data feeders, and RL agents behind one interface, with I/O bounds that *auto-derive* observation/action spaces; and (iii) a principled treatment of the **episode-reset problem in co-simulation**, where federates cannot be cheaply rewound — implemented as configurable `full | soft | rolling | random` reset strategies with causality-aware observation staging. We demonstrate CosimGym across thermal-building HVAC control and PV+battery dispatch, training DQN and SAC agents against the same composable physics that a pure co-simulation would use for verification.

---

## 1. Technical anatomy (grounded in the codebase)

This section is the factual backbone. Every claim below maps to source.

### 1.1 Execution model
- **`ScenarioManager`** (`src/core/ScenarioManager.py`): reads YAML → Pydantic v2 config → starts HELICS broker(s) as subprocesses → spawns one OS process per federate via `federate_launcher.py`. Config is serialized to **Redis** so each subprocess self-configures. This is a *compiler + process orchestrator*, not a monolithic sim loop.
- **`federate_launcher.py`**: per-federate entrypoint; pulls config from Redis; instantiates `BaseFederate` (`type: base`) or `RL_Federate` (`type: rl`).
- **`BaseFederate`** (`src/core/BaseFederate.py`): owns the HELICS pub/sub lifecycle, time stepping, input staging, storage, reset.
- **`RL_Federate` / `HelicsGymEnv`** (`src/core/RL_Federate.py`): subclass of `BaseFederate` that *is* a `gymnasium.Env`. It routes HELICS subscriptions → observations and actions → HELICS publications.

### 1.2 Declarative scenario → typed config → runtime
- YAML scenarios live in `src/scenarios/` (16 example scenarios already present: spring-mass-damper, building+HP+PID, building+HP+RL, PV+battery+RB, PV+battery+RL, multi-federation, FMU feedthrough, EnergyPlus FMU "Adelaide").
- Parsed into **Pydantic v2** models in `src/utils/config_dataclasses.py`. Discriminated union on `type` (`BaseFederateConfig` | `RLFederateConfig`); validators enforce unique IDs, federate counts, RL training-or-test presence, checkpoint path normalization, auto-injection of federation/federate names and IDs.
- For RL, `ScenarioManager._modify_config_for_online_training()` *synthesizes* an extra `rl_federation` containing an `rl_agent` federate, wiring its pub/sub from the `reinforcement_learning_config` block. **The user never writes the agent's HELICS plumbing.**

### 1.3 Model Catalog (the extensibility core)
- `src/models/model_catalog/catalog.yaml`: static registry. Each entry = `class_name` + `module_path` + typed I/O spec (inputs/outputs/parameters, each with `type`, `min_value`, `max_value`, `unit`, `tags`).
- Loaded into Redis at startup (`catalog_loader.py`); resolved at runtime by `RedisCatalog`.
- Built-in models already span: thermal building (1R1C), heat pump (Carnot-fraction COP), PID controller, rule-based BEMS, PV (`pv_dest`), battery, weather/CSV readers, spring-mass-damper, plus **FMI 2.0 EnergyPlus** (`bui0_building_fmu`, `adelaide_test`) and **FMI 3.0 feedthrough** FMUs via `base_FMU_model` (fmpy). FMU artifacts can be pulled from **MinIO** object storage (see `adelaide_test`).
- New model = inherit `BaseModel` (`initialize`/`step`/`finalize`) + one catalog entry. Same interface for physics, controllers, and RL agents.

### 1.4 The two pieces of genuine engineering depth

**(A) Timing / causality engine.**
- `real_period` (seconds) is the only required timing field; HELICS time is normalized to integer ticks against the minimum `real_period` (multi-rate support out of the box).
- `time_offset` shifts a federate's first tick to break same-step algebraic loops; `synchronization.auto_offset` computes offsets via **topological sort** of the dependency graph.
- `subscription.causality: same_step | next_step` controls whether a value is consumed immediately or deferred one tick — with cycle validation (`validate_causality_cycles`). `BaseFederate` implements deferred-input staging (`_apply_deferred_inputs`, `staged_value`).
- **Multi-federation**: when >1 federation, `ScenarioManager` inserts a HELICS **hierarchy broker** (`--sub_brokers=N`) and assigns TCP ports dynamically.

**(B) RL episode semantics under co-simulation.**
- Spaces are **auto-built** from catalog I/O bounds (`_prepare_obs_dict`, `_prepare_act_dict`, `build_space`) — Box/Discrete derived from `min_value`/`max_value`/type; float→discrete bin remapping supported.
- `reset_mode: full | soft | rolling | random` plus `rolling_window` — this is the framework's answer to "you cannot truly `reset()` a federation cheaply." Reset uses `reset_observation_defaults`, deferred-input clearing, and forced re-read so the next action is chosen from the *reset* state, not the terminal state.
- Pluggable reward functions referenced by dotted path (`reward_functions.py`).
- Agents: PyTorch DQN, Stable-Baselines3 SAC, simple agent — all catalog entries.
- Storage is partitioned `train`/`test`, records obs-before/after action, actions, rewards, per-episode aggregates → JSON → Streamlit dashboard (parquet-cached).

### 1.5 Honest current-state caveats (must shape claims)
- README marks it **early prototype**; several `TODO`s in RL space handling (discrete remapping, `include_prev_obs`, truncation can't yet vary episode length dynamically).
- Tests are thin (dashboard tests only). **A paper needs a reproducibility + validation harness** (see §6, §7).
- Single-machine focus (multi-computer config scaffolded but not the headline).

---

## 2. Novelty & gap analysis vs. literature

### 2.1 The landscape
| Framework | Physics backend | Coupling | Multi-rate / multi-domain | Multi-agent | Model abstraction | Declarative config |
|---|---|---|---|---|---|---|
| **Sinergym** | EnergyPlus only | Direct (BCVTB/py) | No | Limited | EnergyPlus-bound | Partial |
| **BOPTEST** | Modelica emulators | REST/Gym wrapper | No | No | Test-case bound | No |
| **CityLearn** | Built-in data models | Native Gym | No (fixed hourly) | **Yes (district)** | Domain-fixed | Schema-based |
| **Energym** | Modelica/EnergyPlus | FMU + wrapper | Limited | No | Library of cases | No |
| **PowerGridworld** | OpenDSS + components | Native Gym | **No — synchronous fixed-frequency** (acknowledged limitation) | **Yes (MARL)** | Component gyms | Partial (Python) |
| **GridLAB-D+HELICS+Gym** (Vertical FRL) | GridLAB-D | HELICS, bespoke | Yes (HELICS) | Yes | Hand-wired per study | No |
| **CosimGym** | **Any: Python / FMI 2&3 / CSV / EnergyPlus** | **HELICS, generic** | **Yes (native multi-rate + hierarchical multi-federation)** | Roadmap (single-agent now) | **Catalog-driven, domain-agnostic** | **Yes — full YAML compiler** |

### 2.2 Where CosimGym is genuinely novel / fills gaps

1. **Declarative co-sim→RL compilation.** Most HELICS+RL papers (e.g., resilient microgrid FRL on GridLAB-D) hand-build a bespoke Gym wrapper per study. CosimGym makes the *entire* federation, timing, and the agent's pub/sub wiring a YAML artifact. **Gap filled: reproducibility + zero-boilerplate composition.** This is the strongest, most defensible novelty.

2. **Domain-agnostic, multi-formalism model catalog.** Sinergym is EnergyPlus-locked; CityLearn/PowerGridworld are domain-locked. CosimGym treats native Python models, **FMI 2.0/3.0 FMUs**, EnergyPlus FMUs, data feeders, and RL agents through one `BaseModel` interface + typed catalog. **Gap filled: cross-domain, cross-tool RL benchmarking on one substrate** (the FMI standard is the lingua franca of >100 tools — a natural reach for energy + beyond).

3. **Native multi-rate, causality-aware, hierarchical orchestration.** PowerGridworld explicitly lists *synchronous, fixed-frequency time stepping* and a *limited communication model* as limitations. CosimGym inherits HELICS multi-rate time + adds `same_step/next_step` causality, topological auto-offset to break algebraic loops, and hierarchical multi-federation brokers. **Gap filled: realistic multi-timescale dynamics for RL** (e.g., 1 s controller vs. 1 h weather).

4. **First-class treatment of the co-simulation reset problem.** Gymnasium assumes a cheap `reset()`. A HELICS federation can't be rewound for free. CosimGym's `full/soft/rolling/random` reset modes + observation staging are a *contribution to RL-on-co-simulation methodology*, not just plumbing. **Gap filled: episodic RL semantics over non-resettable federated simulators.** Worth its own subsection / micro-benchmark.

5. **Spaces auto-derived from physical I/O bounds.** Observation/action spaces are generated from catalog `min/max/unit`, not hand-coded. This couples *physical validity* to the RL interface (units, bounds, clipping) — reduces a common source of silent RL-on-physics bugs.

### 2.3 Defensible framing (don't oversell)
- HELICS+RL has prior art; **the novelty is the abstraction layer + declarative compiler + reset methodology**, not "first to connect HELICS to a gym."
- Single-agent today; position MARL as roadmap (HELICS makes it natural — strong future-work story, see §7).
- Lead with *reproducibility, composability, multi-formalism, multi-rate, reset semantics* — these are uncontested.

---

## 3. LaTeX paper outline

```latex
\documentclass[review]{elsarticle} % or IEEEtran / mdpi
\title{CosimGym: A Declarative, Model-Agnostic Framework for
       Reinforcement Learning on HELICS Co-Simulations}

%==================================================================
\section{Introduction}                                 % ~1.5 pp
  \subsection{Motivation: learning control on realistic CPS physics}
  \subsection{The co-sim / RL impedance mismatch}
  \subsection{Contributions}     % the 5 bullets from §2.2
  \subsection{Paper structure}

\section{Background and Related Work}                   % ~2 pp
  \subsection{Co-simulation: HELICS and the FMI standard}
  \subsection{RL environments for energy systems}
        % Sinergym, BOPTEST, CityLearn, Energym, BEAR
  \subsection{RL on co-simulation / power systems}
        % PowerGridworld, GridLAB-D+HELICS FRL, CommonPower
  \subsection{Gap analysis}      % Table 1 (the comparison matrix)

\section{Framework Architecture}                        % ~3 pp (core)
  \subsection{Declarative scenario model (YAML $\to$ typed config)}
  \subsection{Orchestration: brokers, processes, Redis state}
  \subsection{Model Catalog and the BaseModel abstraction}
        % Python / FMI 2&3 / EnergyPlus / data / agents
  \subsection{Timing and causality engine}
        % multi-rate ticks, auto-offset (topo-sort), same/next-step,
        % hierarchical multi-federation brokers
  \subsection{Gymnasium integration}
        % HelicsGymEnv, auto-derived spaces from physical bounds

\section{Episodic RL over Non-Resettable Co-Simulations} % ~1.5 pp (novel)
  \subsection{The reset problem in federated simulation}
  \subsection{Reset strategies: full / soft / rolling / random}
  \subsection{Causality-aware observation staging}
  \subsection{Reward and storage model}

\section{Case Studies}                                   % ~3-4 pp
  \subsection{CS0: Verification on spring-mass-damper (single \& multi-federation)}
  \subsection{CS1: Building HVAC control (1R1C + HP + weather)}
        % RB/PID baseline vs DQN vs SAC; comfort vs energy
  \subsection{CS2: EnergyPlus-FMU building (BUI0 / Adelaide)}  % multi-formalism proof
  \subsection{CS3: PV + battery dispatch}
        % RB BEMS baseline vs DQN/SAC; self-consumption
  \subsection{CS4 (optional): multi-rate / multi-federation coupling}

\section{Results and Discussion}                         % ~2-3 pp
  \subsection{Control performance vs. baselines}
  \subsection{Effect of reset strategy on learning (ablation)}   % unique
  \subsection{Multi-rate fidelity \& reproducibility}
  \subsection{Engineering effort: LOC / config-only comparison}  % composability evidence
  \subsection{Limitations}

\section{Conclusions and Future Work}                    % ~1 pp
        % MARL, distributed multi-host, ML-FMU surrogates, safe-RL,
        % standard benchmark suite

\appendix
\section{Reproducibility}   % scenario YAMLs, seeds, env, hashes
```

**Target length:** 14–20 pages (journal). Figures: architecture diagram, data-flow per case study, learning curves, reset-ablation, multi-rate timeline, comparison table.

---

## 4. Journal targets (ranked)

Strategy: this is a **framework/software + applications** paper. Two viable identities — (a) *software/tools* paper, (b) *applied energy + RL* paper. Pick based on how much quantitative control performance you can show.

### Tier 1 — best ranking, higher bar (need strong results)
1. **Applied Energy** (Elsevier, IF ~10–11, Q1). Best home if HVAC + PV/battery results are quantitatively strong (energy savings, comfort, self-consumption). Loves data-driven control of energy systems. *Highest impact, most competitive.*
2. **Energy and Buildings** (Elsevier, Q1). Excellent fit for the building-HVAC + EnergyPlus-FMU case studies; very receptive to RL+co-simulation for buildings. *High acceptance odds for this exact content.*
3. **IEEE Transactions on Smart Grid** (Q1) — only if you lean into the PV/battery/grid + multi-federation MARL angle. Stronger if you add a grid/OPF federate.

### Tier 2 — software/framework-friendly, strong ranking
4. **SoftwareX** (Elsevier, Q1/Q2, open access). *Purpose-built for research software.* Short format, fast, citable software artifact. **Excellent low-risk first publication** to stake the claim; can be followed by an Applied Energy applications paper. Strongly recommended as the framework-paper home.
5. **Energies** (MDPI, Q2/Q3, open access, fast). Receptive to RL+co-sim energy frameworks; lower bar, quick turnaround. Good fallback / fast option.
6. **Journal of Building Performance Simulation** (Taylor & Francis, Q2) — buildings-focused, methodology-friendly.

### Tier 3 — simulation/CSE venues
7. **SIMPAT — Simulation Modelling Practice and Theory** (Elsevier, Q1/Q2). Strong fit for the *co-simulation orchestration + timing/causality* contribution, less dependent on energy results.
8. **SoftwareX** (again, as artifact) / **Journal of Open Source Software (JOSS)** — lightweight software credit, pairs with any of the above.

### Recommended publication path
- **Step 1 (now-ish):** SoftwareX or SIMPAT — establish the framework + reset methodology + multi-formalism proof. Lower bar, fast, gives a citable artifact.
- **Step 2 (after stronger case studies):** Applied Energy or Energy and Buildings — applications paper with quantitative control wins, citing Step 1.
- This two-paper strategy maximizes total impact and de-risks the high-bar venue.

---

## 5. Storytelling spine

The narrative arc that makes reviewers nod:

1. **Hook (the pain):** "RL promises better control of energy CPS, but every team rebuilds the same fragile bridge between their simulator and their RL stack — and each rebuild is unreproducible and simulator-locked."
2. **Tension (why existing tools don't solve it):** building gyms are simulator-locked (Sinergym/BOPTEST); grid gyms are domain-locked and *synchronous fixed-frequency* (PowerGridworld); HELICS+RL studies are bespoke and unreproducible. None give you *declarative composition of arbitrary, multi-rate, multi-formalism physics* with a standard RL interface.
3. **Insight (the idea):** treat the scenario as a *compilation target*. A single YAML declares physics, wiring, timing, causality, and the RL task; the framework compiles it into a HELICS federation that *is* a Gymnasium env. Models — Python, FMU, EnergyPlus, data, agents — are interchangeable catalog entries.
4. **The hard part (credibility):** co-simulations don't `reset()` cheaply, and multi-rate physics creates algebraic loops. Show the reset-strategy methodology and the causality/auto-offset engine as the non-obvious contributions.
5. **Proof (it works):** same composable physics used for (a) pure co-sim verification and (b) RL training; DQN/SAC beat rule-based/PID baselines on buildings and PV-battery; reset-strategy ablation shows it matters; EnergyPlus-FMU case proves multi-formalism.
6. **Payoff (the vision):** a shared, declarative substrate for reproducible RL-on-physics across energy and beyond — the "Gymnasium for co-simulation." Future: MARL, distributed, ML surrogates, safe-RL, a standard benchmark suite.

**Through-line sentence:** *"Make the realistic physics the easy part, so the research can be about the agent."*

---

## 6. Case studies (to prove each claim)

Map each case study to the contribution it defends. Use the scenarios already in `src/scenarios/`.

| # | Scenario (existing YAML) | Proves | Key metrics |
|---|---|---|---|
| **CS0** | `simple_test.yaml`, `simple_test_multifederations.yaml` | Correctness + multi-federation + multi-rate orchestration | Analytical vs. simulated trajectory error; tick alignment; broker hierarchy works |
| **CS1** | `bui_hp_test_base.yaml` → `bui_hp_DQN.yaml`, `bui_hp_SAC.yaml`, `*_rollingreset.yaml` | RL beats RB/PID on realistic thermal physics; **reset-strategy ablation** | Comfort violation (°C·h), heating energy (kWh), reward curves, full vs rolling vs soft vs random reset |
| **CS2** | `bui0_fmu_test.yaml`, `Adelaide_test.yaml` (EnergyPlus FMU) | **Multi-formalism**: identical RL pipeline on an FMI 2.0 EnergyPlus model | Same agent API; control performance on FMU; FMU pulled from MinIO |
| **CS3** | `pv_batt_test_base.yaml` → `pv_batt_DQN.yaml`, `pv_batt_SAC.yaml` | RL energy management beats rule-based BEMS | Self-consumption %, grid import (kWh), SOC behavior, cost |
| **CS4** | FMI 3.0 `fmu_feedthrough_test.yaml` + a multi-rate combo | Timing/causality engine: `same/next_step`, auto-offset, type coverage | No algebraic-loop deadlock; correct deferred-input alignment; all FMI types round-trip |

**Strongest single figure:** the same building physics driven by (i) PID baseline, (ii) DQN, (iii) SAC, overlaid temperature + energy, with the comfort band shaded.

**The ablation that makes it a methods paper, not a demo:** learning curves for `full` vs `soft` vs `rolling` vs `random` reset on CS1 — quantify sample efficiency and final performance. *No competing framework can even pose this question.*

**Quantify composability:** report lines-of-YAML vs. lines-of-Python a user writes to go from a base co-sim to a trained agent (vs. estimated effort in a bespoke HELICS+Gym wrapper). Concrete evidence for the "zero-boilerplate" claim.

---

## 7. Future enhancements (raise significance; each is a paper-strengthener)

Ordered by impact-per-effort for the paper:

1. **Reproducibility + validation harness** *(do before submission).* Seeded runs, golden-trajectory tests for physics models, CI, a `make reproduce` that regenerates every figure. Reviewers of framework papers demand this; current repo has only dashboard tests.
2. **Standard benchmark suite + leaderboard.** Freeze CS1–CS3 as versioned benchmarks with baselines (RB, PID, MPC, random) and a scoring script. Turns "a framework" into "a benchmark the community can cite" — large citation multiplier.
3. **Multi-agent RL (MARL).** HELICS already routes many federates; expose multiple `rl_agent` federates → PettingZoo/RLlib MARL. Directly answers PowerGridworld's domain and CityLearn's district angle; opens IEEE TSG. *Highest scientific upside.*
4. **MPC baseline + safe-RL / action shielding.** Compare against MPC (the de facto strong baseline in BOPTEST) and add constraint shielding using catalog bounds. Makes results credible to control reviewers.
5. **ML-surrogate FMUs (learned models as FMUs).** Catalog already abstracts FMUs; add learned surrogates exported as FMUs (cf. DNV/Fraunhofer FMU-from-ML) → fast training, then transfer to high-fidelity FMU. Strong "digital twin" framing.
6. **Distributed / multi-host execution.** `multi_computer_config` is scaffolded; demonstrate scaling a large federation across machines → scalability section, appeals to CSE/SIMPAT.
7. **Sim-to-real / hardware-in-the-loop.** HELICS supports real-time; a small HIL demo (even a Raspberry-Pi thermostat) would be a standout figure and a transfer-learning story.
8. **Offline RL + dataset export.** Config already has `RLOfflineTrainingConfig`; ship logged datasets → offline-RL benchmark, a hot subfield.
9. **Uncertainty / domain randomization** via `random` reset + parameter sampling from catalog bounds → robustness study.

**Minimum set for a strong first submission:** #1 (reproducibility) + the reset ablation (CS1) + CS2 multi-formalism + CS3. **For a top-tier (Applied Energy/TSG) submission:** add #3 (MARL) or #4 (MPC + safe-RL).

---

## 8. Seed bibliography (verify + expand with DOIs)

Co-simulation / standards:
- Hardy et al., *HELICS: A Co-Simulation Framework for Scalable Multi-Domain Modeling and Analysis*, IEEE Access, 2024.
- FMI Standard / Blochwitz et al., *Functional Mock-up Interface* (FMI 2.0, 3.0); Wikipedia/fmi-standard.org for the standard overview.
- T&D / SMTD multi-timescale HELICS co-simulation, IEEE 2021/2022.

RL energy environments:
- Jiménez-Raboso et al., *Sinergym: a building simulation and control framework for training RL agents*, BuildSys 2021.
- Vázquez-Canteli et al., *CityLearn* (Gymnasium env for district energy MARL).
- Arroyo et al., *An OpenAI-Gym environment for BOPTEST*, 2021.
- *Energym* (building model library for controller benchmarking).
- *BEAR: Physics-Principled Building Environment for Control and RL*, 2022.

RL on co-simulation / power systems:
- Biagioni et al., *PowerGridworld: A Framework for Multi-Agent RL in Power Systems*, NREL, arXiv:2111.05969 (cite its time-stepping limitation explicitly).
- *Resilient Control of Networked Microgrids using Vertical Federated RL* (GridLAB-D+HELICS+Gym), arXiv:2311.12264 / 2212.08973.
- *CommonPower: A Framework for Safe Data-Driven Smart Grid Control*, arXiv:2406.03231.

RL tooling:
- Towers et al., *Gymnasium* (Farama).
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
- Liang et al., *RLlib / Ray*, ICML 2018.

Algorithms (as used): Mnih et al. DQN 2015; Haarnoja et al. SAC 2018.

> Action: pull exact DOIs/years; add 10–15 more domain refs (DRL for HVAC reviews, demand response RL surveys) to satisfy Applied Energy/Energy & Buildings reviewers.

---

## 9. Immediate next actions (checklist)

- [ ] Decide identity: SoftwareX/SIMPAT (framework-first) vs. Applied Energy (results-first). Recommend framework-first, then applications paper.
- [ ] Build reproducibility harness (#7.1) — gating item.
- [ ] Run CS1 reset-strategy ablation → the signature result.
- [ ] Run CS2 (EnergyPlus FMU) end-to-end → multi-formalism proof.
- [ ] Generate architecture + data-flow figures from the existing scenarios.
- [ ] Quantify config-only vs. bespoke-wrapper effort.
- [ ] Draft §3–§4 (architecture + reset) first; they are the contribution core.
```

---

### Sources consulted (literature grounding)
- [HELICS: A Co-Simulation Framework (IEEE Access 2024)](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/HELICSCoSimFramework.HardyEtAl.IEEEAccess2024.pdf)
- [PowerGridworld (arXiv:2111.05969)](https://arxiv.org/pdf/2111.05969) — note acknowledged synchronous fixed-frequency limitation
- [Sinergym (BuildSys 2021)](https://dl.acm.org/doi/10.1145/3486611.3488729)
- [CityLearn](https://www.citylearn.net/)
- [OpenAI-Gym environment for BOPTEST](https://www.researchgate.net/publication/354386346_An_OpenAI-Gym_environment_for_the_Building_Optimization_Testing_BOPTEST_framework)
- [BEAR: Physics-Principled Building Environment for RL (arXiv:2211.14744)](https://arxiv.org/pdf/2211.14744)
- [Resilient Networked Microgrids via Vertical Federated RL (arXiv:2311.12264)](https://arxiv.org/pdf/2311.12264)
- [CommonPower: Safe Data-Driven Smart Grid Control (arXiv:2406.03231)](https://arxiv.org/html/2406.03231)
- [FMI standard / FMU for digital twins (Fraunhofer)](https://publica.fraunhofer.de/entities/publication/d2324e95-fbe6-4cac-8f46-3697155f5400)
