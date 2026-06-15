# CosimGym — Two-Paper Execution Plan

> Companion to `PAPER_PLAN.md`. This file is the **what-needs-to-be-done** roadmap for two papers:
> - **Paper A — Software/Framework paper** (the tool itself).
> - **Paper B — Application paper** (an interesting case study made easy *because of* the tool).
>
> Case studies in Paper B are planned **incrementally**: each stage reuses the previous one's
> assets and raises ambition. Date drafted: 2026-06-15. Status: planning, no code changes implied.

---

## 0. Strategy in one picture

```
              ┌─────────────────────────────┐
              │  Shared foundation (do once) │
              │  repro harness · CI · figures│
              │  validated physics · metrics │
              └───────────────┬─────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                          ▼
  PAPER A (software)                         PAPER B (application)
  "Here is the tool"                         "Look what the tool makes easy"
  Venue: SoftwareX / SIMPAT                  Venue: Energy & Buildings → Applied Energy
  Sells: architecture, declarative           Sells: a result that is hard/impossible
         compiler, multi-formalism,                 elsewhere, obtained with config-only
         multi-rate, reset methodology              effort. Incremental case studies.
```

**Sequencing:** build the shared foundation first → submit **Paper A** (lower bar, stakes the claim, gives a citable artifact) → use the *same* validated scenarios to grow **Paper B** through Stages B1→B2→B3 and submit when results are strong. Paper B cites Paper A.

**Golden rule for both:** every claim is backed by a script that regenerates the figure from a YAML scenario + seed. If it is not reproducible, it is not in the paper.

---

## 1. Shared foundation (prerequisite for BOTH papers)

Do this once. Both papers depend on it. This is also the weakest part of the repo today (tests are dashboard-only, README says "early prototype").

### F1. Reproducibility + experiment harness  *(gating)*
- [ ] `experiments/` dir: one subfolder per experiment, each with `scenario.yaml`, `seed`, `run.sh`, expected outputs.
- [ ] Deterministic seeding end-to-end (env, agent, numpy, torch); record seeds in results.
- [ ] `make reproduce` (or `python -m experiments.run_all`) regenerates every figure + table.
- [ ] Pin environment: export exact `environment.yml` / lockfile; record HELICS, fmpy, SB3, torch versions; log a config + code git hash into each results folder.
- [ ] Results manifest (JSON) per run: scenario hash, seed, wall-time, metrics.

### F2. Physics validation (credibility of every downstream result)
- [ ] Golden-trajectory tests for built-in models (spring-mass-damper vs analytical; 1R1C building step response; heat-pump COP curve; battery SOC energy balance).
- [ ] FMU round-trip test (FMI 2.0 EnergyPlus + FMI 3.0 feedthrough all types) — already have `fmu_feedthrough_test.yaml`.
- [ ] Multi-rate + causality correctness test: deferred-input alignment, auto-offset breaks algebraic loops, tick normalization.

### F3. Metrics + baselines library
- [ ] Standard metric functions: comfort violation (°C·h), energy (kWh), cost (€), self-consumption %, grid import/export, constraint violations.
- [ ] Baseline controllers as catalog models: random, rule-based, PID (have it), **MPC** (build — the strong baseline reviewers expect), and a "do-nothing".
- [ ] Common evaluation protocol: fixed test horizon, same weather/data split, N seeds, report mean±std.

### F4. CI + quality
- [ ] GitHub Actions: install, run fast tests, run one tiny end-to-end scenario headless.
- [ ] Smoke test that every `src/scenarios/*.yaml` at least loads + validates (Pydantic) and runs K steps.
- [ ] Lint/format; bump README status from "prototype" once green.

### F5. Figures + diagrams toolkit
- [ ] Architecture diagram (ScenarioManager → brokers → federates → Redis → catalog).
- [ ] Auto data-flow diagram generator from any scenario YAML (pub/sub graph) — also a *feature* worth mentioning in Paper A.
- [ ] Plotting module producing publication-ready learning curves, trajectory overlays, ablation bars.

**Exit criterion for foundation:** `make reproduce` regenerates the spring-mass-damper verification figure + the CS1 building learning curves on a clean machine.

---

## 2. PAPER A — Software / Framework paper

**Identity:** research-software paper. **Target:** SoftwareX (fast, citable artifact) or SIMPAT (if you want a fuller architecture/co-sim treatment). **Length:** SoftwareX is short (~6 pp + metadata); SIMPAT allows full treatment.

**Thesis:** *CosimGym compiles a declarative YAML scenario into a running HELICS federation that is simultaneously a Gymnasium environment, unifying arbitrary multi-formalism, multi-rate physics behind one RL interface — with a principled solution to episodic reset over non-resettable co-simulations.*

### A-contributions (what reviewers must remember)
1. Declarative scenario compiler (YAML → typed Pydantic → orchestrated processes).
2. Catalog-driven, multi-formalism model abstraction (Python / FMI 2&3 / EnergyPlus / data / agents).
3. Multi-rate + causality + hierarchical multi-federation engine.
4. Episodic-reset methodology for non-resettable federations (`full/soft/rolling/random`).
5. Auto-derived RL spaces from physical I/O bounds.

### A-tasks
- [ ] **Minimal but complete validation** (not full case-study depth): one verification scenario (spring-mass-damper analytical match), one RL scenario (building, DQN+SAC beat PID), one FMU scenario (proves multi-formalism), one multi-rate/causality demo. These exist as YAMLs — wire them to `make reproduce`.
- [ ] **Reset-strategy micro-benchmark** — the signature methodological figure. Even small: full vs soft vs rolling vs random on CS1, learning curves + final performance. *No competitor can pose this.*
- [ ] **Composability evidence** — table: lines-of-YAML + lines-of-Python to go base co-sim → trained agent, vs estimated effort for a hand-built HELICS+Gym wrapper. Concrete "zero-boilerplate" proof.
- [ ] **Software-paper requisites:** clear install, API/extension docs ("add a model in N lines"), licensing, versioned release + DOI (Zenodo), code metadata, example gallery.
- [ ] Write architecture section (the core), reset section (the novelty), short results.
- [ ] Honest limitations: single-agent now, single-host focus, prototype maturity → frame as roadmap.

### A-figures
1. Architecture diagram. 2. YAML→runtime compilation flow. 3. Reset-strategy ablation. 4. One learning curve + trajectory. 5. Comparison table (from `PAPER_PLAN.md` §2.1).

### A-exit criterion
Reviewer can `pip/conda install`, run an example from the paper, and reproduce one figure. Artifact has a DOI.

---

## 3. PAPER B — Application paper (incremental case studies)

**Identity:** applied-energy results paper. The case study must show a result that is **hard or impossible to obtain elsewhere** but **easy in CosimGym** — multi-formalism + multi-rate + reset semantics doing real work.

**Build it in three stages.** Each stage is publishable-on-its-own evidence and a strict superset of the previous. Submit when Stage B1 (minimum) or B1+B2 (strong) is solid; B3 is the high-ceiling extension.

### Stage B1 — Multi-rate building + EnergyPlus-FMU + RL  *(minimum viable application paper)*
**Why first:** highest realism-per-effort, uses existing FMU assets (`bui0_fmu_test.yaml`, `Adelaide_test.yaml`), and directly showcases the FMU + multi-rate combo that Sinergym/BOPTEST can't compose.

- [ ] Compose federation: weather (1 h) → EnergyPlus-FMU building (10 min) → heat pump (1 min) → RL agent. Different `real_period` per federate = multi-rate proof.
- [ ] Train DQN + SAC; baselines PID + **MPC** + rule-based.
- [ ] Metrics: comfort violation (°C·h), heating energy (kWh), cost, COP utilization.
- [ ] **Reset-strategy ablation on a real FMU** (full/soft/rolling/random) — the methodological differentiator carried into application.
- [ ] Robustness: test on a different weather year / climate (TO vs Adelaide) — generalization story.
- [ ] **Effort claim:** show the entire study is config-only (diff between base-cosim YAML and RL YAML).

**B1 venue:** Energy and Buildings (best fit) or Journal of Building Performance Simulation.

### Stage B2 — Add generation + storage (PV + battery coupling)
**Why second:** turns "control one building" into "manage a prosumer," adds an economic objective, broadens to Applied Energy.

- [ ] Couple PV (`pv_dest`) + battery + load with the B1 building (cross-federation subscriptions).
- [ ] RL jointly controls HVAC modulation + battery dispatch (multi-variable action space — auto-derived from catalog bounds).
- [ ] Baselines: rule-based BEMS (have `rb_bems`) + PID + MPC.
- [ ] Metrics: self-consumption %, grid import/export (kWh), energy cost under a tariff, comfort kept.
- [ ] Show multi-objective trade-off (comfort vs cost vs self-consumption) — Pareto-ish frontier across reward weightings.

**B2 venue:** Applied Energy (strong) — the multi-objective prosumer result is its sweet spot.

### Stage B3 — Coupled district / multi-federation (high ceiling)
**Why last:** showcases the unique hierarchical multi-federation feature; opens IEEE TSG; natural MARL extension. Needs the most new work.

- [ ] Multiple buildings (each its own federation) + shared PV/storage/grid federation, coupled by the hierarchy broker (`--sub_brokers`).
- [ ] Single-agent coordinator first; then **MARL** (one `rl_agent` federate per building) once multi-agent support lands (see `PAPER_PLAN.md` §7.3).
- [ ] Add a simple grid/OPF or power-balance federate for grid-aware rewards (cf. PowerGridworld's OPF-in-the-loop, but multi-rate + multi-formalism).
- [ ] Metrics: peak shaving, transformer/feeder limit violations, fairness across buildings, total cost.
- [ ] Scalability: runtime vs number of federations/federates; optionally multi-host (`multi_computer_config`).

**B3 venue:** IEEE Transactions on Smart Grid / Applied Energy.

### B-narrative spine
"Realistic control studies usually force a choice: realistic physics (hard to wire to RL) *or* a convenient RL interface (toy physics). With CosimGym the realistic, multi-rate, multi-formalism physics is config-only — so we can study [B1 building → B2 prosumer → B3 district] with the same agent code, and we quantify what that realism buys (and what reset strategy / multi-rate fidelity costs)."

---

## 4. Division of content (avoid self-overlap / salami concerns)

| Content | Paper A (software) | Paper B (application) |
|---|---|---|
| Architecture, compiler, catalog | **Full** | Cite A, 1 short paragraph |
| Reset methodology | **Full (method)** | **Use + ablate on real FMU (result)** |
| Multi-rate/causality engine | **Full** | Cite + exploit |
| Verification (spring-mass-damper) | Yes (brief) | No |
| Quantitative control wins vs MPC | Minimal (1 demo) | **Full, multiple baselines/stages** |
| Multi-objective / district results | No | **Full (B2/B3)** |
| Effort/composability claim | **Headline** | Supporting |

Two papers are distinct: A is *method + tool*, B is *scientific result enabled by the tool*. The reset ablation appears in both but in different roles (method demo vs applied result) — acceptable and common.

---

## 5. Consolidated checklist (ordered)

**Phase 0 — Foundation (blocks everything)**
- [ ] F1 repro harness + seeding + `make reproduce`
- [ ] F2 physics golden tests + FMU round-trip + multi-rate correctness
- [ ] F3 metrics lib + baselines (incl. MPC)
- [ ] F4 CI + scenario smoke tests
- [ ] F5 figure/diagram toolkit + architecture diagram

**Phase 1 — Paper A (software)**
- [ ] Wire 4 minimal scenarios to repro harness
- [ ] Reset micro-benchmark figure
- [ ] Composability table
- [ ] Docs + Zenodo DOI release
- [ ] Draft A → submit SoftwareX/SIMPAT

**Phase 2 — Paper B (incremental)**
- [ ] B1 building+FMU+multi-rate, DQN/SAC vs PID/MPC, reset ablation, climate generalization
- [ ] (strong) B2 add PV+battery, multi-objective prosumer
- [ ] (high ceiling) B3 district multi-federation, optional MARL + grid federate + scalability
- [ ] Draft B → submit Energy and Buildings (B1) or Applied Energy (B1+B2)

---

## 6. Risks & mitigations
- **Maturity/repro gap** → Phase 0 is non-negotiable; do not draft results before `make reproduce` works.
- **Reviewer asks "why not Sinergym/BOPTEST?"** → lead with multi-formalism + multi-rate + composability; B1 is literally a thing they cannot compose.
- **MPC baseline missing** → build it in F3; without it, applied-energy reviewers discount RL wins.
- **Salami-slicing concern** → §4 division table; A=method, B=result; cite each other.
- **Single-agent limitation** → keep B3/MARL as explicit roadmap; don't claim MARL until implemented.

---

### Pointers
- Novelty/gap analysis, journal tiers, bibliography: see `PAPER_PLAN.md`.
- Existing scenarios to reuse: `src/scenarios/` (building+HP+RL, PV+battery+RL, FMU tests, multi-federation).
