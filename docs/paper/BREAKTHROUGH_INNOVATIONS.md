# CosimGym — Breakthrough Innovations from the Architecture's Potential

> Forward-looking companion to `PAPER_PLAN.md` / `TWO_PAPER_PLAN.md`.
> Question answered: *given what the architecture makes possible (not just current
> prototype status), where are the genuine breakthroughs?*
> Each item: the bet · why CosimGym is uniquely positioned · what to build ·
> the experiment that proves it · venue · honesty check.
> Drafted 2026-06-15. Planning only.

The recurring pattern: CosimGym's three structural properties —
**(1) declarative composition**, **(2) multi-formalism catalog**, **(3) native
multi-rate + federated communication** — are exactly the ingredients several
open RL research frontiers are missing. The breakthroughs are where one of those
properties removes a constraint that the literature currently treats as fixed.

---

## North star
**"The data engine for control of cyber-physical energy systems."** Just as large
datasets + a generation pipeline unlocked vision and language, a *generator of
diverse, realistic, multi-rate physics environments* could unlock *generalist*
energy-control agents. CosimGym's catalog + compiler is that generator. Every
item below is a step toward, or a spin-off of, that north star.

---

## Tier 1 — Championable (defensible, architecture-native, paper-ready with effort)

### BK1. Multi-timescale / hierarchical RL on natively multi-rate physics
- **The bet:** most RL assumes a single fixed timestep; real energy control spans
  seconds (inverter/HVAC) to hours (markets/weather/thermal mass). An agent that
  *acts and learns across heterogeneous timescales* is under-served because no
  testbed exposes true multi-rate dynamics cleanly.
- **Why CosimGym:** multi-rate is already first-class (`real_period` → normalized
  ticks; `same_step/next_step` causality; hierarchical federations). A fast control
  federate and a slow planning federate can coexist by construction.
- **What to build:** a hierarchical-RL interface — multiple RL federates at
  different `real_period`s (e.g., a slow setpoint/planner agent feeding a fast
  tracking agent), or options/temporal-abstraction wrappers aligned to ticks.
- **Proof:** show a multi-timescale agent beats a single-rate agent on a
  building+storage task where slow (thermal mass / price) and fast (comfort)
  dynamics conflict; ablate the timescale separation.
- **Venue:** NeurIPS/ICLR workshop or IEEE TSG + an energy journal.
- **Honesty:** needs hierarchical-RL plumbing (new code); the testbed is the easy
  part, the algorithm is the contribution. Strong because the substrate is unique.

### BK2. Generalist energy-control agents via domain randomization over the catalog
- **The bet:** policies overfit to one building/topology. A *distribution* of
  composed physics → agents that generalize / transfer / few-shot adapt.
- **Why CosimGym:** models are interchangeable catalog entries with typed,
  bounded parameters. Sampling parameters and re-wiring topology = sampling
  environments. The compiler turns a sampler into an environment factory.
- **What to build:** a scenario sampler that randomizes catalog parameters
  (within bounds), swaps model variants (1R1C ↔ EnergyPlus FMU), and perturbs
  topology; train one policy across the distribution.
- **Proof:** zero/few-shot transfer to unseen buildings/climates vs per-environment
  baselines; sim-to-sim transfer (simple model → FMU). This is the "data engine"
  made concrete.
- **Venue:** Applied Energy / IEEE TSG; ML venue for the transfer angle.
- **Honesty:** the headline ("foundation agent for energy") must be earned with
  breadth of environments; start with a modest distribution and scale.

### BK3. Multi-agent RL with *realistic* communication, not abstracted away
- **The bet:** MARL papers usually assume perfect, instantaneous communication.
  Real distributed energy control has latency, partial observability, message
  loss, and topology — which change what policies work.
- **Why CosimGym:** HELICS endpoints + federated time can model message passing,
  delays, and rates *as part of the simulation*. PowerGridworld explicitly lists a
  *limited communication model* as a limitation — this is the direct gap to fill.
- **What to build:** one RL federate per subsystem (PettingZoo/RLlib MARL),
  inter-agent comms over HELICS endpoints with configurable latency/dropout;
  `next_step` causality as a built-in one-step comms delay.
- **Proof:** show policy performance degrades and coordination strategies change
  as comms latency/loss increase — a result other gyms structurally cannot produce.
- **Venue:** IEEE TSG / AAMAS / NeurIPS workshop.
- **Honesty:** requires MARL support (roadmap item); the comms-realism framing is
  the differentiator, so design the comms model carefully and measure it.

### BK4. Config-only sim-to-real / digital-twin-in-the-loop
- **The bet:** the sim-to-real gap is partly an *engineering* gap — re-plumbing
  from simulator to hardware. Make the swap a one-line config change.
- **Why CosimGym:** HELICS supports real-time execution; a model federate and a
  hardware/SCADA federate are the same abstraction. Swap `type: model` → real
  device federate; everything else (agent, wiring, reward) is unchanged. The reset
  methodology (`soft/rolling`) is exactly what irreversible real systems need.
- **What to build:** a real-time federate adapter (e.g., MQTT/Modbus → HELICS) and
  demonstrate the *identical* scenario YAML running (a) fully simulated and (b) with
  one federate replaced by hardware (even a small thermostat / HIL rig).
- **Proof:** a policy trained in sim, deployed via the same scenario with a hardware
  federate; report the (small) config diff and transfer performance.
- **Venue:** Applied Energy / IEEE TSG; standout demo figure.
- **Honesty:** needs hardware + real-time adapter; high-impact but higher logistics.

---

## Tier 2 — High-upside, lower-effort credibility multipliers

### BK5. Automatic physics-grounded safety shields from catalog metadata
- **The bet:** safe RL usually hand-codes constraints. The catalog *already* holds
  typed I/O bounds (`min_value/max_value/unit`) — these are latent safety
  specifications.
- **Why CosimGym:** spaces are already auto-derived from bounds; extend to an
  automatic action-shield / constraint layer compiled from the same metadata.
- **What to build:** a shield wrapper that clips/projects actions and flags
  observation-constraint violations using catalog bounds; optional CBF-style layer.
- **Proof:** zero hard-constraint violations vs unsafe baseline at minimal
  performance cost — "safety for free from the model registry."
- **Venue:** sub-result inside Paper B, or CommonPower-style safe-control venue.
- **Honesty:** clipping is simple; the *novel* framing is "shields derived
  automatically from model metadata," so emphasize the auto-derivation.

### BK6. An offline-RL dataset factory + benchmark for energy CPS ("D4RL for energy")
- **The bet:** offline RL for energy lacks standardized, realistic datasets.
- **Why CosimGym:** train/test-partitioned storage + many composable scenarios +
  `RLOfflineTrainingConfig` already exist. Logged trajectories across the scenario
  distribution = a dataset generator.
- **What to build:** export standardized offline datasets (random/PID/MPC/RL
  behavior policies) across versioned scenarios; a loader + scoring script.
- **Proof:** release the dataset + baselines; offline-RL algorithms ranked on it.
- **Venue:** dataset/benchmark track (NeurIPS D&B) + energy journal.
- **Honesty:** value comes from breadth + curation, not novelty of any single run.

### BK7. Reset semantics for non-resettable simulators as a recognized RL sub-problem
- **The bet:** "episodic RL over irreversible/streaming simulators" is a real,
  underformalized problem (FMUs, real-time twins, physical plants).
- **Why CosimGym:** the `full/soft/rolling/random` mechanism + causality-aware
  staging is a concrete, ablatable instantiation.
- **What to build:** formalize the problem; benchmark reset strategies across
  several tasks; analyze bias/variance and sample-efficiency trade-offs.
- **Proof:** the reset ablation generalized into a method paper / position.
- **Venue:** RL workshop / methods note; reinforces Papers A and B.
- **Honesty:** more methodological than flashy, but it is genuinely novel framing
  and cheap to elevate from the existing feature.

---

## Tier 3 — Speculative, high-ceiling (flag as vision, do not over-claim)

### BK8. LLM-driven declarative experimentation (self-driving co-simulation research)
- **The bet:** because scenarios are declarative YAML with a typed schema, an LLM
  agent can *synthesize, mutate, run, and analyze* experiments — automated
  curriculum design and hypothesis search over physics+control.
- **Why CosimGym:** the Pydantic schema is a machine-checkable action space for an
  LLM; the compiler executes proposals; telemetry feeds results back.
- **What to build:** an agent loop that proposes scenario YAMLs (curriculum /
  ablations), runs them, reads telemetry, and iterates toward an objective.
- **Proof:** auto-discovered curriculum or scenario that improves a target metric
  beyond a human-designed baseline.
- **Venue:** vision/workshop; risky but on-trend (agentic science).
- **Honesty:** clearly speculative; present as future direction, not a result.

### BK9. Control-and-design co-optimization
- **The bet:** jointly optimize *physical design* (battery kWh, HP rating, envelope)
  and the *control policy* — usually done sequentially.
- **Why CosimGym:** design parameters live in the catalog with bounds; an outer
  loop (BO/evolutionary) over parameters wraps the inner RL loop on the same YAML.
- **Proof:** co-optimized design+policy dominates sequential design-then-control on
  cost/comfort.
- **Venue:** Applied Energy / design-automation venue.
- **Honesty:** compute-heavy; scope to one subsystem first.

---

## Recommended sequencing (impact vs. effort)
1. **Now (strengthens current papers):** BK7 (reset framing) + BK5 (auto safety shields) — cheap, reuse existing features.
2. **Next flagship (own paper):** BK1 (multi-timescale RL) **or** BK3 (realistic-comms MARL) — both architecture-native and fill named gaps. Pick by which algorithm work you want to own.
3. **Data-engine play:** BK2 (generalist via domain randomization) + BK6 (offline dataset) — together they realize the north star and produce community assets (citation multipliers).
4. **Standout demo:** BK4 (config-only sim-to-real) when hardware is available.
5. **Vision section / grants:** BK8, BK9 — name them, don't claim them.

## Single sentence to pitch the vision
*"CosimGym can become the environment-generation engine that lets energy-control
agents learn across realistic, multi-rate, multi-formalism physics at scale —
turning today's bespoke, single-simulator RL studies into reproducible, transferable,
and eventually generalist control research."*
