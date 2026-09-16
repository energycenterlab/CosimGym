# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repo.

It is deliberately **thin on feature detail**. Feature semantics live in `docs/` and,
ultimately, in the code. What lives here is what you cannot get from either: the working
rules, the orientation map, and the dev workflow.

## What This Project Is

CosimGym = Python orchestration framework. Bridges HELICS co-simulation with Gymnasium RL.
Define multi-federate scenarios declaratively in YAML, run as pure physics co-sim or RL
training/testing env.

## Working Rules (READ FIRST — apply to every session)

Standing agreement for how Claude works in this repo. Overrides default habits. Only the
user can change it.

### The only reference documents

Besides `docs/` (the user-facing documentation) and the codebase itself, exactly **three**
reference documents exist:

| File | Role |
| --- | --- |
| `CLAUDE.md` (this file) | Project description + the rules. The durable "how things are". |
| `HANDOFF.md` (repo root) | State of the *last* session, for the *next* session. Rewritten each session. |
| `docs/KNOWN_ISSUES.md` | The single register of everything currently broken or limited. |

`docs/KNOWN_ISSUES.md` is append-and-prune, never a changelog: **add an entry the moment a
bug or limitation is found, delete the entry the moment it is fixed or stops mattering.**
Nothing accumulates there for history — the story of a fix belongs in that feature's
changes report. Keep its `KNOWN_FAIL` counterpart in `tests/regression_suite.py` in sync
(an UNEXPECTED-PASS means: drop the `KNOWN_FAIL` entry *and* delete the issue entry). Do
not open a second bug list anywhere.

Do not create other top-level reference/status/summary/index `.md` files. No
`SUMMARY.md`, `NOTES.md`, `STATUS.md`, `README_<feature>.md`, no "here's what I found"
scratch docs in the repo root or `src/`. Use the scratchpad dir for throwaway notes.

### The only three kinds of support document

When work on a feature needs written support, it is one of exactly three types, and each
has one home:

| # | Type | Where | Filename convention |
| --- | --- | --- | --- |
| 1 | **Plan of changes** — written *before* implementing | `docs/changes_reports/` | `<feature>_plan.md` |
| 2 | **Changes report** — written *after* implementing: what was done, what was modified, **why**, problems hit, alternatives considered and rejected | `docs/changes_reports/` | `<feature>_changes.md` |
| 3 | **Future / TODO** — future enhancements, deferred work, objective changes | `docs/future_and_TODOs/` | `<topic>_followups.md`, `<topic>_plan.md`, `<topic>_TODO.md` |

(Known bugs are *not* a type-3 document — they go in the single
`docs/KNOWN_ISSUES.md` register described above. while limitation are a type-3 and a known issue in the type 3 you specify more detailed the problem, in KNOWN_ISSUES.md you only list it)

Rules for these:

- **One document per feature per type.** Revising a plan = edit the existing
  `<feature>_plan.md`, never `<feature>_plan_v2.md` / `_revised` / `_final`.
- **Write a support doc only when the user asks for one**, or when the work is large
  enough that a plan is genuinely needed. Small fixes get a commit message, not a doc.
- Plan and report for the same feature share the same `<feature>` stem, so they pair up.
- When a plan is fully implemented, mark it: rename to `<feature>_plan_DONE.md` or put a
  **STATUS: DONE** line at the top (existing examples of both are in the tree).
- A type-3 doc whose items are all done gets deleted or marked DONE — do not let
  `future_and_TODOs/` accumulate stale entries.
- Nothing of these three types goes anywhere else in the tree, including `docs/` root.

### Documentation upkeep (do not skip)

After any change that touches **code fundamentals** (architecture, execution flow, core
classes) or **how the user interacts with the framework** (YAML schema keys, CLI
commands, config defaults, catalog format, file layout of results):

1. Update the affected pages under `docs/` — they are the user-facing contract.
2. Update `CLAUDE.md` **only if** the change alters something this file states or should
   state (a new config key, a new subsystem, a changed default). Routine fixes,
   refactors and bug fixes do **not** earn a CLAUDE.md edit. Keep this file dense.
3. Add a scenario (and where relevant a combination) to `tests/regression_suite.py` — it
   is the living feature contract.

### Handoff routine (every session)

Before a session ends — and whenever the user says "handoff" — invoke the `handoff`
skill and **rewrite** `HANDOFF.md` to describe *this* session only: goal, current
progress, what worked, what did not work (so it is not retried), next steps, and any
uncommitted/in-flight state. Replace stale content rather than appending to it; the
durable parts belong in `CLAUDE.md` or a type-2 changes report, not in a growing
HANDOFF. Start of session: read `HANDOFF.md` before acting.

### Scope discipline

- Do what was asked. No speculative refactors, no drive-by renames, no "while I was in
  there" edits to files outside the task.
- Prefer editing an existing file over creating a new one.
- Do not create new top-level directories without asking.
- Before large or long real-hardware/benchmark runs, get an explicit go-ahead.
- Never commit unless the user asks.

## Where the truth lives

Three sources, in this order of authority. **The code is the only one that cannot be
stale.**

1. **The code graph** — `graphify-out/`. Start here for any structural question ("what
   calls X", "what does this change touch", "where does this flow go"). Cheaper and more
   accurate than grep, and it is regenerated from the AST, so it cannot drift into
   fiction the way prose can.
2. **The code** — for exact semantics, defaults, validation rules, error messages. When a
   doc and the code disagree, the code is right and **the doc is a bug**: fix the doc as
   part of the task (see *Documentation upkeep*).
3. **`docs/`** — for intent, rationale and the user-facing contract. Treat it as a map,
   not as the territory. It can lag the code or omit a case. Never quote a default, a key
   name or a limit from a doc into an answer or into new code without confirming it in
   the source.

Corollary: do **not** answer a question about how something behaves purely from `docs/` or
from this file. Confirm it in the graph or the source first. A confident wrong answer read
off a stale doc is worse than a slow one.

### Using the graph

```bash
graphify query "<question>"        # scoped subgraph — first move for codebase questions
graphify path "<A>" "<B>"          # how two things relate
graphify explain "<concept>"       # focused concept view
graphify update .                  # AST-only, no API cost — run after code changes
```

`graphify-out/wiki/index.md` for broad navigation; `graphify-out/GRAPH_REPORT.md` only for
whole-architecture review, or when query/path/explain surface too little. The
`code-review-graph` MCP tools cover the same graph: `semantic_search_nodes`,
`get_impact_radius`, `detect_changes`, `query_graph` (callers_of / callees_of / imports_of
/ tests_for). Prefer either over Grep/Glob/Read for exploration; fall back to raw reads
only for exact lines.

## Orientation

Enough to know where to look. Details are in the linked docs; exact behavior is in the code.

### Execution flow

1. **`ScenarioManager`** (`src/core/ScenarioManager.py`) — reads the YAML scenario, starts
   HELICS brokers as subprocesses, spawns each federate as its own Python process via
   `federate_launcher.py`. The whole scenario config goes through Redis so each subprocess
   can retrieve it.
2. **`federate_launcher.py`** (`src/core/federate_launcher.py`) — per-federate entry point.
   Reads config from Redis, instantiates the federate class for its `type`.
3. **`BaseFederate`** (`src/core/BaseFederate.py`) — HELICS pub/sub lifecycle, time
   stepping, storage, reset. Instantiates models from the catalog, drives `_step()`.
   `InterfaceFederate` subclasses it for digital-twin bridging.
4. **`RLFederate` / `HelicsGymEnv`** (`src/core/RL_Federate.py`) — wraps `BaseFederate` as
   a Gymnasium `Env`, routing observations/actions through HELICS.

### Config pipeline

Scenario YAML (`src/scenarios/`) → `src/utils/config_reader.py` → typed dataclasses in
`src/utils/config_dataclasses.py` (`ScenarioConfig`, `FederationConfig`, `FederateConfig`,
…) → runtime. The dataclasses are the schema of record — read them, not a doc table, when
you need the exact set of keys, types and defaults. RL config is Pydantic with
`extra='forbid'`, so a YAML typo raises at parse time.

### Model catalog

`src/models/model_catalog/catalog.yaml` is the static registry: `model_name` → `class_name`
+ `module_path` + I/O spec. `catalog_loader.py` pushes it into Redis at startup;
`RedisCatalog` resolves it at runtime so `BaseFederate` can import and instantiate the
class dynamically. New model = subclass `BaseModel` (`initialize` / `step` / `finalize`) +
a `catalog.yaml` entry. After editing `catalog.yaml`, reload it into Redis or the change is
not seen. → [Custom Models & Catalog](docs/user_guide/custom_models.md)

### Feature map → documentation

| Feature | Read |
| --- | --- |
| Scenario YAML, all top-level keys | [Scenario Configuration](docs/user_guide/scenario_configuration/overview.md) |
| Timing, `real_period`, offsets, causality | [Synchronization & Causality](docs/user_guide/scenario_configuration/synchronization.md) |
| Federations, brokers, multi-federation hierarchy | [Federation Configuration](docs/user_guide/scenario_configuration/federation.md) |
| Federate keys, storage sinks, parallel model execution | [Federate Configuration](docs/user_guide/scenario_configuration/federate.md) |
| RL config (environment / agent / run / experiment) | [RL Configuration](docs/user_guide/scenario_configuration/rl.md), [RL Integration](docs/user_guide/rl_integration.md) |
| Digital-twin interfaces, streaming, overrides, BK4 | [Digital-Twin Interfaces](docs/user_guide/digital_twin_interfaces.md) |
| Distributed SSH federate spawning | [Distributed Deployment](docs/user_guide/distributed_deployment.md) |
| FMUs, `max_sim_time`, `simulation_horizon`, reset | [FMU Models](docs/user_guide/fmu_models.md) |
| Results, dashboard, live view | [Dashboard & Analytics](docs/user_guide/dashboard.md) |
| Ports, `src/.env`, shared-machine conflicts | [Installation → Configuring ports](docs/Installation_Setup.md#configuring-ports) |
| Currently broken / limited | [Known Issues](docs/KNOWN_ISSUES.md) |

Scenario catalogues: `src/scenarios/scenario_tests.md`, `src/scenarios/paper_casestudies.md`.

## Setup & Common Commands

**Prerequisites:** Conda, Docker, Docker Compose **v2** (`docker compose` plugin — verify
`docker compose version` reports `v2.x`; legacy `docker-compose` v1 rejects the Compose
Spec file), Python 3.12. Shared server without sudo: install the Compose v2 plugin into
your home only. Full instructions: [Installation & Setup](docs/Installation_Setup.md).

```bash
docker compose -f src/docker-compose.yaml up -d   # Redis + MinIO + Mosquitto — required
conda activate cosim_gym
python src/test_script.py        # base co-simulation scenarios
python src/test_script_rl.py     # RL training scenarios
./src/dashboard/run_dashboard.sh # Results + Live dashboard
```

Run every simulation script from the project root with `cosim_gym` active. Redis must be
up first.

### Pre-merge regression suite

`tests/regression_suite.py` is the living feature contract: `pytest` plus ~32 scenarios
covering every feature axis **and** explicit feature combinations (dist+multifed,
dist+parallel, multifed+parallel, parallel+grid). Each runs in an isolated subprocess; long
scenarios are auto-shortened on a throwaway temp copy, originals untouched. It prints a
PASS/FAIL/xfail table and exits non-zero only on an unexpected hard failure.

```bash
conda run -n cosim_gym python tests/regression_suite.py              # needs docker up + passwordless ssh to 127.0.0.1
RUN_CLOUD=1 conda run -n cosim_gym python tests/regression_suite.py  # also the cloud-machine distributed scenarios
```

**Add a scenario — and a combination — there whenever you add a feature.** Three lists:
`SCENARIOS` + `COMBOS` (expected PASS) and `KNOWN_FAIL` (tracked bugs run as `xfail` so the
gate stays green). A `KNOWN_FAIL` that starts passing is flagged UNEXPECTED-PASS — remove
it from the dict *and* delete its entry in `docs/KNOWN_ISSUES.md`.

Run it before merging any feature branch back to `main`.
