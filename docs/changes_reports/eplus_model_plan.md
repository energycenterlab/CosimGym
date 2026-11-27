# EnergyPlus (Python API) generic model: plan

Branch: `eplus_model`. Status: **IMPLEMENTED** (approved 2026-11-27); results and limits in `eplus_model_changes.md`.
**Revised after implementation:** the "catalog entry per building + `register` CLI" parts (sections on the catalog contract and the CLI) were replaced by one generic `eplus_model` entry with the building and `eplus_io` declared in the scenario; see `eplus_model_changes.md` and `docs/user_guide/eplus_models.md`.
Scope rule: work only on the model side (`src/models/`, catalog, tools, tests, docs). No change to `src/core/`, `utils/`, `ScenarioManager`.

## 1. Goal

One generic CosimGym model that co-simulates any EnergyPlus IDF through the EnergyPlus Python API
(`pyenergyplus`), following the standard model integration (subclass `BaseModel`, catalog entry,
`initialize` / `step` / `finalize` / `reset`). Inputs and outputs are declared, not hard-wired in the IDF.

Non-goals: FMU export path (stays `BaseFMUModel`); changing HELICS/federate behavior; a GUI for IDF editing;
Windows/Mac installs of EnergyPlus (only discovery must not break there).

## 2. Verified facts (repo and machine)

| Fact | Source |
| --- | --- |
| Model contract: `BaseModel.__init__` calls `_instantiate()` then `initialize()`; per tick `_step()` -> `step()`; `reset(mode)` hook; horizon restart handled by `BaseModel._step` via `max_sim_time` / `sim_start_date` | `src/models/base_model.py` |
| Only catalog-declared inputs/outputs are initialized in `state`; others are skipped with a warning. RL spaces read catalog bounds | `base_model._instantiate`, `RL_Federate.py:552` |
| The model's input/output names come from the scenario `connections` (subscribes/publishes keys) | `BaseFederate.input_output_names` |
| `BaseFMUModel` is the reference for EnergyPlus-in-CosimGym (horizon, rolling reset, workdir in the run log dir, catalog `user_defined`) | `src/models/base_FMU_model.py` |
| `fmu_catalog_register.py` auto-generates a catalog entry from a model and already has `parse_idf_run_period` | `src/models/model_catalog/` |
| `BaseFederate.finalize()` does **not** call entity `finalize()` (latent bug, still present, core is off-limits) | `BaseFederate.py:1115`, memory note |
| `ScenarioManager` enforces whole-day `rolling_window` when the model's `user_defined.fmu_engine == 'energyplus'` | `ScenarioManager.py:1516` |
| Machine: Ubuntu 20.04, glibc 2.31, no EnergyPlus, no `pyenergyplus` | shell |
| IDF versions: BUI0 23.1, EC_IdealLoads 24.1, four 2_zones_* 26.1. One EPW (San Francisco TMY3) | `resources/` |

To verify before relying on them (not verified yet): EnergyPlus 26.1 Linux build requirements (25.x lists Ubuntu 22.04/24.04 only);
exact `pyenergyplus` API names per version (they changed across 9.x -> 2x.x); contents of the six IDFs (zones, RunPeriod, Timestep, existing Output:Variable / actuators).

## 3. Core design problem: control inversion

`run_energyplus(state, args)` is blocking and calls us back; CosimGym calls `step()` once per tick.
Solution: EnergyPlus runs in a **worker** (separate execution context); callbacks are the rendezvous points.

```
CosimGym federate           EnergyPlusModel (parent)            worker (own process, one E+ version)
  _step(inputs) --------->  step(): send {t, inputs} ---------> callback BEGIN zone timestep:
                                                                  wait for command, set actuators
                                                                E+ advances one zone timestep(s)
  outputs <-------------   step(): recv {outputs} <------------ callback END zone timestep:
                                                                  read variables/meters/actuators, reply
```

- Use **zone-timestep** callbacks, not system-timestep ones (HVAC sub-steps fire system callbacks several times per zone step).
- `real_period` must equal the IDF zone timestep (`3600 / Timestep`) or an integer multiple N (worker holds the inputs for N zone steps and returns the last outputs; policy for averaged vs last-value outputs is a per-output option). Validated in `initialize()`; mismatch = clear error naming the IDF value.
- Warmup and sizing are skipped from the CosimGym point of view: the first `step()` is the first real timestep.

## 4. Version handling (decided: registry + subprocess + auto-upgrade in cache)

- One E+ cannot be imported twice in a process and the shared library can crash the host; so **one worker subprocess per model instance**, started with the chosen install's `pyenergyplus` on its `sys.path`. Worker entry module lives in `src/models/` and imports nothing from helics.
- **Registry** of installs: env var `COSIMGYM_ENERGYPLUS_DIRS` (path list) and `~/.cosimgym/energyplus/<version>/`; each entry is validated by running `energyplus --version`. No hardcoded `/home` or `/Users` paths.
- **Resolution** per model instance: read IDF `Version`; exact match -> use it; otherwise the lowest install **newer** than the IDF -> auto-upgrade via the install's Transition binaries into `~/.cosimgym/eplus_cache/<hash(idf)>/<target>/` (original untouched; log source version, target version, Transition warnings; cache keyed by IDF content hash + target version). No install >= IDF version -> error listing the installs found and what is needed. Downgrade is impossible by design.
- Override in `user_defined`: `eplus_version` (force an install) and `eplus_dir` (explicit path).
- Remote machines (distributed SSH, parallel executor) run the same discovery on their own host; the plan must not assume the registry is identical across machines. Document in `distributed_deployment.md`.
- Known gap: this machine cannot natively run 26.1 (likely). The registry keeps the door open for a non-native worker (container) later; a worker is just "a command that speaks the pipe protocol". Not in this plan's phases unless the user asks.

## 5. Declarative, dynamic I/O (answer to "does the IDF have to be prepared?")

No: for anything EnergyPlus already exposes, the IDF does not have to be touched.

- **Outputs**: `Output:Variable` can be requested at runtime from the API before the run starts (`request_variable`), meters likewise. An output may also read an actuator's current value or an EMS global.
- **Inputs**: actuators (`Schedule:Constant` / `Schedule Value`, thermostat setpoints, internal gain multipliers, ideal-loads setpoints, any `EnergyManagementSystem:Actuator`-class component) are resolved by handle after `api_data_fully_ready`. People/lights/equipment levels not exposed as actuators can be driven via a schedule actuator.
- **Spec format** (under the catalog entry's `user_defined.eplus_io`, overridable in the scenario's model `user_defined`):

```yaml
eplus_io:
  inputs:
    ZoneSetPoint:   {actuator: {component_type: "Zone Temperature Control", control_type: "Heating Setpoint", key: "ZONE1"}}
    LightsLevel:    {actuator: {component_type: "Schedule:Constant", control_type: "Schedule Value", key: "LIGHTS_SCH"}}
  outputs:
    TZone1:         {variable: {name: "Zone Mean Air Temperature", key: "ZONE1"}}
    HeatingEnergy:  {meter: "Heating:EnergyTransfer"}
```

  The key (`ZoneSetPoint`, ...) is the CosimGym name; it is what appears in the scenario `connections` and in the catalog inputs/outputs.
- **Failure mode**: a handle of -1 is a hard error that lists the available names (from the API data listing), never a silent zero.
- **Catalog contract**: catalog entries are static (RL spaces and `_instantiate` need them), so the shape is **one generic class, one catalog entry per building** (same pattern as `bui0_building_fmu`). The entry is *generated*, not hand-written:
  - new CLI `eplus_catalog_register.py` (modeled on `fmu_catalog_register.py`): given an IDF it resolves the version, does a short discovery run, writes a catalog entry draft with `max_sim_time` / `sim_start_date` / `time_step` from RunPeriod / Timestep (reusing `parse_idf_run_period`), and a **discovery listing** (available variables, meters, actuators) so the user chooses what to expose.
  - a pure-runtime (no catalog) mode is rejected: it breaks RL spaces and the catalog-driven state.
- Units, bounds and defaults for inputs/outputs are declared in the spec (`unit`, `min`, `max`, `default`) and copied into the catalog entry.
### 5b. Three levels of I/O (decided 2026-11-27)

Measured on `EC_IdealLoads.idf` / 24.1 (short discovery run, 58 s, mostly sizing): the API lists 6972 actuators (zone heating/cooling setpoints for 33 zones, 39 `Schedule:Compact` values, system-node setpoints, weather override, surface/material ones), 386 meters, 634 variables. A variable **not already requested by the IDF** (e.g. `Zone Mean Air Temperature`) is in the `.rdd` but not reachable by handle: it needs `request_variable` **before** the run. So the output list must be known in `initialize()`, and discovery must be done once at registration, not per run.

| Level | What | Who | When |
| --- | --- | --- | --- |
| 1. Native | variables, meters, actuators E+ already exposes | user names them in `eplus_io` | runtime, IDF untouched |
| 2. Automatic preparation | `request_variable` for unrequested outputs; strip `ExternalInterface*` (BUI0); helper `Schedule:Constant` for inputs that are not actuators (people / lights / equipment), declared as `drive: schedule` | the model, on a **copy** of the IDF in the cache, only from what the spec declares; original never modified | `initialize()` |
| 3. Manual | custom EMS / Python-plugin logic | user, in their own IDF | before loading |

Spec format (decided): explicit EnergyPlus names + a short CosimGym alias (as in the YAML above), with a pattern expansion (`key: "ZONE*"` -> one input per matching zone, aliases suffixed). Presets (shorthand over the explicit form) may be added later; not in scope now. The set of actuators / variables cannot change after start (handles resolved once); the **values** change every tick.

`eplus_catalog_register.py --idf X.idf` writes (a) `<idf>.eplus_io.yaml`: everything discoverable, grouped by kind and zone, surface/material actuators hidden behind a filter flag, for the user to copy lines from; (b) a catalog entry draft (`time_step`, `max_sim_time`, `sim_start_date` from Timestep / RunPeriod).

Validation I/O (my call, reviewable): BUI0 = the same 6 inputs / 2 outputs as `bui0_building_fmu` (allows a direct comparison with the FMU run); EC_IdealLoads = heating/cooling setpoints of 2-3 zones, their zone air temperature and ideal-loads supply rate.

## 6. Time, horizon and reset

- Horizon: `max_sim_time` and `sim_start_date` from the IDF RunPeriod (catalog); `BaseModel` then triggers a `horizon` reset itself.
- EnergyPlus cannot rewind: `reset(mode)` = stop worker, start a new one on a **copy** of the IDF with the RunPeriod rewritten to the position `horizon_pos_s()` / `span_ticks` that `BaseModel._reset` already computed (same logic as `BaseFMUModel._fmi_start_time` / `_fmi_stop_time`). Reuse one output dir per instance (as the FMU does) so hundreds of episodes do not leave hundreds of folders.
- Output dir: next to the scenario logs (reuse the `COSIM_FEDERATE_LOG_FILE` derivation of the FMU model).
- Open: whether `fmu_engine: energyplus` should also be set on these catalog entries so `ScenarioManager` enforces the whole-day `rolling_window` rule. With the API the day constraint probably does not apply, so likely **not** set; verify in phase 1 with a real run, and document the consequence for `rolling_window`.
- Leap years / multi-year: a multi-year RunPeriod must be rejected or normalized with a clear message (to be checked on `EC_IdealLoads`, 1.7 MB).

## 7. Lifecycle and cleanup (risk: `finalize()` is never called)

Because core never calls `finalize()`, the worker must clean itself up:
- worker watches the parent PID / a closed pipe and exits; `atexit` hook in the parent as a second line;
- worker is a child in its own process group so a killed federate leaves no orphan `energyplus` process;
- a hung worker is bounded by a per-step timeout (configurable, error names the IDF and the simulated time).
- `finalize()` is still implemented properly (idempotent) for when core is fixed.

## 8. Phases

| # | Phase | Output | Needs EnergyPlus |
| --- | --- | --- | --- |
| 0 | Environment: obtain an install runnable on this box (24.1 is the last with Ubuntu 20.04 support; confirm), check 26.1 requirements | `~/.cosimgym/energyplus/<ver>` | yes |
| 1 | Spike (throwaway): worker + callbacks on `BUI0.idf`, prove handshake, actuator set / variable read, timing | notes in changes report | yes |
| 2 | IDF utilities (pure Python, unit-testable without E+): version / Timestep / RunPeriod / zone parsing, RunPeriod rewrite on a copy, level-2 preparation (strip `ExternalInterface*`, helper schedules) | module + pytest | no |
| 3 | Registry + resolution + Transition upgrade cache | module + pytest (mock installs) | partly |
| 4 | Worker + IPC protocol + `EnergyPlusModel` (`initialize/step/finalize/reset`) | model class | yes |
| 5 | `eplus_catalog_register.py` + catalog entries for the six IDFs | catalog + CLI | yes |
| 6 | Scenarios in `src/scenarios/` and `tests/regression_suite.py` (single, multi-instance, parallel executor, RL rolling reset, horizon restart) | regression entries | yes |
| 7 | Docs: `models_reference.md`, `custom_models.md`, `developer_guide.md`, `distributed_deployment.md`, `KNOWN_ISSUES.md`; changes report `eplus_model_changes.md` | docs | no |

Validation matrix (generalizability): each of the six IDFs runs a short period with the San Francisco EPW; each exposes at least one input actuator and one output variable; plus one long run on one IDF for horizon restart. Versions that cannot run on this machine are marked as such in the matrix (not silently skipped).

## 9. Risks

1. No runnable EnergyPlus here yet; 26.1 probably not installable natively on glibc 2.31 -> 4 of 6 IDFs unrunnable until a 26.1-capable host or container exists.
2. Transition upgrade can change object semantics; upgraded IDFs must be validated, not assumed equal.
3. Callback/API names differ by version; the version-specific code is confined to the worker.
4. Throughput: one E+ process per instance, per-tick pipe round trip; measure in phase 1 before promising scaling numbers (the repo has a scaling study).
5. Orphan processes (section 7).
6. Single EPW: climate extremes not covered.

## 10. Open questions for the user

- Do the 4 `2_zones_*` IDFs already contain the actuators / schedules you want to drive, or should the plan include the `eplus_prepare` escape hatch from the start?
- Which inputs and outputs do you expect for each IDF (setpoints, internal gains, ideal-loads limits; zone temperatures, energy)? That fixes the validation matrix.
- ~~Container worker for 26.1~~: deferred, see `docs/future_and_TODOs/eplus_model_followups.md`.
- Still needed from the user: the `ran2026_rr_rack_workloads_energyplus_year.csv` file (only matters when 26.1 becomes runnable).

## 11. Phase 0 result (2026-11-27)

- Installed in `~/.cosimgym/energyplus/`: **23.1.0** and **24.1.0** (Ubuntu 20.04 builds). Both: `energyplus --version` OK, `pyenergyplus.api.EnergyPlusAPI` imports from the `cosim_gym` Python 3.12 env, `PreProcess/IDFVersionUpdater` present.
- **26.1 cannot run natively here**: the 26.x Linux builds are Ubuntu 24.04 only (26.2 adds AlmaLinux 9.6); 25.x are 22.04/24.04/CentOS 7. Host is glibc 2.31. Docker 28 works here, so a container worker stays the route for 26.1 (open question 3 of section 10, still unanswered).
- Standalone CLI smoke runs with the San Francisco EPW:
  - `EC_IdealLoads.idf` on 24.1: **completes** (129 s, full year).
  - `BUI0.idf` on 23.1: **fails**, it contains 9 `ExternalInterface*` objects (FMU export) and needs `socket.cfg`. The model must strip them on a copy of the IDF: the `eplus_prepare` copy step is therefore **required, not optional** (moves up to phase 2).
  - `2_zones_*` (4 files): reference `ran2026_rr_rack_workloads_energyplus_year.csv` (10 `Schedule:File`, rack workloads), which is **not in the repo**. Needed from the user. They also need 26.1.

## 12. Phase 1 result: spike (2026-11-27, throwaway code, not in the repo)

Worker subprocess (spawn) + `multiprocessing.Pipe`; callbacks `begin_zone_timestep_after_init_heat_balance` (wait for command, set actuators) and `end_zone_timestep_after_zone_reporting` (read, reply); active only when `api_data_fully_ready` and not `warmup_flag` and `kind_of_sim == 3` (sizing and warmup are skipped by that guard).

**Proven**
- Handshake works on both versions: BUI0 on 23.1 (6 inputs, 2 outputs) and EC_IdealLoads on 24.1 (zone heating setpoint actuator, zone air temperature as an *unrequested* variable via `request_variable`, ideal-loads heating rate). Actuation has the expected effect (BUI0 mean T 15.5 degC at setpoint 12 vs 23.9 at 24; EC T follows 14 vs 22 and the heating rate differs).
- Level-2 rules for BUI0 are exact and mechanical: `To:Schedule` -> `Schedule:Constant` (same name, limits, initial value) driven through actuator `Schedule:Constant / Schedule Value / <name>`; `From:Variable` -> `request_variable`; `ExternalInterface` -> deleted.
- IPC cost is negligible: round trip 0.2 ms/tick (BUI0, whole year in about 4 s), 2.8 ms/tick (EC, dominated by E+ itself).
- Unresolved handle (-1) detection works (reported before the first step).

**Findings that change the plan**
1. `Runtime.stop_simulation` exists in 24.1 but **not in 23.1**. Worker feature-detects it (`hasattr`) and otherwise exits the process (`os._exit`); a failure inside a callback is swallowed by E+ and the run silently continues to the end of the year, so the worker must wrap every callback in try/except and report errors through the pipe.
2. `current_sim_time` at the end-of-zone-timestep callback is **not a tick boundary** whenever the setpoint changes the HVAC state: steps of 450..1350 s on EC (900 s timestep) and 300..700 s on BUI0 at setpoint 24 (BUI0 at setpoint 20 = the schedule's initial value was exact, 600 s). It depends on HVAC sub-stepping. Never use it for time alignment. Instead the begin/end pairing was **verified directly**: `(day_of_year, hour, zone_time_step_number)` read in both callbacks is identical on every tick (192/192 on EC, 144/144 on BUI0; getters exist in 23.1 and 24.1), so the model keeps its own tick counter and can validate it against that clock.
3. **Start-up cost** (sizing + warmup): BUI0 0.5 s, EC_IdealLoads **41 s** (timestamped stdout, one run): about 24 s zone/HVAC sizing, about 7 s design-day simulations, about 10 s run-period warmup (49 warmup days). Every `reset` restarts the worker, so an RL rolling reset on EC pays all of it. Mitigation to be chosen in phase 4 from this split, each opt-in with a warning: skip sizing and design-day runs only if everything is hard-sized (autosized components need sizing); limit warmup days. Document the cost in the model docs.
4. Prepared IDF copies still get a per-instance path / content-hash cache with atomic writes as a design rule, but the earlier "anomalous first tick on BUI0" was **not** a file race: it reproduced solo at setpoint 24 and is the same sub-stepping effect as finding 2.
5. Run-period end: E+ returns normally; the worker reports `done` so the parent can raise a clear horizon error.

**Not done / not verified**: Python-plugin logic (level 3) is unverified: neither test IDF has `PythonPlugin` objects and the plugin runs inside E+'s embedded interpreter (not the 3.12 worker env); treat it as unsupported until tested; comparison of BUI0 API run against the `bui0_building_fmu` outputs (needs the FMU scenario run, phase 6); `drive: schedule` helper for inputs that are not actuators (only the BUI0 `To:Schedule` case was exercised); real-time behaviour with several worker instances in parallel.
