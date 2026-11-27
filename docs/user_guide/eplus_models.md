# EnergyPlus Models (Python API)

`EnergyPlusModel` (`src/models/base_eplus_model.py`, catalog name `eplus_model`) co-simulates **any EnergyPlus IDF**
directly through `pyenergyplus`, with no FMU export step. There is **one** catalog entry: like the CSV reader, the model is
chosen in the scenario and **the resource (the IDF) decides the interface**. No registration step per building.

## Using it in a scenario

```yaml
building_federate:
  timing_configs: {real_period: 600}          # = IDF zone timestep, or a whole multiple of it
  connections:
    subscribes: [{key: Heat, ...}, {key: Cool, ...}]      # the names you declare in eplus_io.inputs
    publishes:  [{key: T, ...}]                           # ... and eplus_io.outputs
  model_configs:
    instantiation: {model_name: eplus_model, n_instances: 1, prefix: bui}
    init_state: {Heat: 20.0}                  # optional values from the first tick (else the input's `default`)
    user_defined:
      eplus:
        idf: src/models/.../BUI0.idf          # repo-relative, absolute or ~
        epw: src/models/.../weather.epw
        # version: "24.1"  eplus_dir: ~/ep    # optional: force an install
        # skip_sizing: false                  # opt-in; ignored if the IDF has autosize
        # burn_in_days: 0                     # see "Reset and restarts"
        # startup_timeout: 1800  step_timeout: 600
      eplus_io:
        inputs:
          Heat:  {actuator: {component_type: "Zone Temperature Control", control_type: "Heating Setpoint", key: "ZONE1"}, default: 20}
          Occ:   {schedule: "OCC_SCH"}        # shorthand, see below
          Lvl:   {global: "EMS_GLOBAL_NAME"}
        outputs:
          T:     {variable: {name: "Zone Mean Air Temperature", key: "ZONE1"}, agg: last}   # last|mean|sum|min|max
          E:     {meter: "Heating:EnergyTransfer"}
          Sp:    {actuator: {...}}            # read an actuator back; {global: NAME} reads an EMS global
        from_external_interface: false        # true: also take the I/O an FMU-export IDF declares (BUI0 does this)
```

Complete examples: `bui0_eplus_test`, `bui0_eplus_multi_test` (2 parallel instances), `eplus_horizon_smoketest`.

**Binding rules.** The names under `eplus_io` are the keys of `connections.subscribes` / `publishes`. A subscribe or publish key
with no `eplus_io` entry is an error (it lists what is declared). `eplus_io` entries no connection uses are silently dropped,
so you can declare a big family and connect only some, **except inputs with a `default`**: they are kept and driven at that
default even if nothing subscribes to them (an explicit default is an instruction; dropping it would leave the IDF's own value).
Omit `default` on an input you want left alone unless connected. A federate with no connections keeps everything.
The catalog entry declares no I/O; the model creates the state for these names itself (so BaseModel logs a harmless
"not defined in catalog, Not initialized" warning per name).

## What can be an input or an output (depends on the IDF)

EnergyPlus accepts values from outside only through **actuators** and **EMS globals**; it must expose them for the IDF you use.
- **Outputs**: any output variable or meter the IDF can report (`Zone Mean Air Temperature`, `Heating:EnergyTransfer`, ...). Variables are requested
  automatically; a meter is added to the IDF copy if missing. Also actuators and EMS globals can be read.
- **Inputs, no IDF change needed**: (a) `schedule: NAME`: converted to the actuator of that `Schedule:Constant/Compact/Year/File`
  (verified: set 0.25 / 0.75, read back identical), so any schedule of the IDF, e.g. a thermostat or occupancy schedule, becomes an input;
  (b) the actuators EnergyPlus registers by itself for objects in the IDF (zone thermostat setpoints, ideal-loads limits, surface/ construction
  controls, ...): `eplus_discover.py` lists them.
- **Inputs that need the IDF to be prepared**: an EMS global (`global:`) only exists if the IDF declares it
  (`EnergyManagementSystem:GlobalVariable` + the EMS program that uses it); an actuator for something EnergyPlus does not expose
  without EMS needs an `EnergyManagementSystem:Actuator` object. The model does **not** write EMS code for you.
  An IDF exported for the FMU (`ExternalInterface:*`) is already prepared: `from_external_interface: true` takes its I/O.
- If a name cannot be resolved, start-up fails **and the error is logged** with the actuators/variables whose names are close to it.

## Key patterns

`key` of an actuator or variable may contain `*` / `?`: it expands at start-up to every key the IDF offers for that
actuator/variable (case-insensitive; the IDF's zones are the fallback), each as `<name>_<key>` with non-alphanumerics replaced by `_` (two keys that collapse to the same name, or a name that clashes with another entry, are an error):
`T: {variable: {name: "Zone Mean Air Temperature", key: "TZ NWP*"}}` gives `T_TZ_NWP1OUT`, `T_TZ_NWP1IN`, ... Use those names in the connections.
The first expansion for an IDF runs EnergyPlus once up to its first timestep to list the interface (sizing + warmup, ~40 s on a large
building); the listing is cached in `~/.cosimgym/eplus_cache/listings` so later runs and other instances do not repeat it.
To see the expansion and everything exposed without running a scenario:

```bash
python src/models/model_catalog/eplus_discover.py --idf B.idf --epw W.epw [--io io.yaml]
```

## Time and horizon

- `real_period` must equal the IDF zone timestep (3600/`Timestep`) or a whole multiple N; with N > 1 the inputs are held N zone steps and each
  output is aggregated with its `agg`.
- `max_sim_time` and `sim_start_date` default to the IDF **RunPeriod** (EnergyPlus cannot run past it). A scenario may set a shorter
  window in `user_defined`; one beyond the RunPeriod is refused. Not needed in a catalog entry.

## How it works, reset and versions

- Each instance runs EnergyPlus in its **own worker subprocess** (`eplus_worker.py`, own process group, killed on finalize / parent death). Callbacks at
  the begin / end of every zone timestep are the rendezvous (inputs applied at the begin, outputs read at the end); every tick the model checks EnergyPlus's clock against its own.
- The IDF you give is **never modified**: each instance runs a prepared copy (`ExternalInterface*` converted or dropped, weather run forced, RunPeriod rewritten).
- **Reset and restarts.** EnergyPlus cannot rewind, so a horizon restart or an RL reset **restarts the worker** at the model's position: it starts at that day's
  midnight and free-runs to the target step with the spec `default` of each input (`init_state` only applies from the first tick; an input
without `default` keeps the IDF's own value during that stretch). `burn_in_days: N` starts N whole days earlier to recover the building's thermal history
  (BUI0 measured: error of the heating load at the same instant 23 % with 0 days, 12 % / 4 % / 0.6 % / 0.04 % with 1 / 3 / 7 / 14 days). Each restart pays the full start-up.
- **Versions.** The install is chosen per IDF (`eplus_registry.py`): the exact major.minor if present, otherwise the lowest newer install, upgrading the IDF with
  EnergyPlus's own `Transition` binaries (cached in `~/.cosimgym/eplus_cache`, override `COSIMGYM_EPLUS_CACHE`). Installs are searched in `COSIMGYM_ENERGYPLUS_DIRS`,
  `~/.cosimgym/energyplus/*`, `/usr/local|/opt/EnergyPlus-*`, `/Applications/EnergyPlus-*`; every host (distributed runs too) needs its own.
- Runtime errors (worker died, handle not found, clock desync, timeout) are logged by the model and raised.

Limits: [Known Issues](../KNOWN_ISSUES.md).
