# Plan: pandapower Power Grid Model Integration

**Target:** A `PandapowerGrid` model class that wraps [pandapower](https://pandapower.readthedocs.io) as a stepped, parametrizable HELICS federate following the CosimGym `BaseModel` integration standard.

**Session goal:** Write the implementation using this document as a blueprint; do NOT change any other existing file except the three artefacts listed in §6.

---

## 1. Problem Analysis

### Why pandapower is different from existing models

All existing models (SimpleBuilding, SimpleHeatPump, RCBuilding, …) have a **fixed, pre-declared** set of inputs and outputs. pandapower's topology — number of buses, loads, generators, lines — is entirely determined by the case file loaded at runtime. This creates a **dynamic I/O problem**:

- Number of inputs = Σ (settable columns × component counts) — unknown at design time.
- Number of outputs = Σ (result columns × component counts) — unknown at design time.

The framework resolves inputs/outputs from HELICS pub/sub keys (`input_output_names()`). We exploit this: the scenario YAML declares exactly which variables to subscribe/publish, and the model maps those flat HELICS keys onto pandapower DataFrames at each step. The catalog entry describes the **schema** (naming convention + available columns per component type), not a fixed list.

### HELICS key naming convention

Flat string keys using dot-separated notation:

```
{component_type}.{index}.{column}       ← inputs
res.{component_type}.{index}.{column}   ← outputs
```

Examples:
```
load.0.p_mw          load.2.q_mvar       sgen.1.p_mw
gen.0.p_mw           gen.0.vm_pu         ext_grid.0.vm_pu
res.bus.0.vm_pu      res.bus.3.va_degree
res.line.0.loading_percent              res.gen.1.q_mvar
res.load.2.p_mw      res.trafo.0.loading_percent
```

The model parses these keys at `initialize()` to build two dispatch tables:
- `_input_map`: `{helics_key → (component_table, row_index, column)}`
- `_output_map`: `{helics_key → (result_table, row_index, column)}`

---

## 2. Architecture

```
ScenarioManager
    │
    ├── pandapower_federate  (BaseFederate)
    │       │
    │       └── PandapowerGrid  (BaseModel)
    │               ├── net: pandapower.pandapowerNet     ← loaded once at initialize()
    │               ├── _input_map: dict                  ← built at initialize()
    │               ├── _output_map: dict                 ← built at initialize()
    │               │
    │               ├── step()
    │               │     1. apply state.inputs → net DataFrames
    │               │     2. runpp() / runopp()
    │               │     3. harvest net.res_* → state.outputs
    │               └── finalize()
    │
    └── [other federates: load_profiles, controllers, RL agents…]
```

---

## 3. Files to Create

| # | Path | Description |
|---|------|-------------|
| 1 | `src/models/model_catalog/physical_models/pandapower_grid.py` | Model class |
| 2 | `src/models/model_catalog/catalog.yaml` | New entry `pandapower_grid` appended |
| 3 | `src/scenarios/pandapower_grid_test_base.yaml` | Smoke-test scenario (IEEE case14) |

No other existing file should be modified.

---

## 4. Model Class Design (`pandapower_grid.py`)

### 4.1 Parameters (fixed before simulation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_file` | string | `""` | Path to pandapower JSON file; OR a built-in case name (e.g. `"case14"`, `"case118"`) |
| `case_file_format` | string | `"json"` | `"json"` \| `"pickle"` \| `"sqlite"` \| `"excel"` \| `"matpower"` \| `"pypower"` \| `"builtin"` |
| `solver_mode` | string | `"pf"` | `"pf"` (Newton-Raphson AC) \| `"dc_pf"` \| `"opf"` \| `"dc_opf"` |
| `pf_algorithm` | string | `"nr"` | `"nr"` \| `"bfsw"` \| `"gs"` \| `"iwamoto_nr"` — ignored when solver_mode is opf |
| `pf_max_iteration` | int | `10` | Maximum NR iterations |
| `pf_tolerance_mva` | float | `1e-8` | Convergence tolerance [MVA] |
| `enforce_q_lims` | bool | `false` | Enforce reactive power limits during PF |
| `fail_on_divergence` | bool | `true` | Raise exception vs log warning on divergence |
| `output_all_res_tables` | bool | `false` | When `true`: auto-publish ALL result columns for ALL components (no explicit publishes needed in scenario — useful for exploration) |
| `topology_updates_key` | string | `""` | If set, the model watches this special JSON-string input for runtime topology change commands (see §7.3) |

### 4.2 Inputs (dynamic — declared in scenario YAML subscribes)

Any key matching `{component_type}.{index}.{column}` where:
- **component_type** ∈ `{load, sgen, gen, ext_grid, shunt, storage, ward, xward, dcline, switch, trafo, line, bus}`
- **index**: integer row index in `net.<component_type>`
- **column**: any writable column of that DataFrame

Most useful settable columns per type:

| Component | Key columns (inputs) |
|-----------|---------------------|
| `load` | `p_mw`, `q_mvar`, `scaling`, `in_service` |
| `sgen` | `p_mw`, `q_mvar`, `scaling`, `in_service` |
| `gen` | `p_mw`, `vm_pu`, `in_service` |
| `ext_grid` | `vm_pu`, `va_degree` |
| `storage` | `p_mw`, `max_e_mwh`, `soc_percent` |
| `shunt` | `q_mvar`, `p_mw`, `in_service` |
| `switch` | `closed` |
| `bus` | `in_service` |
| `trafo` | `tap_pos`, `in_service` |
| `dcline` | `p_mw`, `in_service` |

Special: OPF-only settable columns (ignored in PF mode):
- `load`: `controllable`, `max_p_mw`, `min_p_mw`, `max_q_mvar`, `min_q_mvar`
- `sgen`/`gen`: `controllable`, `max_p_mw`, `min_p_mw`, `cost_per_mw`

### 4.3 Outputs (dynamic — declared in scenario YAML publishes)

Any key matching `res.{component_type}.{index}.{column}`:

| Result table | Key columns (outputs) |
|--------------|-----------------------|
| `res.bus` | `vm_pu`, `va_degree`, `p_mw`, `q_mvar` |
| `res.load` | `p_mw`, `q_mvar` |
| `res.sgen` | `p_mw`, `q_mvar` |
| `res.gen` | `p_mw`, `q_mvar`, `va_degree`, `vm_pu` |
| `res.ext_grid` | `p_mw`, `q_mvar` |
| `res.line` | `p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`, `pl_mw`, `ql_mvar`, `i_from_ka`, `i_to_ka`, `loading_percent` |
| `res.trafo` | `p_hv_mw`, `q_hv_mvar`, `p_lv_mw`, `q_lv_mvar`, `pl_mw`, `loading_percent`, `i_hv_ka`, `i_lv_ka` |
| `res.storage` | `p_mw`, `q_mvar`, `soc_percent` |
| `res.shunt` | `p_mw`, `q_mvar`, `vm_pu` |

Special scalar outputs (always published if declared):
- `convergence` — bool, `1.0` if PF converged else `0.0`
- `total_loss_mw` — system-wide active power loss
- `total_gen_mw` — total generation
- `total_load_mw` — total load

### 4.4 Class skeleton

```python
class PandapowerGrid(BaseModel):
    MODEL_NAME = "pandapower_grid"

    def initialize(self):
        # 1. load case (from file or builtin)
        self.net = self._load_case()
        # 2. build dispatch tables from declared I/O keys
        self._input_map  = self._build_input_map()   # {key: (table, idx, col)}
        self._output_map = self._build_output_map()  # {key: (res_table, idx, col)}
        # 3. seed initial outputs (pre-PF static values / zeros)
        self._seed_outputs()

    def step(self):
        self._apply_inputs()          # write state.inputs → net DataFrames
        self._run_solver()            # runpp() or runopp() with error handling
        self._harvest_outputs()       # read net.res_* → state.outputs

    def finalize(self):
        # log final network state / write optional result dump
        pass

    # --- private helpers ---

    def _load_case(self) -> pp.pandapowerNet:
        fmt  = self._param("case_file_format")
        path = self._param("case_file")
        if fmt == "builtin":
            return getattr(pp.networks, path)()          # e.g. case14()
        loaders = {
            "json":     pp.from_json,
            "pickle":   pp.from_pickle,
            "excel":    pp.from_excel,
            "sqlite":   pp.from_sqlite,
            "matpower": pp.converter.from_mpc,
            "pypower":  pp.converter.from_ppc,
        }
        return loaders[fmt](path)

    def _parse_key(self, key: str):
        """'load.2.p_mw'  → ('load', 2, 'p_mw')
           'res.bus.0.vm_pu' → ('res_bus', 0, 'vm_pu')"""
        parts = key.split(".")
        if parts[0] == "res":
            table = f"res_{parts[1]}"
            idx   = int(parts[2])
            col   = parts[3]
        else:
            table = parts[0]
            idx   = int(parts[1])
            col   = parts[2]
        return table, idx, col

    def _build_input_map(self):
        m = {}
        for key in self.state.inputs:
            try:
                m[key] = self._parse_key(key)
            except (IndexError, ValueError):
                self.logger.warning(f"Cannot parse input key '{key}' — skipped")
        return m

    def _build_output_map(self):
        m = {}
        for key in self.state.outputs:
            try:
                m[key] = self._parse_key(key)
            except (IndexError, ValueError):
                self.logger.warning(f"Cannot parse output key '{key}' — skipped")
        return m

    def _apply_inputs(self):
        for key, val in self.state.inputs.items():
            if val is None or key not in self._input_map:
                continue
            table, idx, col = self._input_map[key]
            if hasattr(self.net, table):
                self.net[table].at[idx, col] = val
            else:
                self.logger.warning(f"Table '{table}' not in network — key '{key}' skipped")

    def _run_solver(self):
        mode = self._param("solver_mode")
        try:
            if mode == "pf":
                pp.runpp(
                    self.net,
                    algorithm=self._param("pf_algorithm"),
                    max_iteration=self._param("pf_max_iteration"),
                    tolerance_mva=self._param("pf_tolerance_mva"),
                    enforce_q_lims=self._param("enforce_q_lims"),
                )
            elif mode == "dc_pf":
                pp.rundcpp(self.net)
            elif mode == "opf":
                pp.runopp(self.net)
            elif mode == "dc_opf":
                pp.rundcopp(self.net)
            self.state.outputs["convergence"] = 1.0
        except pp.powerflow.LoadflowNotConverged:
            self.logger.warning("Power flow did not converge at this time step")
            self.state.outputs["convergence"] = 0.0
            if self._param("fail_on_divergence"):
                raise

    def _harvest_outputs(self):
        for key, (table, idx, col) in self._output_map.items():
            if not hasattr(self.net, table):
                continue
            df = getattr(self.net, table)
            if df is not None and idx in df.index and col in df.columns:
                self.state.outputs[key] = float(df.at[idx, col])
        # scalar convenience outputs
        if "total_loss_mw" in self.state.outputs:
            self.state.outputs["total_loss_mw"] = float(
                self.net.res_line["pl_mw"].sum() + self.net.res_trafo["pl_mw"].sum()
            )
        if "total_gen_mw" in self.state.outputs and len(self.net.res_gen):
            self.state.outputs["total_gen_mw"] = float(self.net.res_gen["p_mw"].sum())
        if "total_load_mw" in self.state.outputs:
            self.state.outputs["total_load_mw"] = float(self.net.res_load["p_mw"].sum())
```

---

## 5. Catalog Entry (`catalog.yaml` — append)

```yaml
  pandapower_grid:
    class_name: PandapowerGrid
    module_path: models.model_catalog.physical_models.pandapower_grid
    version: 1.0.0
    description: >
      Parametrizable pandapower power grid model. Loads any pandapower-compatible
      case file (JSON, Matpower, pickle, builtin IEEE cases), applies per-component
      setpoints received from other federates, runs AC/DC power flow or optimal
      power flow, and publishes result variables (bus voltages, line loadings,
      power flows) as outputs. I/O keys use dot-notation:
      {component}.{index}.{column} for inputs,
      res.{component}.{index}.{column} for outputs.
    author: Pietro Rando Mazzarino
    domain: power_systems
    category: physical_model
    tags: [power_grid, pandapower, powerflow, opf, transmission, distribution]
    dependencies: [pandapower]
    time_step: 3600
    max_time_step: 86400
    min_time_step: 1
    user_defined:
      solver: newton-raphson
      integrator: quasi-static      # no ODE — each step is an independent PF snapshot
    parameters:
      case_file:
        type: string
        default_value: "case14"
        description: Path to pandapower case file OR builtin case name (e.g. "case14", "case118")
        unit: "-"
        required: true
      case_file_format:
        type: string
        default_value: "builtin"
        description: "builtin | json | pickle | excel | sqlite | matpower | pypower"
        unit: "-"
        required: true
      solver_mode:
        type: string
        default_value: "pf"
        description: "pf | dc_pf | opf | dc_opf"
        unit: "-"
        required: false
      pf_algorithm:
        type: string
        default_value: "nr"
        description: "nr | bfsw | gs | iwamoto_nr — PF mode only"
        unit: "-"
        required: false
      pf_max_iteration:
        type: int
        default_value: 10
        description: Max Newton-Raphson iterations
        unit: "-"
        required: false
      pf_tolerance_mva:
        type: float
        default_value: 1.0e-8
        description: Convergence tolerance [MVA]
        unit: "MVA"
        required: false
      enforce_q_lims:
        type: bool
        default_value: false
        description: Enforce generator Q limits during PF
        unit: "-"
        required: false
      fail_on_divergence:
        type: bool
        default_value: true
        description: Raise exception on non-convergence vs publish convergence=0 and continue
        unit: "-"
        required: false
      topology_updates_key:
        type: string
        default_value: ""
        description: Input key for runtime JSON topology-change commands (empty = disabled)
        unit: "-"
        required: false
    # inputs / outputs are DYNAMIC — their actual list is determined by what
    # the scenario YAML subscribes/publishes. The keys below are the SCHEMA
    # (naming convention examples), not an exhaustive enumeration.
    inputs:
      "load.N.p_mw":
        type: float
        default_value: ~
        description: "Active power setpoint for load N [MW]. Replace N with 0-based index."
        unit: "MW"
        required: false
        tags: [load, active_power]
      "load.N.q_mvar":
        type: float
        default_value: ~
        description: "Reactive power setpoint for load N [Mvar]."
        unit: "Mvar"
        required: false
        tags: [load, reactive_power]
      "sgen.N.p_mw":
        type: float
        default_value: ~
        description: "Active power output for static generator N [MW]."
        unit: "MW"
        required: false
        tags: [sgen, active_power]
      "gen.N.p_mw":
        type: float
        default_value: ~
        description: "Active power setpoint for synchronous generator N [MW]."
        unit: "MW"
        required: false
        tags: [gen, active_power]
      "gen.N.vm_pu":
        type: float
        default_value: ~
        description: "Voltage setpoint magnitude for gen N [p.u.]."
        unit: "p.u."
        required: false
        tags: [gen, voltage]
      "switch.N.closed":
        type: bool
        default_value: ~
        description: "Switch N state (true=closed)."
        unit: "-"
        required: false
        tags: [switch, topology]
    outputs:
      "res.bus.N.vm_pu":
        type: float
        default_value: 1.0
        description: "Voltage magnitude at bus N after power flow [p.u.]."
        unit: "p.u."
        required: false
        tags: [bus, voltage, state]
      "res.bus.N.va_degree":
        type: float
        default_value: 0.0
        description: "Voltage angle at bus N [degree]."
        unit: "degree"
        required: false
        tags: [bus, voltage, state]
      "res.line.N.loading_percent":
        type: float
        default_value: 0.0
        description: "Line N loading as percentage of thermal limit."
        unit: "%"
        required: false
        tags: [line, loading]
      "res.gen.N.q_mvar":
        type: float
        default_value: 0.0
        description: "Reactive power dispatch from gen N [Mvar]."
        unit: "Mvar"
        required: false
        tags: [gen, reactive_power]
      "convergence":
        type: float
        default_value: 0.0
        description: "1.0 if power flow converged at this step, 0.0 otherwise."
        unit: "-"
        required: false
        tags: [solver, status]
      "total_loss_mw":
        type: float
        default_value: 0.0
        description: "Total system active power losses [MW]."
        unit: "MW"
        required: false
        tags: [system, losses]
      "total_gen_mw":
        type: float
        default_value: 0.0
        description: "Total generation dispatched [MW]."
        unit: "MW"
        required: false
        tags: [system, generation]
      "total_load_mw":
        type: float
        default_value: 0.0
        description: "Total active load in service [MW]."
        unit: "MW"
        required: false
        tags: [system, load]
```

---

## 6. Smoke-Test Scenario (`pandapower_grid_test_base.yaml`)

Two-federate test: a `test_input_model` drives sinusoidal load scaling, the `pandapower_grid` runs IEEE case14 PF each step.

```yaml
version: "1.0.0"
name: "pandapower_grid_test_base"
scenario_description: >
  IEEE case14 power flow driven by a sinusoidal load-scaling signal.
  Observes bus voltages, line loadings, convergence, and system losses.

start_time: "2024-01-01T00:00:00"
end_time:   "2024-01-01T03:00:00"
log_level: ERROR

memory_config:
  batch_size: 1000
  attrs: ["res.bus.0.vm_pu", "res.bus.3.vm_pu", "res.line.0.loading_percent",
          "convergence", "total_loss_mw", "total_gen_mw", "total_load_mw",
          "load.0.p_mw"]

synchronization:
  auto_offset:
    enabled: true
    offset_step: 0.1
    override_existing_offsets: false
  default_subscription_causality: "same_step"
  validate_causality_cycles: true
  default_startup_sync:
    enabled: true
    force_read_all_subscriptions: true
    require_updated_inputs: false       # grid can run without all inputs at t=0
    missing_inputs_policy: "warn"

federations:
  federation_1:
    broker_config:
      core_type: "zmq"
      port: 23420
      host: "localhost"
      federates: 2
      log_level: ERROR

    federate_configs:

      # Drives load.0.p_mw with a sinusoidal profile
      load_driver_federate:
        name: "load_driver_federate"
        type: "base"
        log_level: ERROR
        core_name: "fed_load_driver"
        core_type: "zmq"
        timing_configs:
          real_period: 3600
        flags:
          terminate_on_error: true
          wait_for_current_time_update: false
        connections:
          endpoints: []
          subscribes: []
          publishes:
            - key: "load.0.p_mw"
              type: "double"
              units: "MW"
        model_configs:
          instantiation:
            model_name: "test_input_model"
            n_instances: 1
            prefix: "driver"
            parallel_execution: false
          parameters:
            amplitude: 50.0       # swing ±50 MW around the case14 default
            period: 6.0           # 6 steps / period
          init_state:
            signal_float: 0.0
          user_defined: {}

      # pandapower IEEE case14 grid
      grid_federate:
        name: "grid_federate"
        type: "base"
        log_level: ERROR
        core_name: "fed_grid"
        core_type: "zmq"
        timing_configs:
          real_period: 3600
          time_offset: 0.1
        flags:
          terminate_on_error: true
          wait_for_current_time_update: false
        connections:
          endpoints: []
          subscribes:
            - key: "load.0.p_mw"
              type: "double"
              units: "MW"
              targets:
                '0': [load_driver_federate.0/load.0.p_mw]
          publishes:
            - key: "res.bus.0.vm_pu"
              type: "double"
              units: "p.u."
            - key: "res.bus.3.vm_pu"
              type: "double"
              units: "p.u."
            - key: "res.line.0.loading_percent"
              type: "double"
              units: "%"
            - key: "convergence"
              type: "double"
              units: "-"
            - key: "total_loss_mw"
              type: "double"
              units: "MW"
            - key: "total_gen_mw"
              type: "double"
              units: "MW"
            - key: "total_load_mw"
              type: "double"
              units: "MW"
        model_configs:
          instantiation:
            model_name: "pandapower_grid"
            n_instances: 1
            prefix: "grid"
            parallel_execution: false
          parameters:
            case_file: "case14"
            case_file_format: "builtin"
            solver_mode: "pf"
            pf_algorithm: "nr"
            fail_on_divergence: false
          init_state: {}
          user_defined: {}
```

---


## 8. Implementation Checklist 

- [ ] Install pandapower in `cosim_gym` env: `conda install -c conda-forge pandapower` or `pip install pandapower`
- [ ] Write `pandapower_grid.py` following §4 skeleton (start with `_load_case`, `_parse_key`, `_apply_inputs`, `_run_solver`, `_harvest_outputs`)
- [ ] Add catalog entry to `catalog.yaml` (§5)
- [ ] Write smoke-test scenario (§6)
- [ ] Run catalog loader: `python src/models/model_catalog/catalog_loader.py`
- [ ] Run scenario: `python src/test_script.py` (add `main('pandapower_grid_test_base')`)
- [ ] Verify `results/pandapower_grid_test_base/*/federation_1/grid_federate_test_storage.json` contains non-trivial `res.bus.0.vm_pu` timeseries
- [ ] (Optional) Implement `topology_updates_key` handler (§7.2)
- [ ] (Optional) Implement OPF cost functions (§7.3)

---

## 9. Key Design Decisions & Risks

| Decision | Rationale | Risk / Mitigation |
|----------|-----------|-------------------|
| Flat dot-notation keys | Compatible with HELICS key naming, human-readable | Long keys in large grids — consider aliasing; document in catalog |
| `_input_map` built at `initialize()` | Avoids string parsing overhead at each step | If topology changes at runtime, map must be rebuilt — see §7.2 |
| `state.inputs.get(key)` fallback | Keys not subscribed stay `None`; model uses case-file defaults (no overwrite in `_apply_inputs`) | Federate must subscribe only keys it intends to control; dangling pubs are harmless |
| Quasi-static model | Each step is an independent PF snapshot — no ODE state except optionally `soc_percent` for storage | Storage SOC evolution needs a companion storage model or can be managed within this model with a `_update_storage_soc()` method |
| `fail_on_divergence: false` in smoke test | Lets simulation continue with stale outputs if PF diverges (driven load may exceed limits) | Log a WARNING and publish `convergence=0` — downstream models must guard against stale values |
| pandapower not in current conda env | Must be installed separately — add to requirements or docker-compose | Pin version; pandapower releases frequently and API can shift |
