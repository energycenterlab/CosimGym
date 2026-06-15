# CosimGym Models Reference

Three physical models available in the catalog. All follow the same integration pattern:
- **Parameters** → fixed at initialization from scenario YAML (`model_configs.parameters`)
- **Inputs** → received each step via HELICS subscriptions (scenario `subscribes`)
- **Outputs** → published each step via HELICS publications (scenario `publishes`)

---

## 1. `rc_building` — 5R1C ISO 13790 Building Thermal Model

Single-zone building using the 5-resistance 1-capacitance thermal network (ISO 13790 Annex C / Crank–Nicolson integration). Computes indoor temperatures and HVAC demand each step.

**Scenario snippet:**
```yaml
model_configs:
  instantiation:
    model_name: "rc_building"
  parameters:
    floor_area: 50.0
    t_set_heating: 21.0
    cop_heating: 3.0
```

### Parameters

| Parameter | Unit | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `floor_area` | m² | 35.0 | ✓ | Conditioned floor area |
| `window_area` | m² | 4.0 | | Total window area |
| `walls_area` | m² | 11.0 | | Total opaque external wall area |
| `room_vol` | m³ | 105.0 | | Air volume of zone |
| `total_internal_area` | m² | 142.0 | | Total internal surface area (A_t) |
| `u_walls` | W/(m²K) | 0.2 | | U-value of opaque walls |
| `u_windows` | W/(m²K) | 1.1 | | U-value of windows |
| `ach_vent` | 1/h | 1.5 | | Mechanical ventilation air changes per hour |
| `ach_infl` | 1/h | 0.5 | | Infiltration air changes per hour |
| `ventilation_efficiency` | – | 0.6 | | Heat recovery efficiency |
| `thermal_capacitance_per_floor_area` | J/(m²K) | 165000 | | Effective thermal capacitance per floor area (c_m) |
| `t_set_heating` | °C | 20.0 | | Heating setpoint fallback (used when not connected) |
| `t_set_cooling` | °C | 26.0 | | Cooling setpoint fallback (used when not connected) |
| `max_heating_power_per_floor_area` | W/m² | 500.0 | | Max deliverable heating power density |
| `max_cooling_power_per_floor_area` | W/m² | −500.0 | | Max deliverable cooling power density (negative) |
| `cop_heating` | – | 3.0 | | COP for heating thermal→electrical conversion |
| `cop_cooling` | – | 3.0 | | COP/EER for cooling thermal→electrical conversion |
| `default_solar_gains` | W | 0.0 | | Solar gains used when `solar_gains` input not connected |
| `default_internal_gains` | W | 0.0 | | Internal gains used when `internal_gains` input not connected |
| `t_m_initial` | °C | 20.0 | | Initial thermal mass temperature |

### Inputs (declare in `subscribes`)

| Key | Unit | Required | Description |
|-----|------|----------|-------------|
| `T_ext` | °C | ✓ | Outdoor air temperature |
| `solar_gains` | W | | Solar heat gains entering zone |
| `internal_gains` | W | | Internal heat gains (occupants, equipment) |
| `t_set_heating` | °C | | **Optional control input** — heating setpoint; overrides parameter if connected |
| `t_set_cooling` | °C | | **Optional control input** — cooling setpoint; overrides parameter if connected |

> `t_set_heating` and `t_set_cooling` are dual-mode: when subscribed they act as live dynamic inputs; when not subscribed they fall back to the parameter defaults.

### Outputs (declare in `publishes`)

| Key | Unit | Description |
|-----|------|-------------|
| `T_indoor` | °C | Indoor air temperature |
| `T_mass` | °C | Thermal mass node temperature |
| `T_surface` | °C | Internal surface node temperature |
| `T_operative` | °C | Operative comfort temperature (0.3 · T_air + 0.7 · T_s) |
| `Q_heating` | W | Heating power delivered (≥ 0) |
| `Q_cooling` | W | Cooling power extracted (≥ 0) |
| `energy_demand` | W | Signed demand (+ heating / − cooling) |
| `P_elec` | W | Electrical consumption = demand / COP |

---

## 2. `pandapower_grid` — Power Grid AC/DC Power Flow

Wraps [pandapower](https://pandapower.readthedocs.io). Loads any IEEE/MATPOWER case or custom JSON network, applies per-component setpoints each step, runs power flow or OPF, and publishes results. Topology and number of I/O are determined by the loaded case — keys are declared in the scenario YAML, not pre-fixed in the catalog.

**Key naming convention:**
```
inputs  →  {component}.{index}.{column}       e.g.  load.0.p_mw
outputs →  res.{component}.{index}.{column}   e.g.  res.bus.3.vm_pu
scalars →  convergence | total_loss_mw | total_gen_mw | total_load_mw
```

**Scenario snippet:**
```yaml
model_configs:
  instantiation:
    model_name: "pandapower_grid"
  parameters:
    case_file: "case14"
    case_file_format: "builtin"
    solver_mode: "pf"
subscribes:
  - key: "load.0.p_mw"
    type: "double"
    units: "MW"
    targets: {'0': [other_federate.0/signal_key]}
publishes:
  - key: "res.bus.0.vm_pu"
    type: "double"
    units: "p.u."
  - key: "convergence"
    type: "double"
    units: "-"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `case_file` | `"case14"` | Builtin case name (`case14`, `case118`, …) or path to network file |
| `case_file_format` | `"builtin"` | `builtin` \| `json` \| `pickle` \| `excel` \| `sqlite` \| `matpower` \| `pypower` |
| `solver_mode` | `"pf"` | `pf` (AC NR) \| `dc_pf` \| `opf` \| `dc_opf` |
| `pf_algorithm` | `"nr"` | `nr` \| `bfsw` \| `gs` \| `iwamoto_nr` — AC PF only |
| `pf_max_iteration` | `10` | Max Newton–Raphson iterations |
| `pf_tolerance_mva` | `1e-8` | Convergence tolerance [MVA] |
| `enforce_q_lims` | `false` | Enforce generator reactive power limits |
| `fail_on_divergence` | `true` | Raise exception on non-convergence; set `false` to publish `convergence=0` and continue |

### Inputs (settable per component)

Index `N` = 0-based row in the corresponding pandapower DataFrame.

| Component | Input key pattern | Unit | Description |
|-----------|------------------|------|-------------|
| Load | `load.N.p_mw` | MW | Active power demand |
| Load | `load.N.q_mvar` | Mvar | Reactive power demand |
| Load | `load.N.scaling` | – | Demand scaling factor |
| Load | `load.N.in_service` | bool | Connect/disconnect load |
| Static gen | `sgen.N.p_mw` | MW | Active power injection |
| Static gen | `sgen.N.q_mvar` | Mvar | Reactive power injection |
| Static gen | `sgen.N.scaling` | – | Output scaling factor |
| Sync gen | `gen.N.p_mw` | MW | Active power setpoint |
| Sync gen | `gen.N.vm_pu` | p.u. | Terminal voltage setpoint |
| Sync gen | `gen.N.in_service` | bool | Connect/disconnect generator |
| External grid | `ext_grid.N.vm_pu` | p.u. | Slack bus voltage magnitude |
| External grid | `ext_grid.N.va_degree` | ° | Slack bus voltage angle |
| Storage | `storage.N.p_mw` | MW | Active power (+ charge / − discharge) |
| Storage | `storage.N.soc_percent` | % | State of charge |
| Switch | `switch.N.closed` | bool | Switch open/close state |
| Transformer | `trafo.N.tap_pos` | – | Tap changer position |
| Bus | `bus.N.in_service` | bool | Connect/disconnect bus |

> Any column writable in the corresponding pandapower DataFrame can be used as an input key.

### Outputs (per result table)

| Output key pattern | Unit | Description |
|-------------------|------|-------------|
| `res.bus.N.vm_pu` | p.u. | Bus voltage magnitude |
| `res.bus.N.va_degree` | ° | Bus voltage angle |
| `res.bus.N.p_mw` | MW | Net active power injection at bus |
| `res.bus.N.q_mvar` | Mvar | Net reactive power injection at bus |
| `res.line.N.loading_percent` | % | Line thermal loading |
| `res.line.N.p_from_mw` | MW | Active power flow (from-bus end) |
| `res.line.N.p_to_mw` | MW | Active power flow (to-bus end) |
| `res.line.N.pl_mw` | MW | Active power loss in line |
| `res.line.N.i_from_ka` | kA | Current magnitude (from-bus end) |
| `res.trafo.N.loading_percent` | % | Transformer thermal loading |
| `res.trafo.N.p_hv_mw` | MW | Active power (HV side) |
| `res.gen.N.p_mw` | MW | Generator active power dispatch |
| `res.gen.N.q_mvar` | Mvar | Generator reactive power dispatch |
| `res.gen.N.vm_pu` | p.u. | Generator terminal voltage |
| `res.load.N.p_mw` | MW | Load active power consumed |
| `res.sgen.N.p_mw` | MW | Static gen active power |
| `res.ext_grid.N.p_mw` | MW | Slack bus active power exchange |
| `res.ext_grid.N.q_mvar` | Mvar | Slack bus reactive power exchange |
| `res.storage.N.p_mw` | MW | Storage power (+ charge / − discharge) |
| **`convergence`** | – | 1.0 = converged, 0.0 = diverged |
| **`total_loss_mw`** | MW | System-wide active power losses (lines + trafos) |
| **`total_gen_mw`** | MW | Total generation (gens + ext_grids) |
| **`total_load_mw`** | MW | Total active load in service |

---

## 3. `pandapipes_grid` — Fluid / Gas Network Pipeflow

Wraps [pandapipes](https://www.pandapipes.org). Loads any pandapipes network (built-in gas/heat cases or custom JSON), applies per-component setpoints, runs hydraulic or thermal pipeflow, and publishes results. Same dynamic I/O convention as pandapower_grid.

**Key naming convention:**
```
inputs  →  {component}.{index}.{column}           e.g.  sink.0.mdot_kg_per_s
outputs →  res.{component}.{index}.{column}       e.g.  res.junction.0.p_bar
scalars →  convergence | total_supply_mdot_kg_per_s | total_demand_mdot_kg_per_s
```

**Scenario snippet:**
```yaml
model_configs:
  instantiation:
    model_name: "pandapipes_grid"
  parameters:
    case_file: "gas_tcross1"
    case_file_format: "builtin"
    pf_mode: "hydraulics"
subscribes:
  - key: "sink.0.mdot_kg_per_s"
    type: "double"
    units: "kg/s"
    targets: {'0': [other_federate.0/signal_key]}
publishes:
  - key: "res.junction.0.p_bar"
    type: "double"
    units: "bar"
  - key: "convergence"
    type: "double"
    units: "-"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `case_file` | `"gas_tcross1"` | Builtin network name or path to network file |
| `case_file_format` | `"builtin"` | `builtin` \| `json` \| `pickle` |
| `pf_mode` | `"hydraulics"` | `hydraulics` \| `heat` \| `sequential` \| `bidirectional` |
| `max_iter_hyd` | `10` | Max hydraulic solver iterations |
| `max_iter_therm` | `10` | Max thermal solver iterations (heat/sequential modes) |
| `friction_model` | `"nikuradse"` | `nikuradse` \| `colebrook` — friction factor correlation |
| `fail_on_divergence` | `true` | Raise exception on non-convergence; set `false` to publish `convergence=0` and continue |

**Available builtin networks:**

| Name | Description |
|------|-------------|
| `gas_one_pipe1` | Single pipe, 2 junctions, 1 sink |
| `gas_one_pipe2` | Single pipe variant |
| `gas_strand_2pipes` | Two-pipe strand, 3 junctions |
| `gas_strand_pump` | Strand with pump |
| `gas_tcross1` | T-cross, 4 junctions, 3 pipes, 2 sinks |
| `gas_tcross2` | T-cross variant |
| `gas_meshed_square` | Meshed loop (4-node) |
| `gas_meshed_delta` | Meshed delta topology |
| `gas_meshed_pumps` | Meshed with pumps |
| `gas_meshed_two_valves` | Meshed with valves |
| `gas_2eg_hnet` | Two external grids, heat network |
| `gas_3parallel` | 3 parallel pipes |
| `gas_stanet_path` | Real STANET path network |
| `gas_versatility` | Multi-component test network |
| `heat_transfer_delta` | Heat transfer delta |
| `heat_transfer_delta_2sinks` | Heat transfer with 2 sinks |
| `heat_tranfer_modelica_path` | Modelica heat path |
| `heat_transfer_heights` | Network with elevation |

### Inputs (settable per component)

Index `N` = 0-based row in the component DataFrame.

| Component | Input key pattern | Unit | Description |
|-----------|------------------|------|-------------|
| Sink | `sink.N.mdot_kg_per_s` | kg/s | Mass flow demand |
| Sink | `sink.N.scaling` | – | Demand scaling factor |
| Sink | `sink.N.in_service` | bool | Connect/disconnect sink |
| Source | `source.N.mdot_kg_per_s` | kg/s | Mass flow injection |
| Source | `source.N.scaling` | – | Supply scaling factor |
| Source | `source.N.in_service` | bool | Connect/disconnect source |
| External grid | `ext_grid.N.p_bar` | bar | Supply pressure |
| External grid | `ext_grid.N.t_k` | K | Supply temperature |
| External grid | `ext_grid.N.in_service` | bool | Connect/disconnect ext_grid |
| Valve | `valve.N.opened` | bool | Open (true) / close (false) |
| Pipe | `pipe.N.in_service` | bool | Connect/disconnect pipe |
| Pipe | `pipe.N.u_w_per_m2k` | W/(m²K) | Heat transfer coefficient (heat mode) |
| Pipe | `pipe.N.qext_w` | W | External heat input/extraction |
| Pump | `pump.N.in_service` | bool | Enable/disable pump |
| Compressor | `compressor.N.pressure_ratio` | – | Compression ratio |
| Circ pump (pressure) | `circ_pump_pressure.N.p_flow_bar` | bar | Pressure setpoint |
| Circ pump (mass) | `circ_pump_mass.N.mdot_kg_per_s` | kg/s | Mass flow setpoint |
| Heat exchanger | `heat_exchanger.N.qext_w` | W | Heat exchange power |
| Pressure control | `press_control.N.controlled_p_bar` | bar | Pressure setpoint |
| Flow control | `flow_control.N.controlled_mdot_kg_per_s` | kg/s | Flow setpoint |

### Outputs (per result table)

| Output key pattern | Unit | Description |
|-------------------|------|-------------|
| `res.junction.N.p_bar` | bar | Junction pressure |
| `res.junction.N.t_k` | K | Junction temperature |
| `res.pipe.N.v_mean_m_per_s` | m/s | Mean flow velocity |
| `res.pipe.N.v_from_m_per_s` | m/s | Velocity at from-end |
| `res.pipe.N.v_to_m_per_s` | m/s | Velocity at to-end |
| `res.pipe.N.p_from_bar` | bar | Pressure at from-end |
| `res.pipe.N.p_to_bar` | bar | Pressure at to-end |
| `res.pipe.N.t_from_k` | K | Temperature at from-end |
| `res.pipe.N.t_to_k` | K | Temperature at to-end |
| `res.pipe.N.mdot_from_kg_per_s` | kg/s | Mass flow entering pipe |
| `res.pipe.N.mdot_to_kg_per_s` | kg/s | Mass flow leaving pipe |
| `res.pipe.N.vdot_norm_m3_per_s` | m³/s | Volumetric flow (normalised) |
| `res.pipe.N.dp_friction_loss_bar` | bar | Frictional pressure loss |
| `res.pipe.N.reynolds` | – | Reynolds number |
| `res.ext_grid.N.mdot_kg_per_s` | kg/s | Mass flow from ext_grid |
| `res.sink.N.mdot_kg_per_s` | kg/s | Mass flow consumed by sink |
| `res.source.N.mdot_kg_per_s` | kg/s | Mass flow from source |
| `res.valve.N.v_mean_m_per_s` | m/s | Mean velocity through valve |
| `res.pump.N.deltap_bar` | bar | Pump pressure rise |
| `res.compressor.N.deltap_bar` | bar | Compressor pressure rise |
| **`convergence`** | – | 1.0 = converged, 0.0 = diverged |
| **`total_supply_mdot_kg_per_s`** | kg/s | Total supply from all ext_grids |
| **`total_demand_mdot_kg_per_s`** | kg/s | Total demand from all sinks |

---

## Common Integration Notes

### Wiring pattern

The sub `key` name becomes the model input key. It does **not** need to match the publisher's key name — it maps via `targets`:

```yaml
subscribes:
  - key: "load.0.p_mw"          # name the model receives in state.inputs
    type: "double"
    units: "MW"
    targets:
      '0': [other_federate.0/signal_float]   # actual pub topic to subscribe to
```

### Only declare what you need

pandapower_grid and pandapipes_grid are **dynamic I/O models** — only keys listed in `subscribes`/`publishes` are active. Undeclared keys are ignored (no overhead, no errors). For exploration, declare a wide set; for production, trim to what's needed.

### Catalog reload

After any `catalog.yaml` edit:
```bash
python src/models/model_catalog/catalog_loader.py
```

### Smoke-test scenarios

| Model | Scenario file |
|-------|---------------|
| `rc_building` | `src/scenarios/rc_building_test_base.yaml` |
| `pandapower_grid` | `src/scenarios/pandapower_grid_test_base.yaml` |
| `pandapipes_grid` | `src/scenarios/pandapipes_grid_test_base.yaml` |
