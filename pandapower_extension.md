
## 7. Enhancements & Additional Features

### 7.1 `output_all_res_tables` mode *(exploration helper)*

When `output_all_res_tables: true`, `initialize()` runs an initial PF snapshot, introspects all `res_*` DataFrames, and **auto-populates** `state.outputs` (and registers HELICS pubs) for every numeric column of every component. This removes the need to enumerate publishes in the scenario YAML. Useful during development / dashboarding. Implementation requires a hook into `BaseFederate._register_pubs()` or a pre-step that extends the pub/sub list before `_register_connections()` is called — needs careful integration.

### 7.2 Runtime topology changes *(add/remove components)*

The parameter `topology_updates_key` names a JSON-string input key. At each step, if this input carries a non-empty JSON payload the model parses and executes topology commands before the PF:

```json
[
  {"op": "create_load",  "bus": 5, "p_mw": 20.0, "q_mvar": 5.0, "name": "new_load"},
  {"op": "remove_sgen",  "index": 2},
  {"op": "toggle_switch","index": 0, "closed": false},
  {"op": "update_gen",   "index": 1, "column": "p_mw", "value": 80.0}
]
```

After a `create_*` command the new element gets index `len(net.<table>)-1`; after `remove_*` indices shift — the JSON interface must use **element names** (`name` column), not raw indices, to be robust across topology changes. An internal `_name_to_index` cache maps `{(table, name) → index}` and is rebuilt after every topology-change step.

This enables:
- Connecting/disconnecting feeders, DERs, or storage units at runtime.
- Modelling N-1 contingencies driven by a controller federate.
- Grid topology optimisation by an RL agent.

### 7.3 OPF cost functions via parameters

When `solver_mode: "opf"`, add parameters:
- `cost_type`: `"linear"` | `"polynomial"` (per pandapower `create_poly_cost` / `create_pwl_cost`)
- `gen_cost_coefficients`: list of `[cp0_eur, cp1_eur_per_mw, cp2_eur_per_mw2]` per generator
- Passed to `pp.create_poly_cost()` during `initialize()`.

### 7.4 DC approximation fast-path

When `solver_mode: "dc_pf"`, use `pp.rundcpp()` (10-100× faster). Outputs voltage angles and line MW flows (no Q, no losses). Useful for RL training where speed matters more than accuracy.

### 7.5 Multi-area / multi-federation grids

Multiple `pandapower_grid` federates each owning a sub-network (area), exchanging **tie-line flows** through HELICS pubs/subs. Requires defining boundary buses and an explicit inter-federate coupling pattern in the scenario YAML. Supports very large grids (>10k buses) by decomposition.

### 7.6 Timeseries warm-starting

At each step, the previous solution stored in `net.res_bus.vm_pu` / `va_degree` is automatically used as the initial guess for the next NR iteration (pandapower default). For rapidly varying grids this can be enhanced by:
- Saving `net._ppc` and restoring it as a warm start.
- Configuring `net.converged` to control fallback to flat-start.

### 7.7 Measurement/state-estimation integration

Add support for `pp.runse()` (state estimation) by introducing:
- Input category `measurement.{meas_type}.{index}.{value/std_dev}` — receives noisy meter readings.
- Parameter `solver_mode: "se"`.
- Outputs `res.bus.{n}.vm_pu_est` / `va_degree_est` from `net.res_bus_est`.

### 7.8 Diagnostics and observability outputs

Always-available outputs regardless of topology:
- `pf_iterations` — how many NR iterations were needed (proxy for grid stress).
- `max_bus_vm_deviation` — max |vm_pu - 1.0| across all buses.
- `max_line_loading_percent` — worst loaded line.
- `n_oos_elements` — count of out-of-service elements (indicator for contingency tracking).

### 7.9 Case file auto-detection

If `case_file_format` is omitted and `case_file` is provided, detect format from extension:
- `.json` → `"json"`, `.p` / `.pkl` → `"pickle"`, `.xlsx` → `"excel"`, `.m` → `"matpower"`.
- If no extension and no path separator → try `getattr(pp.networks, case_file)`.

### 7.10 Catalog registration of network metadata at startup

During `initialize()`, push a summary of the loaded network (number of buses, loads, gens, lines, nominal voltage levels) to Redis under a per-sim key. The dashboard can then display "what grid is simulated" without reading the case file.

---