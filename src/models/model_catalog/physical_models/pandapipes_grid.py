"""
pandapipes_grid.py

Stepped pandapipes fluid/gas network model for CosimGym co-simulation.

I/O keys follow dot-notation:
  inputs  : {component}.{index}.{column}       e.g.  sink.0.mdot_kg_per_s
  outputs : res.{component}.{index}.{column}   e.g.  res.junction.0.p_bar
  scalars : convergence | total_supply_mdot_kg_per_s | total_demand_mdot_kg_per_s

Author: Pietro Rando Mazzarino
"""

import pandapipes as ppipe
import pandapipes.networks as ppnet

from ...base_model import BaseModel

_SCALAR_OUTPUTS = frozenset({
    "convergence",
    "total_supply_mdot_kg_per_s",
    "total_demand_mdot_kg_per_s",
})


class PandapipesGrid(BaseModel):
    MODEL_NAME = "pandapipes_grid"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def initialize(self):
        self.net = self._load_case()
        self._input_map = self._build_input_map()
        self._output_map = self._build_output_map()
        # Pre-register input keys so storage buffer tracks them
        for key in self._input_map:
            if key not in self.state.inputs:
                self.state.inputs[key] = None
        self._seed_outputs()

    def step(self):
        self._apply_inputs()
        self._run_solver()
        self._harvest_outputs()

    def finalize(self):
        self.logger.info(f"PandapipesGrid '{self.name}' finalised.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _param(self, key):
        return self.state.parameters[key]

    def _load_case(self):
        fmt  = self._param("case_file_format")
        path = self._param("case_file")
        if fmt == "builtin":
            builder = getattr(ppnet, path, None)
            if builder is None or not callable(builder):
                raise ValueError(f"Built-in network '{path}' not found in pandapipes.networks")
            return builder()
        loaders = {
            "json":   ppipe.from_json,
            "pickle": ppipe.from_pickle,
        }
        if fmt not in loaders:
            raise ValueError(f"Unknown case_file_format '{fmt}'")
        return loaders[fmt](path)

    def _parse_key(self, key: str):
        """
        'sink.0.mdot_kg_per_s'     → ('sink',         0, 'mdot_kg_per_s')
        'res.junction.0.p_bar'     → ('res_junction',  0, 'p_bar')
        'res.pipe.0.v_mean_m_per_s'→ ('res_pipe',      0, 'v_mean_m_per_s')
        """
        parts = key.split(".")
        if parts[0] == "res":
            if len(parts) < 4:
                raise ValueError(f"res key too short: '{key}'")
            return f"res_{parts[1]}", int(parts[2]), ".".join(parts[3:])
        if len(parts) < 3:
            raise ValueError(f"input key too short: '{key}'")
        return parts[0], int(parts[1]), ".".join(parts[2:])

    def _build_input_map(self):
        m = {}
        for key in (self.config.inputs or []):
            if key in _SCALAR_OUTPUTS:
                continue
            try:
                m[key] = self._parse_key(key)
            except (IndexError, ValueError) as exc:
                self.logger.warning(f"Cannot parse input key '{key}': {exc} — skipped")
        return m

    def _build_output_map(self):
        m = {}
        for key in (self.config.outputs or []):
            if key in _SCALAR_OUTPUTS:
                continue
            try:
                m[key] = self._parse_key(key)
            except (IndexError, ValueError) as exc:
                self.logger.warning(f"Cannot parse output key '{key}': {exc} — skipped")
        return m

    def _seed_outputs(self):
        for key in (self.config.outputs or []):
            if key not in self.state.outputs:
                self.state.outputs[key] = 0.0

    def _apply_inputs(self):
        for key, val in self.state.inputs.items():
            if val is None or key not in self._input_map:
                continue
            table, idx, col = self._input_map[key]
            tbl = getattr(self.net, table, None)
            if tbl is None:
                self.logger.warning(f"Table '{table}' not in network — key '{key}' skipped")
                continue
            if idx not in tbl.index:
                self.logger.warning(f"Index {idx} not in net.{table} — key '{key}' skipped")
                continue
            if col not in tbl.columns:
                self.logger.warning(f"Column '{col}' not in net.{table} — key '{key}' skipped")
                continue
            tbl.at[idx, col] = val

    def _run_solver(self):
        mode = self._param("pf_mode")
        try:
            ppipe.pipeflow(
                self.net,
                mode=mode,
                max_iter_hyd=int(self._param("max_iter_hyd")),
                max_iter_therm=int(self._param("max_iter_therm")),
                friction_model=self._param("friction_model"),
            )
            self.state.outputs["convergence"] = 1.0
        except ppipe.PipeflowNotConverged as exc:
            self.logger.warning(f"Pipeflow did not converge: {exc}")
            self.state.outputs["convergence"] = 0.0
            if self._param("fail_on_divergence"):
                raise

    def _harvest_outputs(self):
        for key, (table, idx, col) in self._output_map.items():
            df = getattr(self.net, table, None)
            if df is None:
                continue
            if idx in df.index and col in df.columns:
                self.state.outputs[key] = float(df.at[idx, col])

        # scalar system-level outputs
        if "total_supply_mdot_kg_per_s" in self.state.outputs:
            supply = 0.0
            if self.net.res_ext_grid is not None and len(self.net.res_ext_grid):
                supply = float(abs(self.net.res_ext_grid["mdot_kg_per_s"].sum()))
            self.state.outputs["total_supply_mdot_kg_per_s"] = supply

        if "total_demand_mdot_kg_per_s" in self.state.outputs:
            demand = 0.0
            if self.net.res_sink is not None and len(self.net.res_sink):
                demand = float(self.net.res_sink["mdot_kg_per_s"].sum())
            self.state.outputs["total_demand_mdot_kg_per_s"] = demand
