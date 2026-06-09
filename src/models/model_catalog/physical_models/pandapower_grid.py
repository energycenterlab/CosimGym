"""
pandapower_grid.py

Stepped pandapower power grid model for CosimGym co-simulation.

I/O keys follow dot-notation:
  inputs  : {component}.{index}.{column}       e.g.  load.0.p_mw
  outputs : res.{component}.{index}.{column}   e.g.  res.bus.3.vm_pu
  scalars : convergence | total_loss_mw | total_gen_mw | total_load_mw

Author: Pietro Rando Mazzarino
"""

import pandapower as pp
import pandapower.networks as ppn

from ...base_model import BaseModel

_SCALAR_OUTPUTS = frozenset({"convergence", "total_loss_mw", "total_gen_mw", "total_load_mw"})


class PandapowerGrid(BaseModel):
    MODEL_NAME = "pandapower_grid"

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
        self.logger.info(f"PandapowerGrid '{self.name}' finalised.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _param(self, key):
        return self.state.parameters[key]

    def _load_case(self):
        fmt  = self._param("case_file_format")
        path = self._param("case_file")
        if fmt == "builtin":
            builder = getattr(ppn, path, None)
            if builder is None or not callable(builder):
                raise ValueError(f"Built-in case '{path}' not found in pandapower.networks")
            return builder()
        loaders = {
            "json":     pp.from_json,
            "pickle":   pp.from_pickle,
            "excel":    pp.from_excel,
            "sqlite":   pp.from_sqlite,
            "matpower": pp.converter.from_mpc,
            "pypower":  pp.converter.from_ppc,
        }
        if fmt not in loaders:
            raise ValueError(f"Unknown case_file_format '{fmt}'")
        return loaders[fmt](path)

    def _parse_key(self, key: str):
        """
        'load.2.p_mw'       → ('load',    2, 'p_mw')
        'res.bus.0.vm_pu'   → ('res_bus', 0, 'vm_pu')
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
        mode = self._param("solver_mode")
        try:
            if mode == "pf":
                pp.runpp(
                    self.net,
                    algorithm=self._param("pf_algorithm"),
                    max_iteration=int(self._param("pf_max_iteration")),
                    tolerance_mva=float(self._param("pf_tolerance_mva")),
                    enforce_q_lims=bool(self._param("enforce_q_lims")),
                    numba=False,
                )
            elif mode == "dc_pf":
                pp.rundcpp(self.net, numba=False)
            elif mode == "opf":
                pp.runopp(self.net)
            elif mode == "dc_opf":
                pp.rundcopp(self.net)
            else:
                raise ValueError(f"Unknown solver_mode '{mode}'")
            self.state.outputs["convergence"] = 1.0
        except (pp.LoadflowNotConverged, pp.OPFNotConverged) as exc:
            self.logger.warning(f"Solver did not converge: {exc}")
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
        if "total_loss_mw" in self.state.outputs:
            loss = 0.0
            if self.net.res_line is not None and "pl_mw" in self.net.res_line.columns:
                loss += float(self.net.res_line["pl_mw"].sum())
            if self.net.res_trafo is not None and "pl_mw" in self.net.res_trafo.columns:
                loss += float(self.net.res_trafo["pl_mw"].sum())
            self.state.outputs["total_loss_mw"] = loss

        if "total_gen_mw" in self.state.outputs:
            gen_total = 0.0
            if self.net.res_gen is not None and len(self.net.res_gen) and "p_mw" in self.net.res_gen.columns:
                gen_total += float(self.net.res_gen["p_mw"].sum())
            if self.net.res_ext_grid is not None and len(self.net.res_ext_grid) and "p_mw" in self.net.res_ext_grid.columns:
                gen_total += float(self.net.res_ext_grid["p_mw"].sum())
            self.state.outputs["total_gen_mw"] = gen_total

        if "total_load_mw" in self.state.outputs:
            if self.net.res_load is not None and len(self.net.res_load) and "p_mw" in self.net.res_load.columns:
                self.state.outputs["total_load_mw"] = float(self.net.res_load["p_mw"].sum())
