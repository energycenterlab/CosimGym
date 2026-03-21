"""
battery_dest.py

Battery model wrapper for integrating energy storage systems into the co-simulation.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import sys
sys.path.append('src/models/model_catalog/physical_models/')
from ...base_model import BaseModel
from resources.Battery_Dest import BESS


class battery_dest(BaseModel):
    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        
    def initialize(self):
        self._model = BESS(self.real_period,**self.state.parameters)
        # Keep internal battery SOC aligned with configured/reset state.
        initial_soc = self.state.outputs.get('SOC', self.state.parameters.get('SOC', self._model.SOC))
        self._model.setSOC(float(initial_soc))
        self.state.outputs['SOC'] = self._model.SOC
        self.state.outputs['P_net'] = float(self.state.outputs.get('P_net', 0.0))
        self.state.outputs['P_clipped'] = float(self.state.outputs.get('P_clipped', 0.0))

    def step(self):
        self._model.calculatepower(self.state.inputs['Battery_power'],self.state.inputs['P_load'],self.state.inputs['PV_power'], self.real_period)
        self.state.outputs['SOC'] = self._model.SOC
        self.state.outputs['Energy_out'] = self._model.Energy_out
        self.state.outputs['P_net'] = self._model.P_net
        self.state.outputs['P_clipped'] = self._model.P_clipped

    def reset(self, mode='full', ts=None, time=None):
        super().reset(mode=mode, ts=ts, time=time)
        if not hasattr(self, "_model") or self._model is None:
            return

        reset_soc = self.state.outputs.get('SOC', self.state.parameters.get('SOC', self._model.SOC))
        self._model.setSOC(float(reset_soc))
        self._model.P_net = float(self.state.outputs.get('P_net', 0.0))
        self._model.P_clipped = float(self.state.outputs.get('P_clipped', 0.0))
        self._model.Energy_out = 0.0
       
        
    def finalize(self):
        return super().finalize()
