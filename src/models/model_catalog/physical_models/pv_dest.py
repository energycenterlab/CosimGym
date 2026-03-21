"""
pv_dest.py

Photovoltaic (PV) system model for generating solar power outputs based on environmental inputs.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import sys
sys.path.append('src/models/model_catalog/physical_models/')
from ...base_model import BaseModel
from resources.PV_dest_model.PV_Dest import PV_model



class PV_dest(BaseModel):
    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        
    def initialize(self):
        self._model = PV_model(**self.state.parameters)
        self.logger.debug(f"PV_dest model initialized with parameters: {self._model.__dict__}")

    def step(self):
        hour_day = self.state.ts % 60
        self.state.outputs['PV_power'] = self._model.step(hour_day, self.state.inputs['GHI'], self.state.inputs['DHI'], self.state.inputs['T_ext'])
        
    def finalize(self):
        return super().finalize()