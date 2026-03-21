from ...base_model import BaseModel

class rb_bems(BaseModel):
    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        
    def initialize(self):
        self.P_load = None
        self.P_gen = None
        self.SOC = None
        

    def step(self):
        self.P_load = self.state.inputs['P_load']
        self.P_gen = self.state.inputs['P_gen']
        self.SOC = self.state.inputs['SOC']

        if self.P_load >= self.P_gen:
            # discharging battery
            self.state.outputs['Battery_power'] = self.P_load - self.P_gen  
        elif self.P_load < self.P_gen:
            # charging battery
            self.state.outputs['Battery_power'] = self.P_load - self.P_gen  

        # elif self.SOC < self.state.parameters['SOC_min']:
        #     self.outputs['Battery_power'] = max(self.P_load - self.P_gen, 0)
        # elif self.SOC > self.state.parameters['SOC_max']:
        #     self.outputs['Battery_power'] = min(self.P_load - self.P_gen, 0)
        else:
            self.logger.warning(f"Unexpected state: P_load={self.P_load}, P_gen={self.P_gen}, SOC={self.SOC}")

        
    def finalize(self):
        return super().finalize()