from ...base_model import BaseModel
import time
import math
import random

class Inputs4Spring(BaseModel):
    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        pass

    def step(self) -> None:
        # Sinusoidal force: oscillates between -10 and +10 N (amplitude 10, centered at 0)
        
        # update state outputs MANDATORY STEP! 
        self.state.outputs['force'] = 10       
        self.state.outputs['disturbance'] = random.uniform(-self.init_state.outputs['disturbance'], self.init_state.outputs['disturbance'])

    def finalize(self):
        pass