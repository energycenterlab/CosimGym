import numpy as np
from model_Dest.PV_Dest import PV_model
from _baseModels.Model import Model #rember the top level running python script is al;ways main
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

class PV(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.initialization()
    def initialization(self):
        self.model = PV_model(**self.params)

    def step(self, ts):
        ts-=1
        self.outputs['Power_PV'] = self.model.step(ts, self.inputs['G_H_R'], self.inputs['D_H_R'], self.inputs['Ambient_temperature'])
        return super().step(ts)

    def finalize(self):
        return super().finalize()
        pass