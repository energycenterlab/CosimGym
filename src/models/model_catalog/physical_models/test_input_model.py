import math

from ...base_model import BaseModel


class TestInputModel(BaseModel):
    """Minimal signal generator for feeding deterministic test inputs to other federates."""

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        self._counter = 0

    def step(self) -> None:
        amplitude = self.state.parameters['amplitude']
        period = self.state.parameters['period']
        toggle_every = self.state.parameters['bool_toggle_every']
        int_max = self.state.parameters['int_max']

        self.state.outputs['signal_float'] = amplitude * math.sin(2 * math.pi * self.state.ts / period)
        self.state.outputs['signal_int'] = self._counter % (int_max + 1)
        self.state.outputs['signal_bool'] = (self.state.ts // toggle_every) % 2 == 0

        self._counter += 1

    def finalize(self):
        pass
