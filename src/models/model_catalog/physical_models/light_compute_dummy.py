"""
heavy_compute_dummy.py

Self-contained, deliberately CPU-heavy dummy model used to benchmark parallel
(worker-process) vs sequential execution of a federate's model instances.

Pure-Python float arithmetic in a busy loop is intentional: it is GIL-bound,
so it demonstrates why threads give no speedup for this kind of workload and
persistent worker *processes* (see core/parallel_executor.py) are needed
instead.

No required external inputs — the model is self-contained so it can be used
in a minimal benchmark scenario with no weather/grid federates.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import math

from ...base_model import BaseModel


class LightComputeDummy(BaseModel):
    """
    Dummy model whose step() burns CPU doing real (non-optimizable-away)
    floating point work, scaled by the `iterations` parameter.

    Parameters:
        - iterations : number of busy-loop iterations per step [-]

    Outputs:
        - result : running accumulator value (float, no physical meaning)
        - step_count : number of steps executed so far [-]
    """

    MODEL_NAME = "light_compute_dummy"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Seed the accumulator/counter outputs."""
        self.state.outputs["result"] = 0.0
        self.state.outputs["step_count"] = 0
        self.init_state.outputs["result"] = 0.0
        self.init_state.outputs["step_count"] = 0

    def step(self) -> None:
        """doing nothing just fast forwarding data to the next step, accessing input and writing outputs dictionsries this used in the vertical scaling test"""
        iterations = int(self.state.parameters.get("iterations", 100000))
        acc = self.state.outputs.get("result", 0.0)



        self.state.outputs["result"] = acc
        self.state.outputs["step_count"] = self.state.outputs.get("step_count", 0) + 1

    def finalize(self):
        self.logger.info(
            f"LightComputeDummy '{self.name}' finalized. "
            f"Final result: {self.state.outputs['result']:.4f}, "
            f"steps: {self.state.outputs['step_count']}"
        )
