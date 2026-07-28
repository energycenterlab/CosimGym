"""
exchange_dummy.py

Self-contained, deliberately comms-heavy dummy model used to benchmark HELICS
data-exchange cost. It is the comms-isolating counterpart of
heavy_compute_dummy (which is compute-isolating and has NO inputs, so it
cannot receive data): exchange_dummy does near-zero compute but consumes
every subscribed input and publishes a configurable-width vector payload, so
a scaling study can measure HELICS data-exchange cost in isolation from CPU
cost.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-28

"""
import math

from ...base_model import BaseModel


class ExchangeDummy(BaseModel):
    """
    Dummy model that consumes every subscribed input value each step and
    publishes a configurable-width vector payload, scaled by the
    `msg_width` parameter. Optional negligible-by-default compute
    (`iterations`) and a publish-cadence knob (`publish_every`) let a
    scaling study isolate HELICS data-exchange cost from CPU cost.

    Parameters:
        - msg_width : length of the published payload vector [-]
        - iterations : optional busy-loop iterations per step [-]
        - publish_every : publish the payload only every k-th step [-]

    Outputs:
        - payload : published vector payload (list[float], length msg_width)
        - n_received : count of subscribed input values consumed on the last step [-]
    """

    MODEL_NAME = "exchange_dummy"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        self._acc = 0.0
        self._step_count = 0
        self._payload_template = None

    def initialize(self):
        """Seed the payload/received-count outputs."""
        msg_width = int(self.state.parameters.get("msg_width", 1))
        # Build the payload template ONCE. Rebuilding an msg_width-long list with
        # per-element float math every step would make the model's own CPU cost
        # scale with msg_width -- which is exactly the axis a data-exchange study
        # sweeps, so that compute would be mistaken for HELICS payload cost. Here
        # step() only copies the template and overwrites element 0, leaving
        # serialisation/transport as the only width-dependent term.
        self._payload_template = [float(i) for i in range(msg_width)]
        self.state.outputs["payload"] = [0.0] * msg_width
        self.state.outputs["n_received"] = 0.0
        self.init_state.outputs["payload"] = [0.0] * msg_width
        self.init_state.outputs["n_received"] = 0.0

    def step(self) -> None:
        """Consume every subscribed input, fold it into a persistent
        accumulator, and (on the configured cadence) publish a vector
        payload derived from that accumulator."""
        msg_width = int(self.state.parameters.get("msg_width", 1))
        iterations = int(self.state.parameters.get("iterations", 0))
        publish_every = int(self.state.parameters.get("publish_every", 1))

        acc = 0.0
        n_received = 0
        for value in self.state.inputs.values():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                acc += float(sum(value))
            else:
                acc += float(value)
            n_received += 1

        if iterations > 0:
            x = 0.123456789 + self.mod_num * 1e-3
            for i in range(iterations):
                x = math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
                acc += x

        self._acc += acc
        self._step_count += 1

        if self._step_count % publish_every == 0:
            if self._payload_template is None or len(self._payload_template) != msg_width:
                self._payload_template = [float(i) for i in range(msg_width)]
            # .copy() (not the template itself) because BaseFederate keeps the
            # published object in self.outputs and the storage sink may retain a
            # reference -- mutating a shared list in place would rewrite history.
            payload = self._payload_template.copy()
            payload[0] = self._acc
            self.state.outputs["payload"] = payload
        else:
            # Setting payload to None is load-bearing: BaseFederate._publish_outputs()
            # skips publication when the output value is None, which is how the
            # "publish every k-th tick" frequency knob actually reduces wire traffic.
            self.state.outputs["payload"] = None

        self.state.outputs["n_received"] = float(n_received)

    def finalize(self):
        self.logger.info(
            f"ExchangeDummy '{self.name}' finalized. "
            f"Final accumulator: {self._acc:.4f}, "
            f"steps: {self._step_count}"
        )
