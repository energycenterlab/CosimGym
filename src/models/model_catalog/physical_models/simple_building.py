"""
simple_building.py

Simplified thermal building model using 1R1C representation for energy demand simulations.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from ...base_model import BaseModel


class SimpleBuilding(BaseModel):
    """
    Simple 1R1C (single thermal resistance/capacitance) building model.

    Models the indoor air temperature based on:
      - Heat exchange with the outdoor environment through the building envelope
      - A heating load injected into the zone (e.g. from a heat pump)

    Thermal equation (Euler integration):
        C * dT_in/dt = Q_heat - (T_in - T_ext) / R

    Inputs:
        - T_ext   : Outdoor (external) temperature [°C]
        - Q_heat  : Heating load supplied to the building [W]

    Outputs:
        - T_indoor : Indoor air temperature [°C]

    Parameters:
        - thermal_capacitance : Effective thermal mass of the building [J/K]
        - thermal_resistance  : Thermal resistance of the envelope [K/W]
        - T_initial           : Initial indoor temperature [°C]
    """

    MODEL_NAME = "simple_building"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Validate parameters and set initial indoor temperature."""
        C = self.state.parameters["thermal_capacitance"]
        R = self.state.parameters["thermal_resistance"]
        if C <= 0:
            raise ValueError("thermal_capacitance must be positive")
        if R <= 0:
            raise ValueError("thermal_resistance must be positive")

        # Seed the indoor temperature output from the parameter
        T0 = self.state.parameters["T_initial"]
        self.state.outputs["T_indoor"] = T0
        self.init_state.outputs["T_indoor"] = T0

    def step(self) -> None:
        """Advance indoor temperature by one simulation time step (Euler)."""
        # Retrieve current state
        T_in = self.state.outputs["T_indoor"]          # current indoor temp [°C]
        T_ext = self.state.inputs.get("T_ext", 10.0)  # outdoor temp [°C]
        Q_heat = self.state.inputs.get("Q_heat", 0.0) # heating load [W]

        # Building thermal parameters
        C = self.state.parameters["thermal_capacitance"]  # [J/K]
        R = self.state.parameters["thermal_resistance"]   # [K/W]

        # Use real_period as the integration time step [s]
        dt = self.real_period

        # Heat balance: supplied heat minus envelope losses
        dT = (Q_heat - (T_in - T_ext) / R) / C

        # Euler integration
        T_new = T_in + dT * dt

        self.state.outputs["T_indoor"] = T_new

    def finalize(self):
        self.logger.info(
            f"Building '{self.name}' finalized. "
            f"Final indoor temperature: {self.state.outputs['T_indoor']:.2f} °C"
        )
