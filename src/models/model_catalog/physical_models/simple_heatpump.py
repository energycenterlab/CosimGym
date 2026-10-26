"""
simple_heatpump.py

Variable COP air-source heat pump model for heating demand and electrical consumption.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from ...base_model import BaseModel


class SimpleHeatPump(BaseModel):
    """
    Simple air-source heat pump model based on a variable COP.

    The COP (Coefficient of Performance) is estimated from the temperature lift
    between the indoor supply temperature and the outdoor source, using a
    Carnot-fraction approach:

        COP = eta_carnot * T_supply / (T_supply - T_ext)   [temperatures in K]

    The COP is then clipped to a realistic range [COP_min, COP_max].

    Control input is the compressor modulation signal (0–1), representing the
    fraction of maximum electrical power consumed.

    Physics:
        P_elec  = modulation * P_rated          [W]  electrical consumption
        Q_heat  = P_elec * COP                  [W]  thermal output to building

    Inputs:
        - T_ext      : Outdoor (external) temperature [°C]
        - modulation : Compressor modulation signal  [-] (0 = off, 1 = full load)

    Outputs:
        - Q_heat  : Heating power delivered to the building [W]
        - P_elec  : Electrical power consumed               [W]
        - COP     : Current coefficient of performance      [-]

    Parameters:
        - P_rated      : Rated (maximum) electrical power input [W]
        - eta_carnot   : Carnot efficiency fraction (typical 0.4–0.6) [-]
        - T_supply     : Fixed supply water/air temperature [°C]
        - COP_min      : Minimum realistic COP (safety clip) [-]
        - COP_max      : Maximum realistic COP (safety clip) [-]
    """

    MODEL_NAME = "simple_heatpump"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Validate parameters."""
        if self.state.parameters["P_rated"] <= 0:
            raise ValueError("P_rated must be positive")
        if not (0 < self.state.parameters["eta_carnot"] <= 1):
            raise ValueError("eta_carnot must be in (0, 1]")

    def step(self) -> None:
        """Compute heating output and electrical consumption for one time step."""
        # Inputs
        T_ext = self.state.inputs.get("T_ext", 5.0)          # outdoor temp [°C]
        modulation = self.state.inputs.get("modulation", 0.0) # 0–1

        # Parameters
        P_rated = self.state.parameters["P_rated"]
        eta_carnot = self.state.parameters["eta_carnot"]
        T_supply_C = self.state.parameters["T_supply"]
        COP_min = self.state.parameters["COP_min"]
        COP_max = self.state.parameters["COP_max"]

        # Convert temperatures to Kelvin for Carnot formula
        T_supply_K = T_supply_C + 273.15
        T_ext_K = T_ext + 273.15

        # Avoid division by zero if T_supply == T_ext
        delta_T = T_supply_K - T_ext_K
        if delta_T <= 0:
            # No useful lift; heat pump cannot operate as a heater
            COP = COP_min
        else:
            COP = eta_carnot * T_supply_K / delta_T

        # Clip COP to realistic bounds
        COP = max(COP_min, min(COP_max, COP))

        # Clamp modulation to [0, 1]
        modulation = max(0.0, min(1.0, modulation))

        # Electrical and thermal power
        P_elec = modulation * P_rated
        Q_heat = P_elec * COP

        # Update outputs
        self.state.outputs["Q_heat"] = Q_heat
        self.state.outputs["P_elec"] = P_elec
        self.state.outputs["COP"] = COP

    def finalize(self):
        self.logger.info(
            f"HeatPump '{self.name}' finalized. "
            f"Last COP: {self.state.outputs.get('COP', 0):.2f}"
        )
