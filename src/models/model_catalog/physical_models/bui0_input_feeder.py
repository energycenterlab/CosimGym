from ...base_model import BaseModel


class BUI0InputFeeder(BaseModel):
    """Time-varying input feeder for the BUI0 EnergyPlus FMU.

    Drives the six FMU input schedules (occupancy, lighting, equipment, other
    equipment radiant/fan-coil and the zone temperature set-point) from the
    hour of day held in ``self.state.time``. Profiles are deliberately simple
    (occupied vs. unoccupied square waves with a night set-back) so the
    co-simulation has deterministic, physically plausible boundary conditions.
    """

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        pass

    def step(self) -> None:
        p = self.state.parameters
        hour = self.state.time.hour + self.state.time.minute / 60.0

        occupied = p['occupied_start_hour'] <= hour < p['occupied_end_hour']

        if occupied:
            self.state.outputs['PeopleNumber'] = p['people_occupied']
            self.state.outputs['LightsWatt'] = p['lights_peak_w']
            self.state.outputs['EEquipWatt'] = p['eequip_peak_w']
            self.state.outputs['OthEquRadWatt'] = p['otheq_rad_peak_w']
            self.state.outputs['OthEquFCWatt'] = p['otheq_fc_peak_w']
            self.state.outputs['ZoneSetPoint'] = p['setpoint_day_c']
        else:
            self.state.outputs['PeopleNumber'] = 0.0
            self.state.outputs['LightsWatt'] = p['lights_base_w']
            self.state.outputs['EEquipWatt'] = p['eequip_base_w']
            self.state.outputs['OthEquRadWatt'] = p['otheq_rad_base_w']
            self.state.outputs['OthEquFCWatt'] = p['otheq_fc_base_w']
            self.state.outputs['ZoneSetPoint'] = p['setpoint_night_c']

    def finalize(self):
        pass
