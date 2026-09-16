from ...base_model import BaseModel


class BUI0InputFeeder(BaseModel):
    """Time-varying input feeder for the BUI0 EnergyPlus FMU.

    Drives the six FMU input schedules (occupancy, lighting, equipment, other
    equipment radiant/fan-coil and the zone temperature set-point) from the
    hour of day and the month held in ``self.state.time``. Profiles are
    deliberately simple (occupied vs. unoccupied square waves with a night
    set-back) so the co-simulation has deterministic, physically plausible
    boundary conditions.

    The set-point schedule is season-aware. The heating season spans
    ``heating_season_start_month`` .. ``heating_season_end_month`` (inclusive,
    wrapping across new year, e.g. 10..4 = October to April); every other month
    is the cooling season. What is published during the cooling season depends
    on ``cooling_season_mode``:

    - ``setback`` (default): publish ``cooling_season_setback_c``. Correct for
      heating-only zones such as BUI0, whose thermostat is a
      ``ThermostatSetpoint:SingleHeating`` with zero cooling capacity — a
      summer set-point of 26 degC would make the heating coil chase 26 degC.
    - ``cooling``: publish ``cooling_setpoint_day_c`` / ``cooling_setpoint_night_c``.
      For zones whose thermostat actually has a cooling set-point.

    ``setpoint_day_c`` / ``setpoint_night_c`` are kept as deprecated aliases of
    the heating-season set-points: when set in a scenario they win over
    ``heating_setpoint_day_c`` / ``heating_setpoint_night_c``.

    Besides ``ZoneSetPoint`` the feeder publishes ``HeatingSeason`` (1.0 during
    the heating season, 0.0 otherwise) so downstream models, reward functions
    and dashboards can condition on the season without re-deriving it.
    """

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        pass

    def _is_heating_season(self, month: int) -> bool:
        p = self.state.parameters
        start = int(p['heating_season_start_month'])
        end = int(p['heating_season_end_month'])
        if start <= end:
            return start <= month <= end
        # season wraps across the new year (e.g. 10 .. 4)
        return month >= start or month <= end

    def _season_setpoints(self, heating_season: bool):
        """Return the (day, night) set-point pair active for the current season."""
        p = self.state.parameters
        if heating_season:
            day = p['setpoint_day_c']
            night = p['setpoint_night_c']
            if day is None:
                day = p['heating_setpoint_day_c']
            if night is None:
                night = p['heating_setpoint_night_c']
            return day, night

        mode = str(p['cooling_season_mode']).lower()
        if mode == 'cooling':
            return p['cooling_setpoint_day_c'], p['cooling_setpoint_night_c']
        if mode != 'setback':
            self.logger.warning(
                f"Unknown cooling_season_mode '{mode}', falling back to 'setback'"
            )
        setback = p['cooling_season_setback_c']
        return setback, setback

    def step(self) -> None:
        self.logger.debug("state: %s", self.state)
        p = self.state.parameters
        hour = self.state.time.hour + self.state.time.minute / 60.0

        occupied = p['occupied_start_hour'] <= hour < p['occupied_end_hour']
        heating_season = self._is_heating_season(self.state.time.month)
        setpoint_day, setpoint_night = self._season_setpoints(heating_season)

        if occupied:
            self.state.outputs['PeopleNumber'] = p['people_occupied']
            self.state.outputs['LightsWatt'] = p['lights_peak_w']
            self.state.outputs['EEquipWatt'] = p['eequip_peak_w']
            self.state.outputs['OthEquRadWatt'] = p['otheq_rad_peak_w']
            self.state.outputs['OthEquFCWatt'] = p['otheq_fc_peak_w']
            self.state.outputs['ZoneSetPoint'] = setpoint_day
        else:
            self.state.outputs['PeopleNumber'] = 0.0
            self.state.outputs['LightsWatt'] = p['lights_base_w']
            self.state.outputs['EEquipWatt'] = p['eequip_base_w']
            self.state.outputs['OthEquRadWatt'] = p['otheq_rad_base_w']
            self.state.outputs['OthEquFCWatt'] = p['otheq_fc_base_w']
            self.state.outputs['ZoneSetPoint'] = setpoint_night

        self.state.outputs['HeatingSeason'] = 1.0 if heating_season else 0.0

    def finalize(self):
        pass
