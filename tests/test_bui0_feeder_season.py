"""
test_bui0_feeder_season.py — season-aware set-point schedule of BUI0InputFeeder
(src/models/model_catalog/physical_models/bui0_input_feeder.py).

The model's step() only reads self.state (parameters + time) and writes
self.state.outputs, so it is exercised without HELICS/Redis by building the
instance with object.__new__ and injecting a State.

Covers: heating/cooling season boundaries (including the new-year wrap and a
non-wrapping season), day/night set-points per season, both cooling-season
modes, the deprecated setpoint_day_c/setpoint_night_c aliases, the
HeatingSeason output flag, and that the catalog declares every parameter the
model reads.

Run: pytest tests/test_bui0_feeder_season.py -v
"""

import logging
import os
import sys
from datetime import datetime

import pytest
import yaml

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC)

from models.base_model import State  # noqa: E402
from models.model_catalog.physical_models.bui0_input_feeder import BUI0InputFeeder  # noqa: E402

CATALOG = os.path.join(SRC, 'models', 'model_catalog', 'catalog.yaml')

DEFAULT_PARAMS = {
    'occupied_start_hour': 8.0,
    'occupied_end_hour': 18.0,
    'people_occupied': 4.0,
    'lights_peak_w': 400.0,
    'lights_base_w': 40.0,
    'eequip_peak_w': 500.0,
    'eequip_base_w': 50.0,
    'otheq_rad_peak_w': 100.0,
    'otheq_rad_base_w': 0.0,
    'otheq_fc_peak_w': 100.0,
    'otheq_fc_base_w': 0.0,
    'heating_season_start_month': 10,
    'heating_season_end_month': 4,
    'heating_setpoint_day_c': 21.0,
    'heating_setpoint_night_c': 18.0,
    'cooling_season_mode': 'setback',
    'cooling_season_setback_c': 12.0,
    'cooling_setpoint_day_c': 26.0,
    'cooling_setpoint_night_c': 28.0,
    'setpoint_day_c': None,
    'setpoint_night_c': None,
}


def make_feeder(**param_overrides):
    """Build a BUI0InputFeeder without running BaseModel.__init__."""
    feeder = object.__new__(BUI0InputFeeder)
    feeder.logger = logging.getLogger('test_bui0_feeder_season')
    params = dict(DEFAULT_PARAMS)
    params.update(param_overrides)
    feeder.state = State(parameters=params)
    return feeder


def step_at(feeder, when: datetime):
    feeder.state.time = when
    feeder.step()
    return feeder.state.outputs


class TestSeasonBoundaries:

    @pytest.mark.parametrize('month', [10, 11, 12, 1, 2, 3, 4])
    def test_wrapping_season_heating_months(self, month):
        feeder = make_feeder()
        assert feeder._is_heating_season(month) is True

    @pytest.mark.parametrize('month', [5, 6, 7, 8, 9])
    def test_wrapping_season_cooling_months(self, month):
        feeder = make_feeder()
        assert feeder._is_heating_season(month) is False

    def test_non_wrapping_season(self):
        feeder = make_feeder(heating_season_start_month=1, heating_season_end_month=3)
        assert feeder._is_heating_season(2) is True
        assert feeder._is_heating_season(3) is True
        assert feeder._is_heating_season(4) is False
        assert feeder._is_heating_season(12) is False


class TestSetPointSchedule:

    def test_heating_season_day_and_night(self):
        feeder = make_feeder()
        assert step_at(feeder, datetime(2024, 1, 15, 12, 0))['ZoneSetPoint'] == 21.0
        assert step_at(feeder, datetime(2024, 1, 15, 3, 0))['ZoneSetPoint'] == 18.0

    def test_cooling_season_setback_mode(self):
        feeder = make_feeder()
        assert step_at(feeder, datetime(2024, 7, 15, 12, 0))['ZoneSetPoint'] == 12.0
        assert step_at(feeder, datetime(2024, 7, 15, 3, 0))['ZoneSetPoint'] == 12.0

    def test_cooling_season_cooling_mode(self):
        feeder = make_feeder(cooling_season_mode='cooling')
        assert step_at(feeder, datetime(2024, 7, 15, 12, 0))['ZoneSetPoint'] == 26.0
        assert step_at(feeder, datetime(2024, 7, 15, 3, 0))['ZoneSetPoint'] == 28.0

    def test_unknown_cooling_mode_falls_back_to_setback(self):
        feeder = make_feeder(cooling_season_mode='nonsense')
        assert step_at(feeder, datetime(2024, 7, 15, 12, 0))['ZoneSetPoint'] == 12.0

    def test_deprecated_aliases_override_heating_setpoints(self):
        feeder = make_feeder(setpoint_day_c=22.5, setpoint_night_c=17.5)
        assert step_at(feeder, datetime(2024, 1, 15, 12, 0))['ZoneSetPoint'] == 22.5
        assert step_at(feeder, datetime(2024, 1, 15, 3, 0))['ZoneSetPoint'] == 17.5
        # aliases are heating-season only
        assert step_at(feeder, datetime(2024, 7, 15, 12, 0))['ZoneSetPoint'] == 12.0


class TestOtherOutputs:

    def test_heating_season_flag(self):
        feeder = make_feeder()
        assert step_at(feeder, datetime(2024, 1, 15, 12, 0))['HeatingSeason'] == 1.0
        assert step_at(feeder, datetime(2024, 7, 15, 12, 0))['HeatingSeason'] == 0.0

    def test_occupancy_schedule_unaffected_by_season(self):
        feeder = make_feeder()
        for when in (datetime(2024, 1, 15, 12, 0), datetime(2024, 7, 15, 12, 0)):
            out = step_at(feeder, when)
            assert out['PeopleNumber'] == 4.0
            assert out['LightsWatt'] == 400.0
            assert out['EEquipWatt'] == 500.0
        for when in (datetime(2024, 1, 15, 3, 0), datetime(2024, 7, 15, 3, 0)):
            out = step_at(feeder, when)
            assert out['PeopleNumber'] == 0.0
            assert out['LightsWatt'] == 40.0
            assert out['EEquipWatt'] == 50.0


class TestCatalogContract:

    def test_catalog_declares_every_parameter_the_model_reads(self):
        with open(CATALOG) as fh:
            catalog = yaml.safe_load(fh)
        entry = catalog['models']['bui0_input_feeder'] if 'models' in catalog else catalog['bui0_input_feeder']
        assert set(DEFAULT_PARAMS) <= set(entry['parameters'])
        assert 'HeatingSeason' in entry['outputs']
        assert entry['parameters']['setpoint_day_c']['default_value'] is None
        assert entry['parameters']['setpoint_night_c']['default_value'] is None
