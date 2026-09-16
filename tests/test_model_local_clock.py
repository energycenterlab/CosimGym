"""
test_model_local_clock.py

The model-local clock in BaseModel: the sawtooth simulated-time axis a model
keeps when it cannot run for the whole scenario in one go (an EnergyPlus FMU
stops at the end of its RunPeriod and has to be restarted).

Covers the two things that must hold:
  * a model that declares no ``max_sim_time`` behaves exactly as before - local
    time is federation time and nothing is ever restarted;
  * a model that declares one is repositioned at its horizon, and by the reset
    modes, with the right origin and anchor.

Run: pytest tests/test_model_local_clock.py
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.base_model import BaseModel                      # noqa: E402
from models.model_catalog.ModelCatalog import ModelMetadata  # noqa: E402


REAL_PERIOD = 900.0          # 15 min per tick
HORIZON = 9000.0             # 10 ticks of model-local time


class ClockProbe(BaseModel):
    """Minimal concrete model that records every reposition it is asked for."""

    def __init__(self, *args, **kwargs):
        self.repositions = []
        self.steps = []
        self.dates = []
        super().__init__(*args, **kwargs)

    def initialize(self):
        pass

    def step(self):
        self.steps.append(self.local_time())
        self.dates.append(self.state.time)

    def finalize(self):
        pass

    def _reposition_backend(self, target_ts):
        self.repositions.append(target_ts)


def make_model(max_sim_time=None):
    metadata = ModelMetadata(
        name='clock_probe',
        class_name='ClockProbe',
        module_path='tests.test_model_local_clock',
        version='1.0.0',
        description='probe',
        max_sim_time=max_sim_time,
    )
    config = SimpleNamespace(
        user_defined={},
        inputs=[],
        outputs=[],
        parameters={},
        init_state={},
        start_time='2024-01-01T00:00:00',
        real_period=REAL_PERIOD,
        reset_mode=None,
    )
    return ClockProbe('clock_probe.0', metadata, config, logging.getLogger('test'))


# ---------------------------------------------------------------------------
# No horizon declared: nothing changes
# ---------------------------------------------------------------------------

def test_unbounded_model_local_time_is_federation_time():
    m = make_model(max_sim_time=None)
    for ts in range(1, 50):
        m._step(ts, {})
    # First tick sits at local time 0, then one real_period per tick.
    assert m.steps == [max(0, ts - 1) * REAL_PERIOD for ts in range(1, 50)]


def test_unbounded_model_is_never_repositioned_by_the_horizon_guard():
    m = make_model(max_sim_time=None)
    for ts in range(1, 200):
        m._step(ts, {})
    assert m.repositions == []
    assert m.epoch_index == 0


# ---------------------------------------------------------------------------
# Horizon declared: the model wraps on its own clock
# ---------------------------------------------------------------------------

def test_horizon_restarts_the_model_and_restarts_local_time():
    m = make_model(max_sim_time=HORIZON)
    for ts in range(1, 26):
        m._step(ts, {})

    # 10 ticks fit inside the horizon (0 .. 9*period, the 11th would end past it).
    n_per_epoch = int(HORIZON // REAL_PERIOD)
    assert m.steps[:n_per_epoch] == [i * REAL_PERIOD for i in range(n_per_epoch)]
    assert m.steps[n_per_epoch] == 0.0, "the step after the horizon restarts at local 0"
    assert max(m.steps) + REAL_PERIOD <= HORIZON, "no step may run past the horizon"


def test_horizon_wrap_is_periodic_and_counts_epochs():
    m = make_model(max_sim_time=HORIZON)
    for ts in range(1, 26):
        m._step(ts, {})
    assert m.repositions == [1, 1], "two restarts in 25 ticks of a 10-tick horizon"
    assert m.epoch_index == 2


def test_horizon_guard_fires_whatever_the_reset_policy_is():
    """The limit belongs to the model, not to the RL task: no reset config here."""
    m = make_model(max_sim_time=HORIZON)
    assert m.reset_mode is None
    for ts in range(1, 15):
        m._step(ts, {})
    assert m.epoch_index == 1


# ---------------------------------------------------------------------------
# Reset modes drive the same primitive
# ---------------------------------------------------------------------------

def test_full_reset_returns_to_local_zero_and_reanchors():
    m = make_model(max_sim_time=HORIZON)
    for ts in range(1, 6):
        m._step(ts, {})
    m.reset(mode='full')

    assert m.repositions[-1] == 1
    assert m.ts_shift == 5, "the next tick must count as the model's first step"
    m._step(6, {})
    assert m.steps[-1] == 0.0


def test_rolling_reset_moves_to_the_requested_absolute_start_point():
    m = make_model(max_sim_time=HORIZON)
    for ts in range(1, 6):
        m._step(ts, {})
    m.reset(mode='rolling', ts=3)

    assert m.repositions[-1] == 3
    m._step(6, {})
    assert m.steps[-1] == 2 * REAL_PERIOD, "the episode replays from the rolling start tick"


def test_rolling_start_point_past_the_horizon_wraps_into_the_cycle():
    m = make_model(max_sim_time=HORIZON)
    m._step(1, {})
    # 13 ticks is past the 10-tick horizon: the window slid off the end of the
    # run period and comes back round to the start of it.
    m.reset(mode='rolling', ts=13)
    assert m.repositions[-1] == 3


def test_none_reset_leaves_the_clock_alone():
    m = make_model(max_sim_time=HORIZON)
    for ts in range(1, 6):
        m._step(ts, {})
    before = (m.ts_shift, m.epoch_index)
    m.reset(mode='none')
    assert (m.ts_shift, m.epoch_index) == before
    assert m.repositions == []


def test_reset_on_an_unbounded_model_does_not_disturb_its_timeline():
    m = make_model(max_sim_time=None)
    for ts in range(1, 6):
        m._step(ts, {})
    m.reset(mode='full')
    m._step(6, {})
    assert m.steps[-1] == 0.0, "a full reset still restarts a plain model's clock"


# ---------------------------------------------------------------------------
# state.time follows the model clock, so every model rewinds together
# ---------------------------------------------------------------------------

def test_datetime_is_plain_elapsed_time_without_restarts():
    """Nothing changes for a model that is never reset or restarted."""
    m = make_model(max_sim_time=None)
    for ts in range(1, 20):
        m._step(ts, {})
        assert m.state.time == m.start_time + timedelta(seconds=ts * REAL_PERIOD)


def test_datetime_rewinds_at_a_horizon_restart():
    m = make_model(max_sim_time=HORIZON)
    n_per_epoch = int(HORIZON // REAL_PERIOD)
    for ts in range(1, n_per_epoch + 2):
        m._step(ts, {})
    # The model is back at the moment of its very first step, not at tick 11.
    assert m.dates[-1] == m.dates[0]


def test_datetime_rewinds_on_a_full_reset():
    """A schedule model must not run on past the model it feeds."""
    m = make_model(max_sim_time=None)
    for ts in range(1, 6):
        m._step(ts, {})
    m.reset(mode='full')
    m._step(6, {})
    assert m.dates[-1] == m.dates[0], "episode 2 starts at the same simulated moment as episode 1"


def test_two_models_reset_together_stay_at_the_same_moment():
    """The point of the whole mechanism: a feeder and a building never diverge."""
    feeder = make_model(max_sim_time=None)
    building = make_model(max_sim_time=HORIZON)
    for ts in range(1, 6):
        feeder._step(ts, {})
        building._step(ts, {})
        assert feeder.state.time == building.state.time

    # The federate resets every model it owns, exactly as BaseFederate does.
    for m in (feeder, building):
        m.reset(mode='full')
    for ts in range(6, 12):
        feeder._step(ts, {})
        building._step(ts, {})
        assert feeder.state.time == building.state.time


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
