"""
test_fmu_state_snapshot.py

Restarting an FMU by saving and restoring its state.

An FMU that implements fmi2GetFMUstate/fmi2SetFMUstate (FMI 3: fmi3GetFMUState)
can be put back at any saved moment instantly: the saved blob carries the whole
internal state, so there is no restart, no replay, and nothing to remember about
the inputs it saw. This is the path every reset takes when the FMU supports it.

FMUs that do not - EnergyPlus exports report canGetAndSetFMUstate=false, and
their real state lives inside the EnergyPlus process rather than in the handful
of declared FMI variables - fall back to re-instantiating the slave and stepping
it forward again.

Uses the FMI 3.0 Feedthrough FMU shipped in the repo, which supports the
capability. Run: pytest tests/test_fmu_state_snapshot.py
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

REPO_ROOT = Path(__file__).resolve().parents[1]
FEEDTHROUGH = REPO_ROOT / 'src/models/model_catalog/physical_models/resources/Feedthrough_FMI3.fmu'

pytestmark = pytest.mark.skipif(not FEEDTHROUGH.exists(), reason='Feedthrough_FMI3.fmu not present')

REAL_PERIOD = 60.0
IN_VAR = 'Float64_continuous_input'
OUT_VAR = 'Float64_continuous_output'


def make_model(max_sim_time=None, reset_mode=None, rolling_window=None, supports_rollback=None):
    from models.base_FMU_model import BaseFMUModel
    from models.model_catalog.ModelCatalog import ModelMetadata, ParameterSpec, ParameterType

    def spec(name):
        return ParameterSpec(name=name, type=ParameterType('float'), default_value=0.0)

    user_defined = {'fmu_source': {'type': 'local', 'path': str(FEEDTHROUGH)}}
    if supports_rollback is not None:
        user_defined['fmu_reset'] = {'supports_rollback': supports_rollback}

    metadata = ModelMetadata(
        name='test_feedthrough_fmi3',
        class_name='BaseFMUModel',
        module_path='models.base_FMU_model',
        version='1.0.0',
        description='feedthrough',
        inputs={IN_VAR: spec(IN_VAR)},
        outputs={OUT_VAR: spec(OUT_VAR)},
        max_sim_time=max_sim_time,
        user_defined=user_defined,
    )
    config = SimpleNamespace(
        user_defined={},
        inputs=[IN_VAR],
        outputs=[OUT_VAR],
        parameters={},
        init_state={IN_VAR: 0.0, OUT_VAR: 0.0},
        start_time='2024-01-01T00:00:00',
        end_time='2024-01-02T00:00:00',
        time_stop=1440,
        real_period=REAL_PERIOD,
        reset_mode=reset_mode,
        rolling_window=rolling_window,
        n_episodes=None,
    )
    return BaseFMUModel('feedthrough.0', metadata, config, logging.getLogger('test'))


@pytest.fixture
def model():
    m = make_model()
    yield m
    m.finalize()


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def test_capability_is_read_from_the_fmu_itself(model):
    """The FMU's own modelDescription decides, not a catalog claim."""
    assert model._can_snapshot is True


def test_catalog_can_disable_a_capability_the_fmu_claims():
    """For an FMU that advertises state save/restore but does not honour it."""
    m = make_model(supports_rollback=False)
    try:
        assert m._can_snapshot is False
    finally:
        m.finalize()


# ---------------------------------------------------------------------------
# Restoring instead of restarting
# ---------------------------------------------------------------------------

def test_initial_state_is_saved_at_startup(model):
    assert 1 in model._state_snapshots


def test_reset_restores_the_state_without_re_instantiating(model):
    """The whole point: no new slave, no replayed steps."""
    for ts in range(1, 6):
        model._step(ts, {IN_VAR: float(ts)})
    instances_before = model._instance_count

    model.reset(mode='full')

    assert model._instance_count == instances_before, (
        "a saved state must be restored in place, not by starting a new slave"
    )


def test_restored_model_reproduces_its_original_trajectory(model):
    """Step, rewind, step again: the second pass must match the first."""
    first_pass = []
    for ts in range(1, 6):
        out = model._step(ts, {IN_VAR: float(ts)})
        first_pass.append(out[OUT_VAR])

    model.reset(mode='full')

    second_pass = []
    for ts in range(6, 11):
        out = model._step(ts, {IN_VAR: float(ts - 5)})
        second_pass.append(out[OUT_VAR])

    assert second_pass == first_pass


def test_no_input_history_is_kept_when_states_can_be_saved():
    """A saved state carries everything, so remembering inputs is pointless."""
    m = make_model(reset_mode='rolling', rolling_window=2)
    try:
        for ts in range(1, 6):
            m._step(ts, {IN_VAR: float(ts)})
        assert m._input_history == []
    finally:
        m.finalize()


def test_rolling_start_point_is_saved_in_passing():
    """The next episode's start point is saved while the current one runs."""
    window = 2
    m = make_model(reset_mode='rolling', rolling_window=window)
    try:
        for ts in range(1, 6):
            m._step(ts, {IN_VAR: float(ts)})
        assert 1 + window in m._state_snapshots, 'start points are 1, 1+W, 1+2W'
    finally:
        m.finalize()


def test_rolling_reset_rewinds_to_the_saved_start_point():
    window = 2
    m = make_model(reset_mode='rolling', rolling_window=window)
    try:
        for ts in range(1, 6):
            m._step(ts, {IN_VAR: float(ts)})
        instances_before = m._instance_count

        m.reset(mode='rolling', ts=1 + window)

        assert m._instance_count == instances_before, "rewind must not restart the slave"
        assert m.local_ts(6) == 1 + window
    finally:
        m.finalize()


# ---------------------------------------------------------------------------
# Horizon restart uses the same mechanism
# ---------------------------------------------------------------------------

def test_horizon_restart_restores_rather_than_re_instantiates():
    horizon = 5 * REAL_PERIOD          # 5 steps
    m = make_model(max_sim_time=horizon)
    try:
        for ts in range(1, 5):
            m._step(ts, {IN_VAR: float(ts)})
        instances_before = m._instance_count

        for ts in range(5, 9):
            m._step(ts, {IN_VAR: float(ts)})

        assert m.epoch_index >= 1, "the horizon must have been reached"
        assert m._instance_count == instances_before, (
            "an FMU that can restore its state need not be re-instantiated at its horizon"
        )
    finally:
        m.finalize()


def test_freed_slave_drops_its_saved_states(model):
    """Saved states belong to a live slave; keeping them after a free would crash."""
    model._step(1, {IN_VAR: 1.0})
    assert model._state_snapshots
    model._teardown()
    assert model._state_snapshots == {}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# ---------------------------------------------------------------------------
# A rewind that rewinds nothing
# ---------------------------------------------------------------------------

def test_reposition_to_the_tick_the_slave_is_on_does_nothing():
    """`rolling_window == reset period` asks for the tick that is coming anyway.

    Restarting for that would cost a restart and put a discontinuity into a run
    that was supposed to be continuous - and it is the configuration the docs
    recommend as the free one, so it has to actually be free. Checked with the
    catalog disabling state save/restore, which is the path that would restart.
    """
    m = make_model(supports_rollback=False, reset_mode='rolling', rolling_window=4)
    try:
        for ts in range(1, 5):
            m._step(ts, {IN_VAR: float(ts)})
        instances_before = m._instance_count

        m.reset(mode='rolling', ts=5)          # tick 5 is the one it is about to run

        assert m._instance_count == instances_before, (
            'the slave must not be re-instantiated to reach the tick it is already on')
        assert m.local_ts(5) == 5
    finally:
        m.finalize()
