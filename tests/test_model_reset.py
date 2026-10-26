"""
test_model_reset.py

Clock and reset mechanics of BaseModel: the absolute clock (ts/time, the
federation's, never moved by a reset) and the relative clock (rel_ts/rel_time, the
simulated moment the model represents, moved by resets and horizon restarts).

Run: pytest tests/test_model_reset.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.base_model import BaseModel                      # noqa: E402
from models.model_catalog.ModelCatalog import ModelMetadata  # noqa: E402

PERIOD = 900.0                       # 15 min per tick
START = datetime(2024, 3, 1)


class Probe(BaseModel):
    """Records the clocks seen in every step and the reset hooks it gets."""

    def __init__(self, *args, **kwargs):
        self.seen = []               # (ts, rel_ts, time, rel_time) at each step
        self.hook_calls = []
        super().__init__(*args, **kwargs)

    def initialize(self):
        pass

    def step(self):
        s = self.state
        self.seen.append((s.ts, s.rel_ts, s.time, s.rel_time))

    def finalize(self):
        pass

    def reset(self, mode='full'):
        self.hook_calls.append((mode, self.state.rel_ts))


def make_model(max_sim_time=None, sim_start_date=None):
    metadata = ModelMetadata(
        name='probe', class_name='Probe', module_path='tests.test_model_reset',
        version='1.0.0', description='probe',
        max_sim_time=max_sim_time, sim_start_date=sim_start_date,
    )
    config = SimpleNamespace(
        user_defined={}, inputs=[], outputs=[], parameters={}, init_state={},
        start_time=START.isoformat(), real_period=PERIOD,
    )
    return Probe('probe.0', metadata, config, logging.getLogger('test'))


def run(model, first_ts, last_ts):
    for ts in range(first_ts, last_ts + 1):
        model._step(ts, {})


def test_no_reset_relative_clock_equals_absolute():
    m = make_model()
    run(m, 1, 5)
    for ts, rel_ts, time, rel_time in m.seen:
        assert rel_ts == ts - 1
        assert time == rel_time == START + timedelta(seconds=ts * PERIOD)
    assert m.state.rel_ts == 5


def test_full_reset_rewinds_relative_clock_only():
    m = make_model()
    run(m, 1, 5)
    m._reset('full')
    assert (m.state.ts, m.state.time) == (5, START + timedelta(seconds=5 * PERIOD))
    assert (m.state.rel_ts, m.state.rel_time) == (0, START)
    assert m.hook_calls == [('full', 0)]
    run(m, 6, 6)
    ts, rel_ts, time, rel_time = m.seen[-1]
    assert (ts, rel_ts) == (6, 0)
    assert rel_time == START + timedelta(seconds=PERIOD)      # same as the very first step


def test_rolling_reset_moves_relative_clock_only():
    m = make_model()
    run(m, 1, 4)
    m._reset('rolling', rel_ts_target=8)
    assert m.state.ts == 4
    assert m.state.rel_ts == 8
    run(m, 5, 5)
    assert m.seen[-1][3] == START + timedelta(seconds=9 * PERIOD)
    assert m.state.rel_ts == 9


def test_reset_restores_init_state_and_stores_span():
    m = make_model()
    m.state.outputs['x'] = 3
    m.init_state.outputs['x'] = 1
    m._reset('full', span_ticks=7)
    assert m.state.outputs['x'] == 1
    assert m.span_ticks == 7


def test_horizon_restarts_model_but_relative_clock_keeps_going():
    # horizon of 10 ticks, origin 1 Mar (== start): restart at rel_ts 10, 20
    m = make_model(max_sim_time=10 * PERIOD, sim_start_date='03-01')
    run(m, 1, 25)
    assert m.hook_calls == [('horizon', 10), ('horizon', 20)]
    assert [s[1] for s in m.seen] == list(range(25))               # never folded back
    assert [s[0] for s in m.seen] == list(range(1, 26))            # absolute clock untouched
    assert m.seen[-1][3] == START + timedelta(seconds=25 * PERIOD)  # next "March", not the previous one


def test_horizon_restart_resets_state_to_init():
    m = make_model(max_sim_time=3 * PERIOD)
    m.init_state.outputs['x'] = 1
    m.state.outputs['x'] = 99
    run(m, 1, 4)                       # restart before the 4th step; step() leaves outputs alone
    assert m.state.outputs['x'] == 1


def test_horizon_origin_before_start():
    # origin one day (96 ticks) before the scenario start, window 200 ticks:
    # first window ends 104 ticks after the start, then every 200 ticks
    m = make_model(max_sim_time=200 * PERIOD, sim_start_date='02-29')
    assert m._origin_s == -96 * PERIOD
    run(m, 1, 320)
    assert [c[1] for c in m.hook_calls] == [104, 304]
    assert m.horizon_pos_s(0) == 96 * PERIOD
    assert m.horizon_pos_s(104) == 0
    assert m.horizon_pos_s(103) == 199 * PERIOD
    assert m.horizon_pos_s(200) == 96 * PERIOD


def test_horizon_pos_without_horizon_is_plain_clock():
    m = make_model()
    assert m.horizon_pos_s(7) == 7 * PERIOD


def test_rolling_beyond_horizon_keeps_raw_target_and_does_not_restart_next_step():
    m = make_model(max_sim_time=100 * PERIOD)        # origin == start
    m._reset('rolling', rel_ts_target=230)
    assert m.state.rel_ts == 230                     # the backend folds it, not the clock
    assert m.horizon_pos_s() == 30 * PERIOD
    run(m, 1, 1)
    assert m.hook_calls == [('rolling', 230)]        # only the rolling hook, no horizon restart
    m._reset('rolling', rel_ts_target=60)
    assert m.state.rel_ts == 60 and m.horizon_pos_s() == 60 * PERIOD


def test_no_horizon_never_restarts():
    m = make_model()
    run(m, 1, 50)
    assert m.hook_calls == []


# ---------------------------------------------------------------------------
# BaseFederate._reset: when it fires and which target each model gets
# ---------------------------------------------------------------------------

def make_federate(reset_type, ts, rolling_window=None, reset_length=10, mode='train',
                  n_resets_train=0, stop_time=1000):
    from core.BaseFederate import BaseFederate
    fed = object.__new__(BaseFederate)
    fed.logger = logging.getLogger('test')
    fed.name = 'fed'
    fed.mode, fed.ts = mode, ts
    fed.reset_type, fed.reset_length, fed.rolling_window = reset_type, reset_length, rolling_window
    fed.reset_count = 0
    fed.n_resets_train = n_resets_train
    fed.stop_time = stop_time
    fed.published = 0
    fed._publish_init_state = lambda: setattr(fed, 'published', fed.published + 1)
    model = make_model()
    run(model, 1, ts)
    fed.entities = [{'id': 'probe.0', 'object': model}]
    return fed, model


def test_federate_none_and_off_period_are_silent():
    for reset_type, ts, mode in (('none', 10, 'train'), (None, 10, 'train'),
                                 ('full', 7, 'train'), ('full', 10, 'test')):
        fed, model = make_federate(reset_type, ts, mode=mode)
        fed._reset()
        assert model.hook_calls == [] and fed.published == 0 and fed.reset_count == 0


def test_federate_full_reset_publishes_and_rewinds():
    fed, model = make_federate('full', 10)
    fed._reset()
    assert model.hook_calls == [('full', 0)]
    assert fed.published == 1 and fed.reset_count == 1


def test_federate_rolling_target_is_reset_count_times_window():
    fed, model = make_federate('rolling', 10, rolling_window=4)
    fed._reset()
    assert model.state.rel_ts == 4
    fed.ts = 20
    fed._reset()
    assert model.state.rel_ts == 8
    assert fed.published == 2 and model.state.ts == 10   # absolute clock untouched


def test_federate_reset_passes_span_to_the_models():
    # 3 resets in training: the first two are followed by another one, the last runs to the end
    fed, model = make_federate('full', 10, n_resets_train=3, stop_time=100)
    seen = []
    model.reset = lambda mode='full': seen.append(model.span_ticks)
    fed._reset()                                  # reset #1 at ts=10
    fed.ts = 20; fed._reset()                     # #2
    fed.ts = 30; fed._reset()                     # #3: last one, span = time_stop - ts
    assert seen == [10, 10, 70]


def test_federate_first_span_and_no_reset_on_last_tick():
    fed, model = make_federate('full', 10, n_resets_train=2, stop_time=100)
    assert fed._next_span() == 10                 # first run ends at the first reset
    fed.n_resets_train = 0
    assert fed._next_span() == 90                 # no resets: the rest of the scenario (ts=10)
    fed.n_resets_train, fed.ts, fed.stop_time = 2, 10, 10
    fed._reset()                                  # last tick of the run: nothing left to simulate
    assert model.hook_calls == [] and fed.reset_count == 0


def test_model_takes_initial_span_from_config():
    metadata = ModelMetadata(name='probe', class_name='Probe', module_path='tests.test_model_reset',
                             version='1.0.0', description='probe')
    config = SimpleNamespace(user_defined={}, inputs=[], outputs=[], parameters={}, init_state={},
                             start_time=START.isoformat(), real_period=PERIOD, initial_span_ticks=42)
    assert Probe('probe.0', metadata, config, logging.getLogger('test')).span_ticks == 42
    assert make_model().span_ticks is None


# ---------------------------------------------------------------------------
# ScenarioManager: agent-tick lengths -> federate ticks
# ---------------------------------------------------------------------------

def test_reset_lengths_convert_to_federate_ticks():
    from core.ScenarioManager import reset_in_federate_ticks
    values = {'episode_length': 20, 'reset_period': 10, 'rolling_window': 5}
    assert reset_in_federate_ticks(values, 60, 60, 'f') == values
    assert reset_in_federate_ticks(values, 60, 300, 'f') == {'episode_length': 4, 'reset_period': 2, 'rolling_window': 1}
    assert reset_in_federate_ticks(values, 900, 60, 'f')['episode_length'] == 300     # slower agent


def test_reset_lengths_not_whole_for_a_federate_raise():
    import pytest
    from core.ScenarioManager import reset_in_federate_ticks
    with pytest.raises(ValueError, match=r"federate 'weather' \(real_period=900s\)"):
        reset_in_federate_ticks({'rolling_window': 10}, 60, 900, 'weather')
