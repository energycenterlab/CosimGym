"""
test_fmu_reset.py

BaseFMUModel start/stop time and restart logic, without an FMU binary: a fake slave
records what the model asks of it.

Run: pytest tests/test_fmu_reset.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.base_FMU_model import BaseFMUModel                 # noqa: E402
from models.model_catalog.ModelCatalog import ModelMetadata    # noqa: E402

DAY = 86400.0
START = datetime(2024, 1, 1)


class FakeSlave:
    def __init__(self, log):
        self.log = log
        self.alive = True

    def setupExperiment(self, startTime, stopTime):
        self.log.append(('setup', startTime, stopTime))

    def doStep(self, currentCommunicationPoint, communicationStepSize):
        self.log.append(('step', currentCommunicationPoint))

    def terminate(self):
        self.log.append(('terminate',))

    def freeInstance(self):
        self.alive = False


class FakeFMU(BaseFMUModel):
    def _load_fmu(self):
        self.unzipdir, self.fmiVersion = 'x', '2.0'

    def _instantiate_fmu(self):
        self.fmu = FakeSlave(self.log)

    def _enter_initialization_mode(self): pass
    def _push_initial_state_to_fmu(self): pass
    def _exit_initialization_mode(self): pass
    def _resolve_fmu_workdir(self): return Path('.')
    def _in_fmu_workdir(self):
        import contextlib
        return contextlib.nullcontext()

    @property
    def log(self):
        return self.__dict__.setdefault('_log', [])


def make(engine=None, period=600.0, max_sim_time=None, span=None, start=START, **ud):
    user_defined = {'fmu_engine': engine} if engine else {}
    metadata = ModelMetadata(name='f', class_name='FakeFMU', module_path='tests.test_fmu_reset',
                             version='1', description='f', max_sim_time=max_sim_time,
                             sim_start_date='01-01' if max_sim_time else None,
                             user_defined=user_defined)
    config = SimpleNamespace(user_defined={}, inputs=[], outputs=[], parameters={}, init_state={},
                             start_time=start.isoformat(), end_time='2024-01-04T00:00:00',
                             real_period=period, time_stop=None, initial_span_ticks=span)
    return FakeFMU('f.0', metadata, config, logging.getLogger('test'))


def setups(m):
    return [e[1:] for e in m.log if e[0] == 'setup']


def test_generic_fmu_start_and_stop_follow_the_span():
    m = make(span=10)
    assert setups(m) == [(0.0, 6000.0)]
    m._reset('rolling', rel_ts_target=100, span_ticks=5)
    assert setups(m)[-1] == (60000.0, 63000.0)
    assert ('terminate',) in m.log


def test_generic_fmu_without_span_runs_to_scenario_end():
    assert setups(make())[0] == (0.0, 3 * DAY)


def test_step_uses_the_clock_position():
    m = make(span=10)
    m._reset('rolling', rel_ts_target=7, span_ticks=3)
    m._step(8, {})
    m._step(9, {})
    assert [e[1] for e in m.log if e[0] == 'step'] == [7 * 600.0, 8 * 600.0]


def test_energyplus_stop_is_rounded_up_to_whole_days_and_capped():
    m = make('energyplus', max_sim_time=10 * DAY, span=100)       # 100 * 600 s < 1 day
    assert setups(m) == [(0.0, DAY)]
    m._reset('rolling', rel_ts_target=int(8 * DAY / 600), span_ticks=int(5 * DAY / 600))
    assert setups(m)[-1] == (8 * DAY, 10 * DAY)                    # capped at the end of the run period


def test_energyplus_horizon_restart_goes_back_to_the_origin_with_remaining_span():
    m = make('energyplus', max_sim_time=2 * DAY, span=int(5 * DAY / 600))
    ticks = int(2 * DAY / 600)
    for ts in range(1, ticks + 2):
        m._step(ts, {})
    assert setups(m) == [(0.0, 2 * DAY), (0.0, 2 * DAY)]           # second window, remaining 3 days capped
    assert [e[1] for e in m.log if e[0] == 'step'][-1] == 0.0


def test_energyplus_start_is_counted_from_january_first():
    m = make('energyplus', start=datetime(2024, 2, 1), span=int(DAY / 600))
    assert setups(m)[0] == (31 * DAY, 32 * DAY)                    # no horizon: seconds since 1 January


def test_energyplus_mid_day_restart_is_refused():
    m = make('energyplus', max_sim_time=10 * DAY, span=int(DAY / 600))
    with pytest.raises(ValueError, match='midnight'):
        m._reset('rolling', rel_ts_target=72, span_ticks=10)       # 12:00


def test_energyplus_rejects_short_step_and_part_day_horizon():
    with pytest.raises(ValueError, match='60 s'):
        make('energyplus', period=30.0)
    with pytest.raises(ValueError, match='whole number'):
        make('energyplus', max_sim_time=1.5 * DAY)
