"""
test_fmu_horizon_detection.py

Detection of an FMU's simulation horizon at registration time: how long the FMU
can run before it has to be restarted, and which calendar date it restarts from.
Both are read from the EnergyPlus RunPeriod the FMU was exported with, or from
the FMU's own DefaultExperiment, so a user never has to work them out by hand.

Run: pytest tests/test_fmu_horizon_detection.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.model_catalog.fmu_catalog_register import (  # noqa: E402
    detect_horizon_from_fmu,
    parse_idf_run_period,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUI0_IDF = REPO_ROOT / 'src/models/model_catalog/physical_models/resources/BUI0.idf'

FULL_YEAR = """
RunPeriod,
    TMY,                      !- Name
    1,                        !- Begin Month
    1,                        !- Begin Day of Month
    ,                         !- Begin Year
    12,                       !- End Month
    31,                       !- End Day of Month
    ,                         !- End Year
    Monday,                   !- Day of Week for Start Day
    No,                       !- Use Weather File Holidays and Special Days
    Yes;                      !- Use Weather File Rain Indicators
"""

MARCH_START = FULL_YEAR.replace('    1,                        !- Begin Month',
                                '    3,                        !- Begin Month')

WINTER_WRAP = """
RunPeriod,
    Heating season,
    11,                       !- Begin Month
    1,                        !- Begin Day of Month
    ,                         !- Begin Year
    3,                        !- End Month
    31,                       !- End Day of Month
    ,                         !- End Year
    Monday;                   !- Day of Week for Start Day
"""


def write_idf(tmp_path, text):
    path = tmp_path / 'model.idf'
    path.write_text(text)
    return str(path)


# ---------------------------------------------------------------------------
# RunPeriod parsing
# ---------------------------------------------------------------------------

def test_full_year_run_period(tmp_path):
    got = parse_idf_run_period(write_idf(tmp_path, FULL_YEAR))
    assert got == {'max_sim_time': 31536000.0, 'sim_start_date': '01-01'}


def test_run_period_start_date_is_kept(tmp_path):
    """A model-local time of 0 is the FMU's own start date, not 1 January."""
    got = parse_idf_run_period(write_idf(tmp_path, MARCH_START))
    assert got['sim_start_date'] == '03-01'


def test_run_period_wrapping_across_the_new_year(tmp_path):
    got = parse_idf_run_period(write_idf(tmp_path, WINTER_WRAP))
    # 1 Nov .. 31 Mar inclusive = 30 + 31 + 31 + 28 + 31 = 151 days
    assert got == {'max_sim_time': 151 * 86400.0, 'sim_start_date': '11-01'}


def test_idf_without_a_run_period_is_not_a_horizon(tmp_path):
    path = tmp_path / 'empty.idf'
    path.write_text('Building,\n    My Building;\n')
    assert parse_idf_run_period(str(path)) is None


def test_missing_idf_file_is_reported_as_no_horizon():
    assert parse_idf_run_period('/nonexistent/model.idf') is None


def test_malformed_run_period_dates_are_rejected(tmp_path):
    bad = FULL_YEAR.replace('    12,                       !- End Month',
                            '    99,                       !- End Month')
    assert parse_idf_run_period(write_idf(tmp_path, bad)) is None


@pytest.mark.skipif(not BUI0_IDF.exists(), reason='BUI0.idf not present')
def test_real_bui0_idf_is_a_full_year_from_january():
    got = parse_idf_run_period(str(BUI0_IDF))
    assert got == {'max_sim_time': 31536000.0, 'sim_start_date': '01-01'}


# ---------------------------------------------------------------------------
# DefaultExperiment fallback
# ---------------------------------------------------------------------------

def test_default_experiment_gives_the_span():
    md = SimpleNamespace(defaultExperiment=SimpleNamespace(startTime=0.0, stopTime=86400.0))
    assert detect_horizon_from_fmu(md) == {'max_sim_time': 86400.0, 'sim_start_date': None}


def test_default_experiment_span_is_measured_from_its_start_time():
    md = SimpleNamespace(defaultExperiment=SimpleNamespace(startTime=3600.0, stopTime=90000.0))
    assert detect_horizon_from_fmu(md)['max_sim_time'] == 86400.0


def test_no_default_experiment_means_unbounded():
    assert detect_horizon_from_fmu(SimpleNamespace(defaultExperiment=None)) is None
    assert detect_horizon_from_fmu(SimpleNamespace()) is None


def test_zero_or_negative_span_is_not_a_horizon():
    md = SimpleNamespace(defaultExperiment=SimpleNamespace(startTime=0.0, stopTime=0.0))
    assert detect_horizon_from_fmu(md) is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
