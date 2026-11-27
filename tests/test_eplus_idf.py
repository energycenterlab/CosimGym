"""Pure-python parts of the EnergyPlus model (no EnergyPlus needed): IDF utilities, I/O spec, install resolution."""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.eplus_idf import Idf, IdfError, normalize_io, parse_version, weekday_after  # noqa: E402
from models.eplus_registry import EnergyPlusNotFound, resolve  # noqa: E402

BUI0 = Path(__file__).resolve().parents[1] / "src/models/model_catalog/physical_models/resources/BUI0.idf"


def test_roundtrip_and_queries():
    idf = Idf.parse(BUI0)
    assert idf.text() == BUI0.read_text()
    assert idf.version()[:2] == (23, 1) and idf.timestep_per_hour() == 6 and idf.zone_step_s() == 600
    rp = idf.run_period()
    assert rp.days == 365 and rp.start_dow == "Monday"


def test_external_interface_conversion_and_run_period_edit():
    idf = Idf.parse(BUI0)
    ext = idf.convert_external_interface()
    assert len(ext.inputs) == 6 and len(ext.outputs) == 2 and not idf.find_prefix("ExternalInterface")
    idf.set_run_period(date(2001, 3, 15), date(2001, 3, 16), weekday_after("Monday", 73))
    rp = idf.run_period()
    assert (rp.begin, rp.end, rp.start_dow) == (date(2001, 3, 15), date(2001, 3, 16), "Thursday")
    ins, outs = normalize_io({}, ext)
    assert ins["ZoneSetPoint"]["component_type"] == "Schedule:Constant" and outs["TBuilding"]["kind"] == "variable"


def test_normalize_io_rejects_bad_spec():
    with pytest.raises(IdfError):
        normalize_io({"inputs": {"x": {"nope": 1}}})
    with pytest.raises(IdfError):
        normalize_io({"outputs": {"y": {"meter": "M", "agg": "median"}}})


def test_versions_and_resolution():
    assert parse_version("V23-1-0") == (23, 1, 0) and parse_version("23.1.0-87ed9199d4") == (23, 1, 0)
    with pytest.raises(EnergyPlusNotFound):
        resolve((99, 1, 0))


def test_schedule_shorthand_and_pattern_expansion():
    from models.eplus_idf import expand_io, has_pattern
    idf = Idf.parse(BUI0)
    sched = next(o.fields[0] for c in ("Schedule:Constant", "Schedule:Compact") for o in idf.find(c))
    ins, _ = normalize_io({"inputs": {"S": {"schedule": sched.lower()}}}, idf=idf)
    assert ins["S"]["control_type"] == "Schedule Value" and ins["S"]["key"] == sched
    with pytest.raises(IdfError):
        normalize_io({"inputs": {"S": {"schedule": "NO_SUCH"}}}, idf=idf)
    io = {"outputs": {"T": {"variable": {"name": "Zone Mean Air Temperature", "key": "Z*"}}}}
    listing = "OutputVariable,Zone Mean Air Temperature,Z A\nOutputVariable,Zone Mean Air Temperature,Z B\nOutputVariable,Zone Mean Air Temperature,Other\n"
    assert has_pattern(io) and not has_pattern({"outputs": {"T": {"meter": "M"}}})
    assert list(expand_io(io, listing, [])["outputs"]) == ["T_Z_A", "T_Z_B"]
    with pytest.raises(IdfError):
        expand_io({"outputs": {"T": {"variable": {"name": "Zone Mean Air Temperature", "key": "Q*"}}}}, listing, [])
    clash = "OutputVariable,Zone Mean Air Temperature,Z-A\nOutputVariable,Zone Mean Air Temperature,Z A\n"
    with pytest.raises(IdfError):          # 'Z-A' and 'Z A' both become T_Z_A
        expand_io(io, clash, [])
    with pytest.raises(IdfError):          # expanded name hits an explicit entry
        expand_io({"outputs": {**io["outputs"], "T_Z_A": {"meter": "M"}}}, listing, [])
