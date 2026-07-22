"""Shared metric definitions for the paper case study.

Comfort / energy metrics are defined ONCE here and reused by every figure/table
script. Definitions match EXPERIMENTS_INSTRUCTIONS.md rule 8:

- Comfort violation = degree-hours outside the deadband [19.5, 20.5] degC
  (= reward setpoint 20 +/- sigma 0.5), integrated over the horizon.
- Energy = integral of heat-pump electrical power (kWh).

Storage JSON layout (verified): each `<federate>_<mode>_storage.json` is
  {"inputs": {ent: {var: [..]}}, "outputs": {ent: {var: [..]}},
   "params": {ent: {p: [..]}}, "time": ["ISO", ...]}
`time` holds ISO datetime strings; each federate has its OWN time base
(multi-rate), so always pair a series with the time list from the SAME file.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

COMFORT_LOWER = 19.5  # degC
COMFORT_UPPER = 20.5  # degC


def latest_sim_dir(scenario: str, results_root="results") -> Path:
    """Return the newest results/<scenario>/<sim_id> directory."""
    base = Path(results_root) / scenario
    subs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not subs:
        raise FileNotFoundError(f"no results under {base}")
    return subs[-1]


def load_storage(sim_dir: Path, federate: str, federation="federation_1", mode="test") -> dict:
    f = Path(sim_dir) / federation / f"{federate}_{mode}_storage.json"
    return json.loads(f.read_text())


def series(storage: dict, direction: str, entity: str, var: str):
    """Return (times_datetime, values) for one variable. direction: inputs|outputs."""
    vals = storage[direction][entity][var]
    times = [datetime.fromisoformat(t) for t in storage["time"]]
    n = min(len(times), len(vals))
    return times[:n], vals[:n]


def _dt_hours(times) -> float:
    if len(times) < 2:
        return 0.0
    secs = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    secs.sort()
    return secs[len(secs) // 2] / 3600.0  # median step, in hours


def comfort_degree_hours(times, T_zone, lower=COMFORT_LOWER, upper=COMFORT_UPPER) -> float:
    dt_h = _dt_hours(times)
    viol = sum(max(0.0, lower - t) + max(0.0, t - upper) for t in T_zone)
    return viol * dt_h


def energy_kwh(times, P_elec_watt) -> float:
    dt_h = _dt_hours(times)
    return sum(P_elec_watt) * dt_h / 1000.0


if __name__ == "__main__":
    # self-test on cs_s1_baseline
    import sys
    scen = sys.argv[1] if len(sys.argv) > 1 else "cs_s1_baseline"
    d = latest_sim_dir(scen)
    b = load_storage(d, "building_federate")
    hp = load_storage(d, "heatpump_federate")
    tt, T = series(b, "outputs", "building_federate.0", "T_indoor")
    tp, P = series(hp, "outputs", "heatpump_federate.0", "P_elec")
    print(f"{scen}: comfort_degree_hours={comfort_degree_hours(tt, T):.3f}  energy_kwh={energy_kwh(tp, P):.4f}")
