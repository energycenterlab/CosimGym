"""tab_s3_metrics: S3 model-formalism swap (SAC on the BUI0 EnergyPlus FMU).

Reports the same columns as tab_s2_metrics for the FMU building, PLUS the honest
config diff that the swap required — that diff IS the paper's claim, so it is
emitted as data rather than prose.

IMPORTANT / honesty note: the BUI0 FMU exposes no electrical power. Its energy
column is the THERMAL heating load (HeatingLoadTarget, W -> kWh), NOT electricity,
so it is NOT directly comparable to S1/S2 energy numbers. The emitted table states
which signal produced each number (`signals_used`).

Usage: python tab_s3_metrics.py [--seeds 42 43 44]
"""
from __future__ import annotations
import argparse, statistics
from pathlib import Path

import metrics as M

OUT = Path("results/paper_case_study")
OUT.mkdir(parents=True, exist_ok=True)

# Config deltas required to swap simple_building (S2) -> BUI0 EnergyPlus FMU (S3).
# Verified against src/scenarios/cs_s2_sac.yaml and bui0_setpoint_SAC.yaml.
SWAP_DIFF = [
    ("building model", "simple_building (1R1C, hand-written Python)", "bui0_building_fmu (EnergyPlus FMI 2.0)"),
    ("federates in federation", "3: weather (CSV) + heatpump + building", "2: feeder (5 gain schedules) + building_federate (FMU)"),
    ("observations", "weather.0.T_ext + building.0.T_indoor", "building_federate.0.TBuilding (+ HeatingLoadTarget as unobserved reward extra)"),
    ("action var", "heatpump.0.modulation", "building_federate.0.ZoneSetPoint"),
    ("action space", "implicit from catalog (modulation, dimensionless)", "explicit box, bounds [16,24] °C"),
    ("actuator", "separate simple_heatpump federate (Q_heat/P_elec/COP)", "none — HVAC internal to the FMU"),
    ("reward", "…reward_functions.building_heatpump_comfort", "…reward_functions.bui0_setpoint_comfort"),
    ("agent.policy", "unset (Box obs, backend default)", "MultiInputPolicy (Dict obs space)"),
    ("real_period", "60 s", "600 s (fixed by FMU co-sim step)"),
    ("episode_length", "2880 steps (2 days)", "144 steps (1 day; FMU needs whole-day multiples)"),
    ("reset.mode", "full / rolling / none (model re-inits per episode)", "none (EnergyPlus cannot re-initialize mid-run)"),
]


def fmt(vals):
    if not vals:
        return "MISSING"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    a = ap.parse_args()

    cdh, en, labels = [], [], set()
    for n in [f"cs_s3_fmu_s{s}" for s in a.seeds] + ["cs_s3_fmu"]:
        try:
            m = M.scenario_metrics(n)
        except Exception:
            continue
        if m["comfort_degree_hours"] is None and m["energy_kwh"] is None:
            continue
        if m["comfort_degree_hours"] is not None:
            cdh.append(m["comfort_degree_hours"])
        if m["energy_kwh"] is not None:
            en.append(m["energy_kwh"])
        labels.add(f"{m['temp_var']} + {m['power_var']}")

    hdr = ("controller", "building_model", "comfort_degree_hours", "energy_kwh(THERMAL)", "signals_used")
    row = ("SAC", "BUI0 EnergyPlus FMU", fmt(cdh), fmt(en), "; ".join(sorted(labels)) or "MISSING")
    (OUT / "tab_s3_metrics.csv").write_text(
        ",".join(hdr) + "\n" + ",".join(f'"{c}"' for c in row) + "\n")

    md = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |",
          "| " + " | ".join(row) + " |", "",
          "**Config diff required by the formalism swap (S2 → S3):**", "",
          "| aspect | S2 (`simple_building`) | S3 (EnergyPlus FMU) |", "| --- | --- | --- |"]
    md += [f"| {a_} | `{b}` | `{c}` |" for a_, b, c in SWAP_DIFF]
    md += ["",
           "_The RL agent class, backend, and the four-axis config structure are UNCHANGED; "
           "only the declarative bindings above differ. Energy for S3 is the FMU's THERMAL "
           "heating load — not electricity — so it is NOT comparable to the S1/S2 energy column._"]
    (OUT / "tab_s3_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
