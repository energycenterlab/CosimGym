"""fig_s1_traces: zone temperature (comfort band shaded) + heat-pump power vs time.

Reads the latest cs_s1_baseline results and writes PDF+PNG (300 dpi) plus the
tab_s1 baseline metrics row (CSV+MD) into results/paper_case_study/.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

OUT = Path("results/paper_case_study")
OUT.mkdir(parents=True, exist_ok=True)
SCEN = "cs_s1_baseline"


def hours(times):
    t0 = times[0]
    return [(t - t0).total_seconds() / 3600.0 for t in times]


def main():
    d = M.latest_sim_dir(SCEN)
    b = M.load_storage(d, "building_federate")
    hp = M.load_storage(d, "heatpump_federate")
    tT, T = M.series(b, "outputs", "building_federate.0", "T_indoor")
    tP, P = M.series(hp, "outputs", "heatpump_federate.0", "P_elec")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.axhspan(M.COMFORT_LOWER, M.COMFORT_UPPER, color="tab:green", alpha=0.15,
                label=f"comfort band [{M.COMFORT_LOWER}, {M.COMFORT_UPPER}] °C")
    ax1.plot(hours(tT), T, color="tab:blue", lw=1.2, label="zone temp")
    ax1.set_ylabel("Zone temp (°C)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(hours(tP), [p / 1000.0 for p in P], color="tab:red", lw=1.0)
    ax2.set_ylabel("HP power (kW)")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(alpha=0.3)

    fig.suptitle("S1 baseline (PID) — Turin winter, 48 h", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s1_traces.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cdh = M.comfort_degree_hours(tT, T)
    e = M.energy_kwh(tP, P)
    (OUT / "tab_s1_metrics.csv").write_text(
        "controller,comfort_degree_hours,energy_kwh\n"
        f"PID,{cdh:.4f},{e:.4f}\n")
    (OUT / "tab_s1_metrics.md").write_text(
        "| controller | comfort degree-hours | energy kWh |\n"
        "| --- | --- | --- |\n"
        f"| PID | {cdh:.3f} | {e:.3f} |\n")
    print(f"fig_s1_traces written; PID comfort={cdh:.3f} dh, energy={e:.3f} kWh (sim {d.name})")


if __name__ == "__main__":
    main()
