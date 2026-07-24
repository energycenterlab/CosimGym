"""fig_mass_scale_curve: wall-clock and setup_duration vs K (n_instances/site
count) for the single-machine mass-instance sweep (cs_mass_k*, cs_mass_shard_1m_k*).

Reads results/paper_case_study/mass_scale_metrics.csv (hand-consolidated from
the real logs/cs_mass_*/*/execution_metrics.json files left by the interrupted
mass-scale agent — see mass_scale_bottleneck_analysis.md for how each row was
verified). Plots ONLY rows with result == PASS; FAIL/INTERRUPTED rows are
annotated as vertical failure markers, not plotted as if they were valid
completed-run timings.

Two series (both single machine, 4 federates: weather n=1, building/heatpump/
pid n=K, cross-wired dict-target subscriptions):
  - local_tcp:   core_type=tcp,    federation "district"
  - local_zmqss: core_type=zmq_ss, federation "shard_1" (the 1-machine control
                 for the real-SSH shard design, which mandates zmq_ss for NAT)

Two curves per series: total wall-clock (dominated by the 6-tick simulation
loop cost) and setup_duration (pure one-time registration cost) — the split
isolates BaseFederate._register_pubs/_register_subs (one-time, ~L383-444)
from the per-tick BaseFederate.run()/_publish_outputs()/update_storage()
serial `for entity in self.entities` loops (~L490-492, L790-811, L851-891),
both O(n_instances) but with a different multiplier (n_ticks=6 here).
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
CSV = OUT / "mass_scale_metrics.csv"

SERIES_STYLE = {
    "local_tcp":   dict(color="tab:blue",   marker="o", label="core_type=tcp (local, \"district\")"),
    "local_zmqss": dict(color="tab:red",    marker="s", label="core_type=zmq_ss (local, \"shard_1\" — real-SSH-compatible design)"),
}

# (series, K) -> label for annotated failure points, from rows with FAIL_* result
FAIL_POINTS = []


def main():
    if not CSV.exists():
        print(f"missing {CSV}")
        return

    rows = list(csv.DictReader(CSV.open()))
    series_data = {k: {"k": [], "wall": [], "setup": []} for k in SERIES_STYLE}
    for row in rows:
        fam = row["family"]
        if fam not in SERIES_STYLE:
            continue
        if row["result"] == "PASS":
            series_data[fam]["k"].append(int(row["k_per_machine"]))
            series_data[fam]["wall"].append(float(row["total_duration_s"]))
            series_data[fam]["setup"].append(float(row["setup_duration_s"]))
        elif row["result"].startswith("FAIL"):
            FAIL_POINTS.append((fam, int(row["k_per_machine"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for fam, style in SERIES_STYLE.items():
        d = series_data[fam]
        if not d["k"]:
            continue
        order = sorted(range(len(d["k"])), key=lambda i: d["k"][i])
        ks = [d["k"][i] for i in order]
        wall = [d["wall"][i] for i in order]
        setup = [d["setup"][i] for i in order]
        ax1.plot(ks, wall, marker=style["marker"], color=style["color"], label=style["label"])
        ax2.plot(ks, setup, marker=style["marker"], color=style["color"], label=style["label"])

    # Mark real measured failure points (time-to-failure, NOT a completed-run
    # timing — plotted as a distinct red X at the top of the axis, not on the
    # wall-clock trend line, to avoid implying it's a valid capacity point).
    for fam, k in FAIL_POINTS:
        ax1.axvline(k, color="firebrick", linestyle=":", alpha=0.5, linewidth=1)
    if FAIL_POINTS:
        ymax = ax1.get_ylim()[1] if series_data["local_tcp"]["wall"] else 100
        for i, (fam, k) in enumerate(sorted(set(FAIL_POINTS))):
            y_frac = 0.15 + 0.13 * (i % 3)
            ax1.annotate(f"FAIL K={k}\n(lost comms)", xy=(k, ymax * 0.02), xytext=(k, ymax * y_frac),
                         fontsize=6.5, color="firebrick", ha="center",
                         arrowprops=dict(arrowstyle="-", color="firebrick", alpha=0.5))

    ax1.set_xscale("log")
    ax1.set_xlabel("K (n_instances = buildings on this machine)")
    ax1.set_ylabel("total wall-clock (s)")
    ax1.set_title("Total wall-clock vs K\n(6-tick fixed horizon — dominated by per-tick\nserial entity loops, not one-time setup)", fontsize=9.5)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, loc="upper left")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("K (n_instances = buildings on this machine)")
    ax2.set_ylabel("setup_duration (s)")
    ax2.set_title("One-time setup/registration cost vs K\n(BaseFederate._register_pubs/_register_subs,\n~L383-444 — O(K) serial HELICS API calls)", fontsize=9.5)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Mass-scale single-machine sweep — real measured points only (FAIL = time-to-failure, not plotted as completed)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_mass_scale_curve.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_mass_scale_curve written.")
    for fam, d in series_data.items():
        print(f"  {fam}: K={d['k']} wall={d['wall']} setup={d['setup']}")
    print(f"  FAIL points: {FAIL_POINTS}")


if __name__ == "__main__":
    main()
