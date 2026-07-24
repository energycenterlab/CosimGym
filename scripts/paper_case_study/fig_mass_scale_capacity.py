"""fig_mass_scale_capacity: total buildings ACHIEVED (real, PASS only) vs
machine count, for the mass-scale K-per-shard design (the SAME sharded
4-federates-per-machine pattern as generate_mass_instances.py --shard-machines,
mandatory core_type=zmq_ss so it composes with real SSH to machine_a/machine_b).

Contrast with fig_s4c_capacity.py's finding (federate-COUNT sharding scales
~linearly with machine count): here the axis being pushed per machine is K
(n_instances inside 3 of the 4 federates), and the max K/machine that
completes reliably SHRINKS as real machines are added — this is the headline,
counter-intuitive-but-real finding of this sweep. See
mass_scale_bottleneck_analysis.md for the full narrative.

Data: results/paper_case_study/mass_scale_metrics.csv, family == real_ssh
(2m/3m, real SSH to machine_a/machine_b) plus the 1-machine zmq_ss control
(family local_zmqss) as the machines=1 anchor point (same core_type/design,
no actual network hop). Only PASS rows plotted; the largest FAILING K at each
machine count is annotated as the observed ceiling.
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


def main():
    if not CSV.exists():
        print(f"missing {CSV}")
        return
    rows = list(csv.DictReader(CSV.open()))

    # best (max K) PASS row per machine count, and best (min K) FAIL row per
    # machine count (the first-observed failure just above the safe ceiling)
    best_pass = {}   # machines -> (k, total_buildings)
    first_fail = {}  # machines -> k
    for row in rows:
        if row["family"] not in ("local_zmqss", "real_ssh"):
            continue
        m = int(row["machines"])
        k = int(row["k_per_machine"])
        if row["result"] == "PASS":
            if m not in best_pass or k > best_pass[m][0]:
                best_pass[m] = (k, int(row["total_buildings"]))
        elif row["result"].startswith("FAIL"):
            if m not in first_fail or k < first_fail[m]:
                first_fail[m] = k

    machines = sorted(best_pass)
    totals = [best_pass[m][1] for m in machines]
    ks = [best_pass[m][0] for m in machines]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(machines, totals, "o-", color="tab:purple", markersize=9)
    for m, tot, k in zip(machines, totals, ks):
        label = f"{tot} buildings (K={k}/machine)"
        if m in first_fail:
            label += f"\n[K={first_fail[m]}/machine FAILS]"
        ax.annotate(label, xy=(m, tot), xytext=(14, 12), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5)

    ax.set_xticks(machines)
    ax.set_xlim(0.7, max(machines) + 0.8)
    ax.set_xlabel("number of real machines (manager + SSH remotes, zmq_ss)")
    ax.set_ylabel("total buildings — largest K/machine that completed reliably")
    ax.set_ylim(0, max(totals) * 1.55)
    ax.set_title(
        "Mass-scale (K-per-shard) capacity DROPS as real machines are added\n"
        "— opposite of the federate-COUNT sharding result in fig_s4c_capacity",
        fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_mass_scale_capacity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_mass_scale_capacity written. best_pass={best_pass} first_fail={first_fail}")


if __name__ == "__main__":
    main()
