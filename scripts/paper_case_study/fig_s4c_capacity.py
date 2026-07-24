"""fig_s4c_capacity: total sites/federates achieved vs machine count (S4c-real).

Reads the cs_s4c_shard_{1,2,3}m rows from results/paper_case_study/exec_metrics.csv
(produced by run_profiled.py on the REAL manager/machine_a/machine_b infrastructure)
and plots total capacity (sites, federates annotated) achieved per machine count,
alongside the per-shard wall-clock. The story: capacity scales via FEDERATION
SHARDING (one broker per machine, each sized at the known-safe ~33-federate
ceiling), not via a raw single-federation speedup.
"""
from __future__ import annotations
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
CSV = OUT / "exec_metrics.csv"

SITES_PER_SHARD = 8  # matches generate_scale_sharded.py default


def main():
    wall = {}
    if CSV.exists():
        with CSV.open() as f:
            for row in csv.DictReader(f):
                m = re.match(r"cs_s4c_shard_(\d+)m", row["scenario"])
                if not m or "PASS" not in row["marker"]:
                    continue
                try:
                    wall[int(m.group(1))] = float(row["wall_median_s"])
                except ValueError:
                    pass

    if not wall:
        print("No cs_s4c_shard_*m PASS rows in exec_metrics.csv — run the S4c-real sweep first.")
        return

    xs = sorted(wall)
    sites = [x * SITES_PER_SHARD for x in xs]
    feds = [x * (4 * SITES_PER_SHARD + 1) for x in xs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar([str(x) for x in xs], sites, color="tab:blue")
    ax1.set_ylim(0, max(sites) * 1.35)
    for i, x in enumerate(xs):
        ax1.annotate(f"{feds[i]} federates\n({x} shard{'s' if x>1 else ''})",
                     xy=(i, sites[i]), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8)
    ax1.set_xlabel("number of machines")
    ax1.set_ylabel("total sites (buildings) achieved")
    ax1.set_title("Capacity scales linearly with machines\n(via per-machine federation sharding)", fontsize=10)
    ax1.grid(alpha=0.3, axis="y")

    ax2.plot(xs, [wall[x] for x in xs], "o-", color="tab:orange")
    ax2.set_xlabel("number of machines")
    ax2.set_ylabel("wall-clock (s, median of 3)")
    ax2.set_xticks(xs)
    ax2.set_title("Wall-clock per run\n(NOT flat — hierarchy-broker lockstep sync\nacross shards adds cost per additional machine)", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("S4c-real — federation-sharded capacity scaling (manager 112c + 2x cloud 32c, real SSH)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s4c_capacity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_s4c_capacity written. sites={dict(zip(xs, sites))} feds={dict(zip(xs, feds))} wall={wall}")


if __name__ == "__main__":
    main()
