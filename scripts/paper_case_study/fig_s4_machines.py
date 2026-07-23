"""fig_s4_machines: wall-clock vs number of machines (S4c distributed).

Reads the cs_s4_dist_{1,2,3}m rows from results/paper_case_study/exec_metrics.csv
(produced by run_profiled.py) and plots median wall-clock vs machine count.

NOTE: a loopback smoke run (all "machines" = 127.0.0.1) proves the MECHANISM only —
every federate still lands on one physical box, so no speedup is expected and the
figure would be misleading. Pass --loopback to label the figure accordingly.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
CSV = OUT / "exec_metrics.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loopback", action="store_true",
                    help="label as loopback mechanism-check, not a scaling result")
    a = ap.parse_args()

    pts = {}
    if CSV.exists():
        with CSV.open() as f:
            for row in csv.DictReader(f):
                m = re.match(r"cs_s4_dist_(\d+)m", row["scenario"])
                if not m or "PASS" not in row["marker"]:
                    continue
                try:
                    pts[int(m.group(1))] = float(row["wall_median_s"])
                except ValueError:
                    pass
    if not pts:
        print("No cs_s4_dist_* PASS rows in exec_metrics.csv — run the S4c sweep first.")
        return

    xs = sorted(pts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [pts[x] for x in xs], "o-", color="tab:purple")
    ax.set_xlabel("number of machines")
    ax.set_ylabel("wall-clock (s, median of 3)")
    ax.set_xticks(xs)
    title = "S4c — horizontal scaling across machines"
    if a.loopback:
        title += "\n(LOOPBACK mechanism check — all federates on one host, no speedup expected)"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s4_machines.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_s4_machines written. points={pts} loopback={a.loopback}")


if __name__ == "__main__":
    main()
