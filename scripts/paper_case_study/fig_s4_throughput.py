"""fig_s4_throughput: wall-clock vs n_instances, sequential vs parallel model exec.

Reads results/paper_case_study/exec_metrics.csv rows produced by run_profiled.py
for the cs_s4_vert_{seq,par}_N<k> scenarios and plots two curves. Annotates the
worker count (default max_parallel_workers = min(N, cpu_count)) and machine cores.
"""
from __future__ import annotations
import csv, os, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
CSV = OUT / "exec_metrics.csv"


def main():
    seq, par = {}, {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            m = re.match(r"cs_s4_vert_(seq|par)_N(\d+)", row["scenario"])
            if not m or "PASS" not in row["marker"]:
                continue
            mode, n = m.group(1), int(m.group(2))
            try:
                w = float(row["wall_median_s"])
            except ValueError:
                continue
            (seq if mode == "seq" else par)[n] = w

    ns = sorted(set(seq) | set(par))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if seq:
        xs = sorted(seq); ax.plot(xs, [seq[n] for n in xs], "o-", label="sequential", color="tab:gray")
    if par:
        xs = sorted(par); ax.plot(xs, [par[n] for n in xs], "s-", label="parallel (persistent workers)", color="tab:green")
    ax.set_xlabel("model instances per federate (N)  [× 4 federates]")
    ax.set_ylabel("wall-clock (s, median of 3)")
    ax.set_title("S4a — vertical scaling: sequential vs parallel model execution")
    ax.grid(alpha=0.3)
    ax.legend()
    cores = os.cpu_count()
    ax.annotate(f"machine: {cores} cores · workers = min(N, {cores})",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="dimgray")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s4_throughput.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_s4_throughput written. seq={seq} par={par}")


if __name__ == "__main__":
    main()
