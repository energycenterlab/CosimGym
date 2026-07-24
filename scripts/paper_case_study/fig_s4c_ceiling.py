"""fig_s4c_ceiling: single-broker (single-federation, local, zmq_ss) federate
ceiling, characterized FRESH on this real network today.

Reads results/paper_case_study/s4c_ceiling_sweep.csv (produced by the D-task
sweep over src/scenarios/cs_s4c_ceiling_{8,10,...}.yaml — one federation, all
federates local on the manager, core_type zmq_ss) and plots pass/fail vs
federate count.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
CSV = OUT / "s4c_ceiling_sweep.csv"


def main():
    rows = []
    if CSV.exists():
        with CSV.open() as f:
            for row in csv.DictReader(f):
                rows.append({
                    "sites": int(row["sites_per_shard"]),
                    "feds": int(row["n_federates"]),
                    "ok": row["pass"] == "True",
                    "wall": float(row["wall_s"]) if row["wall_s"] not in ("", "NA") else float("nan"),
                })

    if not rows:
        print("No rows in s4c_ceiling_sweep.csv — run the D-task ceiling sweep first.")
        return

    rows.sort(key=lambda r: r["feds"])
    feds = [r["feds"] for r in rows]
    ok = [r["ok"] for r in rows]
    wall = [r["wall"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    colors = ["tab:green" if o else "tab:red" for o in ok]
    ax1.bar([str(f) for f in feds], wall, color=colors)
    for i, f in enumerate(feds):
        label = "PASS" if ok[i] else "FAIL\n([-101] lost comms)"
        ax1.annotate(label, xy=(i, wall[i]), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=7.5, color=colors[i])
    ax1.set_xlabel("federate count (single broker, local, zmq_ss)")
    ax1.set_ylabel("wall-clock (s, 1 rep)")
    ax1.set_title("S4c-real — single-broker zmq_ss ceiling,\ncharacterized fresh on this network today", fontsize=10)
    ax1.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s4c_ceiling.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_s4c_ceiling written. rows={rows}")


if __name__ == "__main__":
    main()
