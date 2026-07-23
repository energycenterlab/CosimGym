"""tab_s2_metrics: PID vs SAC vs DQN — comfort degree-hours + energy kWh.

RL rows are aggregated over seed variants (mean +/- std). Evaluated on the
deterministic TEST partition. PID row comes from the S1 baseline run.

Rows with no results on disk are reported as MISSING (never fabricated).

Usage: python tab_s2_metrics.py [--seeds 42 43 44]
"""
from __future__ import annotations
import argparse, statistics
from pathlib import Path

import metrics as M

OUT = Path("results/paper_case_study")
OUT.mkdir(parents=True, exist_ok=True)


def collect(names):
    """Return (cdh[], energy[], labels, found_names) for the given scenario names."""
    cdh, en, labels, found = [], [], set(), []
    for n in names:
        try:
            m = M.scenario_metrics(n)
        except Exception:
            continue
        if m["comfort_degree_hours"] is None and m["energy_kwh"] is None:
            continue
        found.append(n)
        if m["comfort_degree_hours"] is not None:
            cdh.append(m["comfort_degree_hours"])
        if m["energy_kwh"] is not None:
            en.append(m["energy_kwh"])
        labels.add(f"{m['temp_var']} + {m['power_var']}")
    return cdh, en, sorted(labels), found


def agg(base: str, seeds):
    """Prefer the seed-variant runs; only fall back to the un-suffixed run if NO
    seed variant produced results (otherwise the un-seeded run would contaminate
    the mean/std of the seed sweep)."""
    res = collect([f"{base}_s{s}" for s in seeds])
    if res[3]:
        return res
    return collect([base])


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

    rows = []
    # PID baseline (single deterministic run, no seeds)
    cdh, en, lab, found = agg("cs_s1_baseline", [])
    rows.append(("PID (baseline)", fmt(cdh), fmt(en), len(found), "; ".join(lab)))
    for label, base in (("SAC", "cs_s2_sac"), ("DQN", "cs_s2_dqn")):
        cdh, en, lab, found = agg(base, a.seeds)
        rows.append((label, fmt(cdh), fmt(en), len(found), "; ".join(lab)))

    hdr = ("controller", "comfort_degree_hours", "energy_kwh", "n_runs", "signals_used")
    (OUT / "tab_s2_metrics.csv").write_text(
        ",".join(hdr) + "\n" +
        "\n".join(",".join(f'"{c}"' for c in r) for r in rows) + "\n")
    md = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md.append("")
    md.append(f"_Comfort deadband [{M.COMFORT_LOWER}, {M.COMFORT_UPPER}] °C; "
              f"energy = ∫P dt. RL rows: mean ± std over seeds {a.seeds}. "
              f"MISSING = not yet run (never fabricated)._")
    (OUT / "tab_s2_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
