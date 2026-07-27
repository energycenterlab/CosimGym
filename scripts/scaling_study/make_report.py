#!/usr/bin/env python3
"""Report/plot generator for the CosimGym scaling study (D5).

Reads D2's `bench.csv` (always) and D4's `fit_params.json` (optional --
degrades gracefully if absent, skipping only the model-derived plot/overlay)
and produces PNG charts + a single `report.md` that embeds them with short
captions. Tells the two stories from `docs/future_and_TODOs/scaling_study_plan.md`:
the seq-vs-par crossover / sync-cost / distribution framework (plan §1) and
the zmq_ss federate-ceiling investigation (plan §2).

CLI (CONTRACT.md D5):
    python scripts/scaling_study/make_report.py --bench results/scaling/bench.csv \\
        [--params results/scaling/fit_params.json] [--outdir results/scaling/report]

Design:
    Every plot function is independently defensive -- if the CSV lacks the
    rows/columns/variation a plot needs, it appends a "skipped" note instead
    of raising, so a thin or partial bench.csv (early in the sweep, or a
    smoke run) still produces a valid report. Colors/styling follow the
    `dataviz` skill's validated default palette (see
    ~/.claude/skills/dataviz -- categorical hues in fixed order, sequential
    single-hue, reserved status colors for success/failure, muted chrome).
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- dataviz skill palette (light mode; report is static PNGs, not a themed
# artifact, so we commit to one look) -----------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # blue,orange,aqua,yellow,magenta,green,violet,red
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "system-ui"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 9,
})


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK_PRIMARY, loc="left", pad=10, fontweight="bold")


def legend(ax, **kw):
    leg = ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, **kw)
    return leg


def savefig(fig, path, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def successful_rows(df):
    """Rows with no failure_mode (contract: empty/null on success)."""
    if "failure_mode" not in df.columns:
        return df
    fm = df["failure_mode"]
    return df[fm.isna() | (fm.astype(str).str.strip() == "")]


# ------------------------------------------------------------------------
# Plot 1 -- crossover (seq vs par)
# ------------------------------------------------------------------------
def plot_crossover(df, outdir):
    name, fname = "Crossover (seq vs par)", "01_crossover.png"
    need = {"mode", "M", "work"}
    if not need.issubset(df.columns):
        return skip(name, "bench.csv is missing one of mode/M/work columns")
    ok = successful_rows(df)
    sub = ok[ok["mode"].isin(["seq", "par"])].copy()
    if sub.empty or sub["mode"].nunique() < 2:
        return skip(name, "no successful rows with both mode=seq and mode=par")

    ycol = "sim_wall_s" if "sim_wall_s" in sub.columns else "tick_mean_s"
    sub[ycol] = to_num(sub[ycol])
    sub["work_n"] = to_num(sub["work"])
    sub["M_n"] = to_num(sub["M"])

    xcol, xlabel = None, None
    if sub["work_n"].nunique(dropna=True) >= 2:
        xcol, xlabel = "work_n", "model work parameter"
    elif sub["M_n"].nunique(dropna=True) >= 2:
        xcol, xlabel = "M_n", "model instances per federate (M)"
    if xcol is None:
        return skip(name, "neither work nor M varies across seq/par rows "
                          "(need >=2 x points to show a crossover)")

    grp = sub.groupby(["mode", xcol])[ycol].median().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"seq": CAT[0], "par": CAT[1]}
    for mode_, color in colors.items():
        m = grp[grp["mode"] == mode_].sort_values(xcol)
        if m.empty:
            continue
        ax.plot(m[xcol], m[ycol], marker="o", markersize=6, linewidth=2,
                color=color, label=mode_, zorder=3)

    # locate crossover: x values shared by both series, sign change in (par - seq)
    cross_note = ""
    piv = grp.pivot(index=xcol, columns="mode", values=ycol).dropna().sort_index()
    if {"seq", "par"}.issubset(piv.columns) and len(piv) >= 2:
        diff = piv["par"] - piv["seq"]
        sign = np.sign(diff.values)
        cross_x = None
        for i in range(len(sign) - 1):
            if sign[i] != 0 and sign[i + 1] != 0 and sign[i] != sign[i + 1]:
                x0, x1 = piv.index[i], piv.index[i + 1]
                d0, d1 = diff.values[i], diff.values[i + 1]
                frac = -d0 / (d1 - d0) if (d1 - d0) != 0 else 0.0
                cross_x = x0 + frac * (x1 - x0)
                break
        if cross_x is not None:
            ax.axvline(cross_x, color=INK_MUTED, linestyle="--", linewidth=1, zorder=1)
            ax.annotate(f"crossover ≈ {cross_x:.3g}", xy=(cross_x, ax.get_ylim()[1]),
                        xytext=(4, -4), textcoords="offset points",
                        fontsize=8, color=INK_SECONDARY, va="top")
            cross_note = f" Crossover located near {xlabel}={cross_x:.3g}."
        else:
            winner = "par" if diff.mean() < 0 else "seq"
            cross_note = f" No sign change in the sampled range -- {winner} wins throughout."

    style_axes(ax, xlabel=xlabel, ylabel=f"{ycol} (s, median of repeats)",
                title="Sequential vs parallel model execution")
    legend(ax)
    path = outdir / fname
    savefig(fig, path)
    caption = ("Median sim wall time (or tick time) for sequential vs "
               "`parallel_execution` model stepping, swept over "
               f"{xlabel}. Shows where the per-tick IPC/dispatch overhead "
               "of parallel workers stops paying for itself." + cross_note)
    return rendered(name, fname, caption)


# ------------------------------------------------------------------------
# Plot 2 -- sync curve s(N)
# ------------------------------------------------------------------------
def plot_sync_curve(df, outdir):
    name, fname = "Sync curve s(N)", "02_sync_curve.png"
    if not {"core_type", "tick_mean_s"}.issubset(df.columns):
        return skip(name, "bench.csv is missing core_type/tick_mean_s columns")
    ok = successful_rows(df).copy()
    if ok.empty:
        return skip(name, "no successful rows")
    ok["tick_mean_s"] = to_num(ok["tick_mean_s"])

    if "N" in ok.columns and to_num(ok["N"]).nunique(dropna=True) >= 2:
        ok["x"], xlabel = to_num(ok["N"]), "federates per federation (N)"
    elif {"F", "N"}.issubset(ok.columns):
        ok["x"] = to_num(ok["F"]) * to_num(ok["N"])
        xlabel = "total federates (F×N)"
    else:
        return skip(name, "no N or F*N column to sweep")
    if ok["x"].nunique(dropna=True) < 2:
        return skip(name, "N (or F*N) does not vary across successful rows")

    grp = ok.groupby(["core_type", "x"])["tick_mean_s"].median().reset_index()
    plotted = 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ct in enumerate(sorted(grp["core_type"].dropna().unique())):
        m = grp[grp["core_type"] == ct].sort_values("x")
        if len(m) < 2:
            continue  # need >=2 x points per series to show a curve
        ax.plot(m["x"], m["tick_mean_s"], marker="o", markersize=6, linewidth=2,
                color=CAT[i % len(CAT)], label=ct, zorder=3)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return skip(name, "no core_type has >=2 distinct N (or F×N) points")

    style_axes(ax, xlabel=xlabel, ylabel="tick_mean_s (median of repeats)",
                title="Per-tick HELICS sync cost vs federate count")
    legend(ax)
    path = outdir / fname
    savefig(fig, path)
    caption = ("Median per-tick time as federate count grows, one line per "
               "`core_type`. Rising slope is the HELICS broker sync cost "
               "s(N) the cost model fits per core type -- steeper lines "
               "mean that core type's broker saturates sooner.")
    return rendered(name, fname, caption)


# ------------------------------------------------------------------------
# Plot 3 -- roofline / distribution
# ------------------------------------------------------------------------
def plot_roofline(df, outdir):
    name, fname = "Roofline / distribution", "03_roofline.png"
    need = {"F", "N", "M", "placement", "throughput_inst_steps_s"}
    if not need.issubset(df.columns):
        return skip(name, "bench.csv is missing F/N/M/placement/throughput_inst_steps_s")
    ok = successful_rows(df).copy()
    if ok.empty:
        return skip(name, "no successful rows")
    ok["insts"] = to_num(ok["F"]) * to_num(ok["N"]) * to_num(ok["M"])
    ok["thr"] = to_num(ok["throughput_inst_steps_s"])
    ok = ok.dropna(subset=["insts", "thr"])
    if ok.empty:
        return skip(name, "throughput_inst_steps_s / instance-count all NaN")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    placements = sorted(ok["placement"].dropna().unique())
    if len(placements) > 8:
        placements = placements[:8]
    for i, pl in enumerate(placements):
        m = ok[ok["placement"] == pl].sort_values("insts")
        style = dict(color=CAT[i % len(CAT)], label=pl)
        if len(m) >= 2:
            ax.plot(m["insts"], m["thr"], marker="o", markersize=6, linewidth=2,
                    zorder=3, **style)
        else:
            ax.scatter(m["insts"], m["thr"], s=48, zorder=3, **style)

    style_axes(ax, xlabel="total model instances (F×N×M)",
                ylabel="throughput (instance-steps/s)",
                title="Achieved throughput vs total instances, by placement")
    legend(ax)
    path = outdir / fname
    savefig(fig, path)
    n_single = sum(1 for pl in placements if len(ok[ok["placement"] == pl]) < 2)
    extra = (" Series with a single sample point are shown as markers only "
             "(not enough sweep yet to see saturation)." if n_single else "")
    caption = ("Instance-steps/s achieved as total instance count grows, "
               "split by `placement` (local vs distributed). Distribution "
               "should only lift throughput once compute dominates sync -- "
               "flat/declining lines mean the run is sync-bound, not "
               "compute-bound." + extra)
    return rendered(name, fname, caption)


# ------------------------------------------------------------------------
# Plot 4 -- ceiling vs network
# ------------------------------------------------------------------------
def plot_ceiling(df, outdir):
    name, fname = "Ceiling vs network", "04_ceiling_vs_network.png"
    need = {"core_type", "failure_mode"}
    if not need.issubset(df.columns):
        return skip(name, "bench.csv is missing core_type/failure_mode")
    d = df.copy()
    if "N" in d.columns and to_num(d["N"]).nunique(dropna=True) >= 2:
        d["x"], xlabel = to_num(d["N"]), "federates per federation (N)"
    elif {"F", "N"}.issubset(d.columns):
        d["x"] = to_num(d["F"]) * to_num(d["N"])
        xlabel = "total federates (F×N)"
    else:
        return skip(name, "no N or F*N column to sweep")
    if d["x"].nunique(dropna=True) < 2:
        return skip(name, "N (or F*N) does not vary -- no ceiling to locate")

    d["failed"] = ~(d["failure_mode"].isna() | (d["failure_mode"].astype(str).str.strip() == ""))
    groups = sorted(d["core_type"].dropna().unique())
    if not groups:
        return skip(name, "core_type column is empty")

    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.5), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    any_failure = d["failed"].any()
    for ax, ct in zip(axes, groups):
        sub = d[d["core_type"] == ct]
        ok_pts = sub[~sub["failed"]]
        bad_pts = sub[sub["failed"]]
        ax.scatter(ok_pts["x"], [1] * len(ok_pts), marker="o", s=64,
                   color=STATUS_GOOD, label="success", zorder=3)
        ax.scatter(bad_pts["x"], [1] * len(bad_pts), marker="x", s=80,
                   color=STATUS_CRITICAL, label="failure", linewidths=2, zorder=3)
        if not bad_pts.empty:
            first_fail = bad_pts["x"].min()
            ax.axvline(first_fail, color=STATUS_CRITICAL, linestyle="--",
                        linewidth=1, zorder=1)
        ax.set_yticks([])
        style_axes(ax, xlabel=xlabel, title=str(ct))
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_GOOD,
                          markersize=8, label="success"),
               plt.Line2D([0], [0], marker="x", color=STATUS_CRITICAL, markersize=9,
                          label="failure", linewidth=0)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Federate-count ceiling by core_type / network topology", y=1.1,
                 color=INK_PRIMARY, fontweight="bold", x=0.02, ha="left")
    path = outdir / fname
    savefig(fig, path)
    if any_failure:
        caption = ("Success (green) vs failure (red, marked with the row's "
                   "`failure_mode`) as federate count rises, one panel per "
                   "`core_type`. The dashed line marks the first observed "
                   "failure per panel -- this is the zmq_ss NAT ceiling "
                   "(plan §2) if it is lower than the direct zmq/tcp panel's.")
    else:
        caption = ("Success/failure by federate count, one panel per "
                   "`core_type`. No failures observed yet in this data -- "
                   "the ceiling has not been reached at the sampled N.")
    return rendered(name, fname, caption)


# ------------------------------------------------------------------------
# Plot 5 -- predicted vs measured (needs fit_params.json)
# ------------------------------------------------------------------------
def predict_t_sim(row, params):
    """Plan §1 cost-model formula, evaluated from a locked fit_params.json."""
    model = row.get("model")
    c_tab = params.get("c", {}).get(model)
    if c_tab is None:
        return None
    work = row.get("work_n")
    work = 0.0 if (work is None or (isinstance(work, float) and math.isnan(work))) else work
    c = c_tab.get("a", 0.0) + c_tab.get("b", 0.0) * work

    M = row.get("M_n") or 0
    if row.get("mode") == "par":
        W = row.get("W_n") or 1
        W = W if W and W > 0 else 1
        compute = math.ceil(M / W) * c + params.get("O_par", 0.0)
    else:
        compute = M * c

    s_tab = params.get("s", {}).get(row.get("core_type"))
    if s_tab is None:
        return None
    N = row.get("N_n") or 1
    sync = s_tab.get("s0", 0.0) + s_tab.get("s1", 0.0) * N

    comms = params.get("rtt_s", 0.0) if row.get("placement") not in (None, "local") else 0.0

    t_tick = compute + sync + comms
    n_ticks = row.get("perf_n_ticks_n") or row.get("n_ticks_n") or 0
    return n_ticks * t_tick


def plot_predicted_vs_measured(df, params, outdir):
    name, fname = "Predicted vs measured", "05_predicted_vs_measured.png"
    if params is None:
        return skip(name, "fit_params.json not provided -- skipping model-derived plot")
    need = {"model", "M", "mode", "core_type", "N", "sim_wall_s"}
    if not need.issubset(df.columns):
        return skip(name, "bench.csv is missing columns required for prediction")
    ok = successful_rows(df).copy()
    if ok.empty:
        return skip(name, "no successful rows to compare against predictions")

    for col in ("M", "N", "W", "work", "perf_n_ticks", "n_ticks"):
        if col in ok.columns:
            ok[col + "_n"] = to_num(ok[col])

    preds, measured = [], []
    for _, row in ok.iterrows():
        p = predict_t_sim(row.to_dict(), params)
        m = row.get("sim_wall_s")
        m = to_num(pd.Series([m])).iloc[0]
        if p is None or pd.isna(m):
            continue
        preds.append(p)
        measured.append(m)
    if len(preds) < 2:
        return skip(name, "fewer than 2 rows had both a valid prediction "
                          "(model/core_type covered by fit_params.json) and a measured sim_wall_s")

    preds, measured = np.array(preds), np.array(measured)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = min(preds.min(), measured.min())
    hi = max(preds.max(), measured.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(hi, 1e-9) * 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_MUTED,
            linestyle="--", linewidth=1, zorder=1, label="y = x")
    ax.scatter(preds, measured, s=48, color=CAT[0], zorder=3, label="runs")
    style_axes(ax, xlabel="predicted T_sim (s)", ylabel="measured sim_wall_s (s)",
                title="Cost-model prediction accuracy")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    legend(ax)
    path = outdir / fname
    savefig(fig, path)
    caption = (f"Cost-model-predicted T_sim (plan §1 formula, "
               f"fit_params.json) vs measured `sim_wall_s` for {len(preds)} "
               "runs whose model/core_type were covered by the fit. Points "
               "on the dashed y=x line are exact predictions; systematic "
               "over/under-prediction shows a term the model is missing.")
    return rendered(name, fname, caption)


# ------------------------------------------------------------------------
def skip(name, reason):
    return {"name": name, "rendered": False, "reason": reason}


def rendered(name, fname, caption):
    return {"name": name, "rendered": True, "file": fname, "caption": caption}


def write_report(results, outdir, bench_path, params_path):
    lines = [
        "# CosimGym Scaling Study -- Report",
        "",
        "Generated by `scripts/scaling_study/make_report.py` (D5) from "
        f"`{bench_path}`" + (f" and `{params_path}`" if params_path else
                              " (no fit_params.json -- model-derived plots skipped)") + ".",
        "",
        "Tells the two stories from `docs/future_and_TODOs/scaling_study_plan.md`: "
        "the cost-model framework (§1 -- crossover, sync cost, distribution roofline) "
        "and the zmq_ss federate-ceiling investigation (§2).",
        "",
    ]
    for r in results:
        lines.append(f"## {r['name']}")
        lines.append("")
        if r["rendered"]:
            lines.append(f"![{r['name']}]({r['file']})")
            lines.append("")
            lines.append(r["caption"])
        else:
            lines.append(f"*Skipped: {r['reason']}*")
        lines.append("")
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Build the D5 scaling-study report (plots + report.md).")
    ap.add_argument("--bench", required=True, help="path to D2's bench.csv")
    ap.add_argument("--params", default=None,
                    help="path to D4's fit_params.json (optional; degrades gracefully if absent)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "scaling" / "report"))
    args = ap.parse_args()

    bench_path = Path(args.bench)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bench_path, dtype=str)
    # keep everything as string on load (mixed empty/numeric columns are fiddly
    # otherwise); each plot coerces the columns it needs with to_num().

    params_path = Path(args.params) if args.params else (bench_path.parent / "fit_params.json")
    params = None
    if params_path.is_file():
        try:
            params = json.loads(params_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: could not read {params_path}: {e} -- degrading to no-params mode")
            params = None
    elif args.params:
        print(f"warning: --params {params_path} not found -- degrading to no-params mode")

    results = [
        plot_crossover(df, outdir),
        plot_sync_curve(df, outdir),
        plot_roofline(df, outdir),
        plot_ceiling(df, outdir),
        plot_predicted_vs_measured(df, params, outdir),
    ]
    write_report(results, outdir, bench_path, params_path if params else None)

    n_ok = sum(1 for r in results if r["rendered"])
    print(f"{n_ok}/{len(results)} plots rendered -> {outdir}")
    for r in results:
        status = "OK" if r["rendered"] else "SKIP"
        detail = r["file"] if r["rendered"] else r["reason"]
        print(f"  [{status}] {r['name']}: {detail}")


if __name__ == "__main__":
    main()
