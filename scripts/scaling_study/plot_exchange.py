#!/usr/bin/env python3
"""Phase-D figures — the DATA-EXCHANGE (coupling) view of the scaling study.

Companion to plot_instance_crossover.py / plot_crossover_clean.py (which cover
Part-A: raw compute scaling). Here every federate self-contained control run
(`exchange == none`) is subtracted from a wired run (`exchange == on`) sharing
the same Part-A knobs (F,N,M,mode,W,core_type,model,work,placement,n_ticks) to
isolate the marginal cost of HELICS coupling itself: `delta = wired - control`,
in microseconds. All five figures plot this delta, never a raw tick time.

  10_exchange_cost_vs_load.png  : delta vs n_edges (total links), one series per
                                   `distance`, msg_width=1/freq=1/same_step
                                   held fixed so topology load is the only
                                   variable. Least-squares line per series.
  11_fanout_shape.png           : grouped bars at FIXED n_edges, grouped by
                                   fanout pattern x distance — proves that
                                   WHERE edges concentrate (max_fed_in vs
                                   max_fed_out) matters more than edge count.
  12_payload_width.png          : delta vs msg_width (log-x), one series per
                                   distance; secondary axis = bytes/message.
  13_publish_frequency.png      : delta vs freq, annotated with % reduction
                                   relative to freq=1.
  14_causality.png              : same_step vs next_step, paired by matching
                                   (distance, fanout) topology — dumbbell chart.

Input: findings/phaseD_local_wide.csv if present and non-empty, else
findings/phaseD_local.csv (see CONTRACT.md "Part-B / Phase-D additions").
Rows with a non-empty `failure_mode` are dropped everywhere. Repeats are
aggregated by mean; spread across repeats is carried as error bars.

Any figure whose required slice is absent from the input CSV is skipped with
a printed notice instead of being drawn empty or interpolated.
"""
import argparse
import os
import statistics as st
from collections import defaultdict
from csv import DictReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
FINDINGS = os.path.join(HERE, "findings")

# palette (dataviz reference, light print surface) — matches 01-09 exactly
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

DIST_ORDER = ["intra_fed", "cross_fed", "cross_machine"]
DIST_COLOR = {"intra_fed": BLUE, "cross_fed": ORANGE, "cross_machine": AQUA}
DIST_MARKER = {"intra_fed": "o", "cross_fed": "s", "cross_machine": "^"}
DIST_HATCH = {"intra_fed": "", "cross_fed": "///", "cross_machine": "xxx"}
DIST_LABEL = {"intra_fed": "Intra-federation", "cross_fed": "Cross-federation",
              "cross_machine": "Cross-machine"}

FANOUT_ORDER = ["1to1", "1toN", "Nto1", "all2all"]
CAUS_COLOR = {"same_step": BLUE, "next_step": ORANGE}

BASE_COLS = ["F", "N", "M", "mode", "W", "core_type", "model", "work",
             "placement", "n_ticks"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


# --------------------------------------------------------------------- I/O

def default_bench_path():
    """Prefer the wide matrix if it exists and has data rows; else the narrow one."""
    wide = os.path.join(FINDINGS, "phaseD_local_wide.csv")
    narrow = os.path.join(FINDINGS, "phaseD_local.csv")
    if os.path.exists(wide):
        with open(wide) as f:
            n_lines = sum(1 for _ in f)
        if n_lines > 1:
            return wide
    return narrow


def load_rows(path):
    with open(path) as f:
        rows = list(DictReader(f))
    return [r for r in rows if not r.get("failure_mode", "")]


def basekey(r):
    return tuple(r[c] for c in BASE_COLS)


def wiredkey(r):
    return basekey(r) + (r["distance"], r["fanout"], r["msg_width"], r["freq"],
                          r["causality"])


# ----------------------------------------------------------- delta pipeline

def build_deltas(rows):
    """One aggregate record per wired (base+exchange-knob) combination.

    Returns a list of dicts: distance, fanout, msg_width, freq, causality, M,
    n_edges, n_subs, max_fed_in, max_fed_out, delta_us (mean), delta_std_us
    (sample stdev across repeats), n (repeat count).
    """
    control = defaultdict(list)
    wired = defaultdict(list)
    meta = {}
    for r in rows:
        bk = basekey(r)
        if r["exchange"] == "on":
            wk = wiredkey(r)
            wired[wk].append(float(r["tick_mean_s"]))
            meta[wk] = r
        else:
            control[bk].append(float(r["tick_mean_s"]))

    control_mean = {k: st.mean(v) for k, v in control.items() if v}

    out = []
    for wk, vals in wired.items():
        bk = wk[:len(BASE_COLS)]
        if bk not in control_mean:
            continue  # no matching control -> can't form a delta, skip
        cm = control_mean[bk]
        delta_us = [1e6 * (v - cm) for v in vals]
        r = meta[wk]
        out.append({
            "distance": r["distance"], "fanout": r["fanout"],
            "msg_width": int(r["msg_width"]), "freq": int(r["freq"]),
            "causality": r["causality"], "M": int(r["M"]), "N": int(r["N"]),
            "n_edges": int(r["n_edges"]), "n_subs": int(r["n_subs"]),
            "max_fed_in": int(r["max_fed_in"]), "max_fed_out": int(r["max_fed_out"]),
            "delta_us": st.mean(delta_us),
            "delta_std_us": st.stdev(delta_us) if len(delta_us) > 1 else 0.0,
            "n": len(delta_us),
        })
    return out


def select(records, **fixed):
    out = records
    for k, v in fixed.items():
        out = [r for r in out if r[k] == v]
    return out


def pick_sweep_M(rows, vary_field):
    """Among `rows` (already filtered to one distance/fanout/etc.), find the
    (N, M) whose group has the most distinct values of `vary_field` — the
    topology at which that knob was actually swept, so a different topology's
    single point can't contaminate the series.

    Grouping on (N, M), not M alone: the wide matrix sweeps N as well, and an
    N-only-varying cell shares M with the sweep cells. Keying on M alone pulled
    the N=32 all2all point (22 ms, 4096 edges) into the M=4 width/freq series
    and dwarfed every real point in it.

    Returns ((N, M), rows_at_that_topology) or (None, []) if none has >=2 values.
    """
    by_topo = defaultdict(list)
    for r in rows:
        by_topo[(r["N"], r["M"])].append(r)
    best_key, best_n = None, 1
    for topo, rs in by_topo.items():
        nvals = len(set(r[vary_field] for r in rs))
        if nvals > best_n:
            best_n, best_key = nvals, topo
    if best_key is None:
        return None, []
    return best_key, by_topo[best_key]


# --------------------------------------------------------------- plot base

def base_axes(ax, xlabel, ylabel):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=15.5, color=INK, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=15.5, color=INK, labelpad=9)
    ax.tick_params(axis="both", labelsize=13, colors=INK2, length=4)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def savefig(fig, name, outdir):
    out = os.path.join(outdir, name)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)
    return out


# ------------------------------------------------------------------ fig 10

def fig10_cost_vs_load(records, outdir):
    base = select(records, msg_width=1, freq=1, causality="same_step")
    dists = [d for d in DIST_ORDER if select(base, distance=d)]
    if not dists:
        print("[skip 10] no rows with msg_width=1, freq=1, causality=same_step")
        return
    for d in DIST_ORDER:
        if d not in dists:
            print(f"[skip 10:{d}] no data for distance={d} at fixed msg_width/freq/causality")

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.16)

    report = {}
    y_top = 0
    x_max = 0
    for i, d in enumerate(dists):
        # x = n_edges, the PRIMARY regressor. The wide matrix settled this:
        # per-federate load (max_fed_in) is real but secondary (~2.1 vs ~3.9
        # us/edge), and plotting against it alone hides the actual law -- two
        # cells with identical max_fed_in=64 differ 23x in cost when their edge
        # counts differ 64 vs 1024.
        pts = sorted(select(base, distance=d), key=lambda r: r["n_edges"])
        xs = np.array([r["n_edges"] for r in pts], dtype=float)
        ys = np.array([r["delta_us"] for r in pts], dtype=float)
        yerr = np.array([r["delta_std_us"] for r in pts], dtype=float)
        y_top = max(y_top, float((ys + yerr).max()))
        x_max = max(x_max, float(xs.max()))
        ax.errorbar(xs, ys, yerr=yerr, fmt=DIST_MARKER[d], color=DIST_COLOR[d],
                    ms=9, mfc=SURFACE, mew=2.2, lw=0, elinewidth=1.6, capsize=4,
                    ecolor=DIST_COLOR[d], label=DIST_LABEL[d], zorder=4)
        if len(set(xs)) >= 2:
            # Fit through the origin in RELATIVE terms: weight each point by 1/y
            # so the fit is not dictated by the largest cells (deltas span 3
            # orders of magnitude). Same estimator cost_model.py fit() uses.
            w = 1.0 / np.maximum(ys, 20.0)
            slope = float((w * w * xs * ys).sum() / (w * w * xs * xs).sum())
            intercept = 0.0
            xfit = np.array([xs.min(), xs.max()])
            ax.plot(xfit, slope * xfit, ls="--", lw=1.8,
                    color=DIST_COLOR[d], alpha=0.75, zorder=2)
            report[d] = (slope, intercept)
        else:
            report[d] = (None, None)

    base_axes(ax, "n_edges  (total HELICS input→target links in the scenario)",
              "Δ tick time from HELICS coupling  (µs)")
    # log-log: edge counts span 8 -> 4096 and deltas 50 us -> 22 ms, so a linear
    # axis would compress every district-scale point into the origin.
    ax.set_xscale("log")
    ax.set_yscale("log")
    leg = ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)

    # fit stats as a stacked corner note (axes coords) — avoids clipping at the
    # plot edge and collisions with data points that per-line end-of-fit labels hit.
    k = 0
    for d in dists:
        slope, intercept = report[d]
        if slope is None:
            continue
        ax.text(0.98, 0.04 + 0.065 * k,
                f"{DIST_LABEL[d]}: {slope:.2f} µs per edge",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11.5, color=DIST_COLOR[d],
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
        k += 1
    fig.suptitle("Cost of coupling: Δtick is linear in total edge count",
                 fontsize=18.5, color=INK, x=0.11, ha="left", y=0.955, weight="bold")
    fig.text(0.5, 0.03,
             "msg_width = 1 · freq = 1 (every tick) · causality = same_step · "
             "zmq · local · dashed = least-squares fit",
             fontsize=10, color=MUTED, ha="center")
    savefig(fig, "10_exchange_cost_vs_load.png", outdir)

    print("\nfig10 fit (delta_us = slope * n_edges, relative-weighted, through origin):")
    for d, (slope, intercept) in report.items():
        if slope is None:
            print(f"  {d}: insufficient distinct x values for a fit")
        else:
            print(f"  {d}: slope={slope:.3f} us/edge")


# ------------------------------------------------------------------ fig 11

def fig11_fanout_shape(records, outdir):
    base = select(records, msg_width=1, freq=1, causality="same_step")
    # Pick the n_edges value shared by the most (distance, fanout) combinations —
    # the richest apples-to-apples comparison available in this CSV.
    counts = defaultdict(set)
    for r in base:
        counts[r["n_edges"]].add((r["distance"], r["fanout"]))
    if not counts:
        print("[skip 11] no wired rows at msg_width=1, freq=1, causality=same_step")
        return
    n_edges_pick = max(counts, key=lambda k: len(counts[k]))
    rows = [r for r in base if r["n_edges"] == n_edges_pick]
    if len(rows) < 2:
        print(f"[skip 11] fewer than 2 fanout/distance combinations share an n_edges value "
              f"(best = n_edges={n_edges_pick}, {len(rows)} combo)")
        return

    dists = [d for d in DIST_ORDER if select(rows, distance=d)]
    fanouts = [f for f in FANOUT_ORDER if select(rows, fanout=f)]

    fig, ax = plt.subplots(figsize=(10.2, 6.6), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.16)

    n_groups = len(dists)
    width = 0.8 / max(n_groups, 1)
    x0 = np.arange(len(fanouts))
    report = {}
    y_top = 0
    for i, d in enumerate(dists):
        xs, ys, yerrs, labels = [], [], [], []
        for j, fo in enumerate(fanouts):
            hit = select(rows, distance=d, fanout=fo)
            if not hit:
                continue
            r = hit[0]
            xs.append(x0[j] + (i - (n_groups - 1) / 2) * width)
            ys.append(r["delta_us"])
            yerrs.append(r["delta_std_us"])
            labels.append(f"in={r['max_fed_in']}/out={r['max_fed_out']}")
            report[(d, fo)] = r["delta_us"]
        y_top = max(y_top, max((y + e for y, e in zip(ys, yerrs)), default=0))
        bars = ax.bar(xs, ys, width=width * 0.92, yerr=yerrs, capsize=4,
                      color=DIST_COLOR[d], hatch=DIST_HATCH[d], edgecolor=INK,
                      lw=1.0, ecolor=INK2, label=DIST_LABEL[d], zorder=3)
        # stagger the vertical offset by series index (i) so two adjacent bars
        # with similar heights (e.g. Nto1 intra vs cross) don't collide —
        # anchor above the error-bar cap, not the bar top, for the same reason.
        for bar, val, err, lab in zip(bars, ys, yerrs, labels):
            ax.annotate(f"{val:.0f} µs\n{lab}", xy=(bar.get_x() + bar.get_width() / 2,
                        val + err), xytext=(0, 5 + 15 * i), textcoords="offset points",
                        ha="center", fontsize=9.6, color=INK2)

    ax.set_xticks(x0)
    ax.set_xticklabels(fanouts, fontsize=13.5)
    base_axes(ax, f"Fanout pattern  (all sharing n_edges = {n_edges_pick})",
              "Δ tick time from HELICS coupling  (µs)")
    ax.set_ylim(0, y_top * 1.5)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Where edges attach matters more than how many",
                 fontsize=18.5, color=INK, x=0.10, ha="left", y=0.955, weight="bold")
    fig.text(0.5, 0.03,
             "bars annotated with (max_fed_in / max_fed_out) on the busiest federate · "
             "msg_width = 1 · freq = 1 · same_step · zmq · local",
             fontsize=10, color=MUTED, ha="center")
    savefig(fig, "11_fanout_shape.png", outdir)

    print(f"\nfig11 bar values (n_edges={n_edges_pick}):")
    for (d, fo), v in sorted(report.items()):
        r = [x for x in rows if x["distance"] == d and x["fanout"] == fo][0]
        print(f"  {d:<13} {fo:<8} delta_us={v:7.2f}  max_fed_in={r['max_fed_in']:<3} "
              f"max_fed_out={r['max_fed_out']}")


# ------------------------------------------------------------------ fig 12

def fig12_payload_width(records, outdir):
    # all2all is the only fanout swept across msg_width in this CSV. Pin M to
    # whichever value actually carries the sweep (a different M's lone
    # msg_width=1 baseline must not be mixed into the series).
    base = select(records, freq=1, causality="same_step", fanout="all2all")
    sweeps = {}
    for d in DIST_ORDER:
        M, rows = pick_sweep_M(select(base, distance=d), "msg_width")
        if M is not None:
            sweeps[d] = rows
    dists = [d for d in DIST_ORDER if d in sweeps]
    if not dists:
        print("[skip 12] no distance has >=2 msg_width points at a single M "
              "(freq=1, causality=same_step, fanout=all2all)")
        return
    for d in DIST_ORDER:
        if d not in dists and select(base, distance=d):
            print(f"[skip 12:{d}] no single M has >=2 msg_width points for distance={d}")

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.17)

    all_widths = set()
    for d in dists:
        pts = sorted(sweeps[d], key=lambda r: r["msg_width"])
        xs = [r["msg_width"] for r in pts]
        ys = [r["delta_us"] for r in pts]
        yerr = [r["delta_std_us"] for r in pts]
        all_widths.update(xs)
        ax.errorbar(xs, ys, yerr=yerr, fmt=f"-{DIST_MARKER[d]}", color=DIST_COLOR[d],
                    lw=2.4, ms=9, mfc=SURFACE, mew=2.2, capsize=4, ecolor=DIST_COLOR[d],
                    label=DIST_LABEL[d], zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    widths_sorted = sorted(all_widths)
    ax.set_xticks(widths_sorted)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    base_axes(ax, "msg_width  (payload length, doubles)",
              "Δ tick time from HELICS coupling  (µs, log scale)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axt = ax.twiny()
    axt.set_xscale("log")
    axt.set_xlim(ax.get_xlim())
    axt.set_xticks(widths_sorted)
    axt.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{8 * v:,.0f} B"))
    axt.set_xlabel("Payload size  (bytes/message, 8 B/double)", fontsize=13.5,
                   color=INK2, labelpad=8)
    axt.tick_params(axis="x", labelsize=11.5, colors=MUTED, length=0)
    for s in axt.spines.values():
        s.set_visible(False)

    leg = ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Payload width dominates once messages leave scalar size",
                 fontsize=18, color=INK, x=0.11, ha="left", y=0.945, weight="bold")
    fig.text(0.5, 0.025,
             "fanout = all2all · freq = 1 · causality = same_step · zmq · local",
             fontsize=10, color=MUTED, ha="center")
    savefig(fig, "12_payload_width.png", outdir)

    print("\nfig12 points (delta_us by msg_width, M pinned to the sweep value per series):")
    for d in dists:
        M = sweeps[d][0]["M"]
        for r in sorted(sweeps[d], key=lambda r: r["msg_width"]):
            print(f"  {d:<13} M={M} width={r['msg_width']:<5} ({8*r['msg_width']:,} B)  "
                  f"delta_us={r['delta_us']:8.2f} +/- {r['delta_std_us']:.2f}")


# ------------------------------------------------------------------ fig 13

def fig13_publish_frequency(records, outdir):
    # Pin M to whichever value carries the freq sweep (mirrors fig12's guard —
    # a different M's lone freq=1 baseline must not leak into the series).
    base = select(records, msg_width=1, causality="same_step", fanout="all2all")
    sweeps = {}
    for d in DIST_ORDER:
        M, rows = pick_sweep_M(select(base, distance=d), "freq")
        if M is not None:
            sweeps[d] = rows
    dists = [d for d in DIST_ORDER if d in sweeps]
    if not dists:
        print("[skip 13] no distance has >=2 freq points at a single M "
              "(msg_width=1, causality=same_step, fanout=all2all)")
        return
    for d in DIST_ORDER:
        if d not in dists and select(base, distance=d):
            print(f"[skip 13:{d}] no single M has >=2 freq points for distance={d}")

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.16)

    report = {}
    for d in dists:
        pts = sorted(sweeps[d], key=lambda r: r["freq"])
        xs = [r["freq"] for r in pts]
        ys = [r["delta_us"] for r in pts]
        yerr = [r["delta_std_us"] for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt=f"-{DIST_MARKER[d]}", color=DIST_COLOR[d],
                    lw=2.4, ms=9, mfc=SURFACE, mew=2.2, capsize=4, ecolor=DIST_COLOR[d],
                    label=DIST_LABEL[d], zorder=4)
        base_pt = next((r for r in pts if r["freq"] == 1), None)
        if base_pt is not None:
            for r in pts:
                pct = 100.0 * (base_pt["delta_us"] - r["delta_us"]) / base_pt["delta_us"]
                report[(d, r["freq"])] = (r["delta_us"], pct)
                if r["freq"] != 1:
                    # offset up-and-right of the marker so the label clears the
                    # marker glyph instead of sitting on top of it
                    ax.annotate(f"{pct:+.0f}%", xy=(r["freq"], r["delta_us"]),
                                xytext=(9, 12), textcoords="offset points",
                                ha="left", fontsize=10.5, color=DIST_COLOR[d])

    base_axes(ax, "freq  (publish every k-th tick)",
              "Δ tick time from HELICS coupling  (µs)")
    ax.axhline(0, color="#c3c2b7", ls=":", lw=1.2)
    leg = ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Publishing less often removes most of the coupling cost",
                 fontsize=18, color=INK, x=0.11, ha="left", y=0.955, weight="bold")
    fig.text(0.5, 0.03,
             "% labels = reduction vs. freq=1 · fanout = all2all · msg_width = 1 · "
             "causality = same_step · zmq · local",
             fontsize=10, color=MUTED, ha="center")
    savefig(fig, "13_publish_frequency.png", outdir)

    print("\nfig13 points (delta_us by freq, % reduction vs freq=1):")
    for (d, f), (val, pct) in sorted(report.items()):
        print(f"  {d:<13} freq={f:<3} delta_us={val:8.2f}  reduction={pct:+.1f}%")


# ------------------------------------------------------------------ fig 14

def fig14_causality(records, outdir):
    # Pair same_step/next_step at matching (distance, fanout, M) — a topology
    # is defined by all three; next_step in this CSV only exists at M=4, so
    # pairing must not grab a same_step row from a different M.
    base = select(records, msg_width=1, freq=1)
    pairs = []
    for d in DIST_ORDER:
        for fo in FANOUT_ORDER:
            # Pair on (N, M), not M alone -- same reason as pick_sweep_M: with N
            # swept, an M value alone does not identify a topology, and the pair
            # would compare two different scenarios' causality.
            by_same = {(r["N"], r["M"]): r for r in select(base, distance=d, fanout=fo, causality="same_step")}
            by_next = {(r["N"], r["M"]): r for r in select(base, distance=d, fanout=fo, causality="next_step")}
            for topo in sorted(set(by_same) & set(by_next)):
                pairs.append((d, fo, topo[1], by_same[topo], by_next[topo]))
    if not pairs:
        print("[skip 14] no (distance, fanout, M) topology has both same_step and next_step rows")
        return

    fig, ax = plt.subplots(figsize=(11.6, 1.6 + 1.15 * len(pairs)), dpi=300)
    fig.subplots_adjust(left=0.22, right=0.90, top=0.80, bottom=0.14)

    ys = np.arange(len(pairs))
    x_max = 0
    for i, (d, fo, M, same, nxt) in enumerate(pairs):
        ax.plot([same["delta_us"], nxt["delta_us"]], [i, i], color=MUTED, lw=1.8, zorder=1)
        ax.plot(same["delta_us"], i, "o", color=CAUS_COLOR["same_step"], ms=12,
                mfc=CAUS_COLOR["same_step"], mec=INK, mew=1.0, zorder=3,
                label="same_step" if i == 0 else None)
        ax.plot(nxt["delta_us"], i, "s", color=CAUS_COLOR["next_step"], ms=11,
                mfc=CAUS_COLOR["next_step"], mec=INK, mew=1.0, zorder=3,
                label="next_step" if i == 0 else None)
        lo, hi = min(same["delta_us"], nxt["delta_us"]), max(same["delta_us"], nxt["delta_us"])
        x_max = max(x_max, hi)
        ax.annotate(f"+{hi - lo:.0f} µs", xy=(hi, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=11, color=INK2)

    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlim(right=x_max * 1.22)  # headroom for the "+NN µs" annotation
    base_axes(ax, "Δ tick time from HELICS coupling  (µs)", "")
    # base_axes applies a numeric y-tick formatter; category labels must be
    # (re-)set after it or it clobbers them.
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{DIST_LABEL[d]}\n{fo} (M={M})" for d, fo, M, *_ in pairs], fontsize=12.5)
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.grid(axis="y", visible=False)
    leg = ax.legend(loc="lower right", fontsize=12.5, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("next_step costs more than same_step at matching topologies",
                 fontsize=16.5, color=INK, x=0.22, ha="left", y=0.94, weight="bold")
    fig.text(0.5, 0.02, "msg_width = 1 · freq = 1 · zmq · local",
             fontsize=10, color=MUTED, ha="center")
    savefig(fig, "14_causality.png", outdir)

    print("\nfig14 paired values (delta_us, same_step vs next_step):")
    for d, fo, M, same, nxt in pairs:
        print(f"  {d:<13} {fo:<8} M={M} same_step={same['delta_us']:8.2f}  "
              f"next_step={nxt['delta_us']:8.2f}  diff={nxt['delta_us']-same['delta_us']:+.2f}")


# ----------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=None,
                    help="Input CSV (default: phaseD_local_wide.csv if present and "
                         "non-empty, else phaseD_local.csv)")
    ap.add_argument("--outdir", default=FINDINGS, help="Output directory for PNGs")
    args = ap.parse_args()

    bench = args.bench or default_bench_path()
    if not os.path.exists(bench):
        raise SystemExit(f"input CSV not found: {bench}")
    print(f"using input CSV: {bench}")
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(bench)
    records = build_deltas(rows)
    if not records:
        raise SystemExit("no wired (exchange=on) rows with a matching control found — nothing to plot")

    fig10_cost_vs_load(records, args.outdir)
    fig11_fanout_shape(records, args.outdir)
    fig12_payload_width(records, args.outdir)
    fig13_publish_frequency(records, args.outdir)
    fig14_causality(records, args.outdir)


if __name__ == "__main__":
    main()
