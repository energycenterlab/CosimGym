#!/usr/bin/env python3
"""Phase 1b figures — the INSTANCE axis (M) view of seq-vs-par.

Dual of plot_crossover_clean.py (which fixed M and swept model cost). Here model
cost is fixed and M is swept. Produces three publication panels:

  07_instance_crossover.png : seq vs par per-tick vs M, at two model-cost levels;
                              marks the MEASURED instance-crossover M* (and the
                              law prediction as a secondary annotation).
  08_speedup_vs_M.png       : seq/par speedup vs M; grows toward the ceiling W.
  09_par_staircase.png      : par per-tick vs M for W in {4,8,16}, one subplot
                              each; the ceil(M/W) step function, measured vs a
                              level-anchored prediction (heavy model so steps
                              exceed scheduling jitter).

Inputs (produced by run_bench.py; see matrices/phase1b_*.yaml):
  findings/phase1b_instance_crossover.csv   (required for 07, 08)
  findings/phase1b_staircase.csv            (required for 09)
Missing input -> that panel is skipped with a notice (degrades gracefully).
"""
import csv
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = "/media/space/rando/CODE/CosimGym/scripts/scaling_study/findings"
IC_CSV = os.path.join(HERE, "phase1b_instance_crossover.csv")
ST_CSV = os.path.join(HERE, "phase1b_staircase.csv")

# palette (dataviz reference, light print surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# heavy_compute_dummy cost model + par overhead (phase-1 self-consistent fit)
A, B, O_PAR = 2.65e-5, 1.337e-7, 0.0462


def c_of(work):
    return A + B * work


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def load(csv_path, want_mode=None):
    if not os.path.exists(csv_path):
        return None
    rows = [r for r in csv.DictReader(open(csv_path))
            if r.get("failure_mode", "") in ("", "None")]
    g = defaultdict(list)
    for r in rows:
        if want_mode and r["mode"] != want_mode:
            continue
        key = (float(r["work"]), r["mode"], r["W"] or "-", int(r["M"]))
        g[key].append(float(r["tick_mean_s"]) * 1e3)
    return g


def thin(vals, keep):
    """Return the subset of vals nearest each target in keep (for tick labels)."""
    return [v for v in vals if v in keep]


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


def instance_crossover_M(work, W, Mmax=8192):
    thr = O_PAR / c_of(work)
    for M in range(2, Mmax):
        if M - math.ceil(M / W) >= thr:
            return M
    return None


def measured_crossover(seq_pts, par_pts):
    """log-interp M where median(seq) crosses median(par)."""
    sd = dict(seq_pts); pd = dict(par_pts)
    Ms = sorted(set(sd) & set(pd))
    for i in range(len(Ms) - 1):
        a, b = Ms[i], Ms[i + 1]
        da, db = sd[a] - pd[a], sd[b] - pd[b]
        if da < 0 <= db:
            f = (0 - da) / (db - da)
            return round(10 ** (math.log10(a) + f * (math.log10(b) - math.log10(a))))
    return None


LOG_LABELS = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}  # tick labels to keep

# ---------------------------------------------------------------- panel 07 + 08
g = load(IC_CSV)
if g is None:
    print(f"[skip 07/08] missing {IC_CSV} — run the bench first")
else:
    works = sorted({k[0] for k in g})
    Wn = int(sorted({k[2] for k in g if k[2] != "-"})[0])
    Wfix = str(Wn)

    # ---- 07: seq vs par per-tick vs M, one subplot per work level ----
    fig, axes = plt.subplots(1, len(works), figsize=(7.0 * len(works), 6.2),
                             dpi=300, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.82, bottom=0.20, wspace=0.20)
    for j, work in enumerate(works):
        ax = axes[0][j]
        Mw = sorted({k[3] for k in g if k[0] == work})
        seq = [(M, st.median(g[(work, "seq", "-", M)])) for M in Mw
               if (work, "seq", "-", M) in g]
        par = [(M, st.median(g[(work, "par", Wfix, M)])) for M in Mw
               if (work, "par", Wfix, M) in g]
        sx, sy = zip(*seq); px, py = zip(*par)
        ax.plot(sx, sy, "-o", color=BLUE, lw=2.6, ms=8, mfc=SURFACE, mew=2.2,
                label="Sequential")
        ax.plot(px, py, "-o", color=ORANGE, lw=2.6, ms=8, mfc=SURFACE, mew=2.2,
                label=f"Parallel (W={Wn})")
        Mmeas = measured_crossover(seq, par)
        Mlaw = instance_crossover_M(work, Wn)
        if Mmeas and min(Mw) <= Mmeas <= max(Mw):
            ax.axvline(Mmeas, color=MUTED, ls="--", lw=1.6)
            ax.annotate(f"crossover M* ≈ {Mmeas}\n(parallel wins →)\nlaw: {Mlaw}",
                        xy=(Mmeas, max(sy) * 0.62), xytext=(Mmeas * 1.15, max(sy) * 0.62),
                        fontsize=12.5, color=INK2, va="center")
        ax.set_xscale("log")
        ax.set_xlim(min(Mw) * 0.85, max(Mw) * 1.18)
        ax.set_xticks(thin(Mw, LOG_LABELS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        base_axes(ax, "Model instances per federate  M  (≈ buildings)",
                  "Mean wall time per tick  (ms)" if j == 0 else "")
        ax.set_title(f"model cost c ≈ {c_of(work) * 1e3:.2f} ms  (work = {int(work):,})",
                     fontsize=14.5, color=INK, pad=8)
        if j == 0:
            leg = ax.legend(loc="upper left", fontsize=14, frameon=True,
                            framealpha=0.95, edgecolor=GRID, borderpad=0.7)
            leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Instance-count crossover: sequential vs. parallel vs. M",
                 fontsize=19, color=INK, x=0.08, ha="left", y=0.95, weight="bold")
    fig.text(0.5, 0.035,
             f"single federate · W = {Wn} workers · 20 ticks × 3 repeats · "
             f"heavy_compute_dummy · zmq · local",
             fontsize=10.5, color=MUTED, ha="center")
    out = os.path.join(HERE, "07_instance_crossover.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)

    # ---- 08: speedup seq/par vs M ----
    allM = sorted({k[3] for k in g})
    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.86, bottom=0.14)
    cols = {works[0]: AQUA, works[-1]: BLUE} if len(works) > 1 else {works[0]: BLUE}
    for work in works:
        pts = []
        for M in sorted({k[3] for k in g if k[0] == work}):
            ks, kp = (work, "seq", "-", M), (work, "par", Wfix, M)
            if ks in g and kp in g:
                pts.append((M, st.median(g[ks]) / st.median(g[kp])))
        if pts:
            mx, my = zip(*pts)
            ax.plot(mx, my, "-o", color=cols[work], lw=2.6, ms=8, mfc=SURFACE,
                    mew=2.2, label=f"c ≈ {c_of(work) * 1e3:.2f} ms (work {int(work):,})")
    ax.axhline(Wn, color=MUTED, ls="--", lw=1.6)
    ax.text(max(allM) * 0.97, Wn - 0.15, f"ceiling = W = {Wn}", ha="right",
            fontsize=13, color=MUTED, va="top")
    ax.axhline(1.0, color="#c3c2b7", ls=":", lw=1.4)
    ax.text(min(allM) * 1.05, 1.18, "break-even (1×)", fontsize=12, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(min(allM) * 0.85, max(allM) * 1.18)
    ax.set_xticks(thin(allM, LOG_LABELS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    base_axes(ax, "Model instances per federate  M  (≈ buildings)",
              "Speedup  (sequential ÷ parallel wall time)")
    ax.set_ylim(0, Wn * 1.16)
    leg = ax.legend(loc="upper left", fontsize=13.5, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Parallel speedup grows with M toward the ceiling W",
                 fontsize=18.5, color=INK, x=0.10, ha="left", y=0.955, weight="bold")
    out = os.path.join(HERE, "08_speedup_vs_M.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)

# ------------------------------------------------------------------- panel 09
gs = load(ST_CSV, want_mode="par")
if gs is None:
    print(f"[skip 09] missing {ST_CSV} — run the bench first")
else:
    work = max({k[0] for k in gs})  # heaviest work present (steps above noise)
    c = c_of(work)
    Ws = sorted({int(k[2]) for k in gs if k[2] != "-"})
    wcol = {4: ORANGE, 8: BLUE, 16: AQUA}
    fig, axes = plt.subplots(1, len(Ws), figsize=(5.2 * len(Ws), 5.6), dpi=300,
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.80, bottom=0.20, wspace=0.10)
    ymax = max(st.median(v) for v in gs.values()) * 1.12
    for i, W in enumerate(Ws):
        ax = axes[0][i]
        Ms = sorted({k[3] for k in gs if int(k[2]) == W})
        meas = [(M, st.median(gs[(work, "par", str(W), M)])) for M in Ms]
        mx, my = zip(*meas)
        # level-anchored predicted staircase: ceil(M/W)*c + offset, offset chosen
        # so predicted mean matches measured mean (isolates STEP SHAPE from the
        # known additive-model absolute-calibration offset).
        pred_raw = [math.ceil(M / W) * c * 1e3 for M in Ms]
        offset = st.mean(my) - st.mean(pred_raw)
        pm = list(range(min(Ms), max(Ms) + 1))
        pp = [math.ceil(M / W) * c * 1e3 + offset for M in pm]
        col = wcol.get(W, MUTED)
        ax.step(pm, pp, where="post", color=col, ls="--", lw=1.8, alpha=0.75,
                label="predicted ⌈M/W⌉·c")
        ax.plot(mx, my, "-o", color=col, lw=2.4, ms=8, mfc=SURFACE, mew=2.2,
                label="measured")
        # mark the W-multiples where a step occurs
        for k in range(1, max(Ms) // W + 2):
            xm = k * W
            if min(Ms) <= xm <= max(Ms):
                ax.axvline(xm, color=GRID, lw=1.0, zorder=0)
        base_axes(ax, "Instances  M", "Parallel wall time / tick (ms)" if i == 0 else "")
        ax.set_title(f"W = {W} workers", fontsize=15, color=INK, pad=6)
        ax.set_xticks(Ms)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.set_ylim(min(my) * 0.9, ymax)
        if i == 0:
            leg = ax.legend(loc="upper left", fontsize=12.5, frameon=True,
                            framealpha=0.95, edgecolor=GRID, borderpad=0.6)
            leg.get_frame().set_facecolor(SURFACE)
    fig.suptitle("Parallel cost is a staircase in M  (one step of size c per worker refill)",
                 fontsize=18, color=INK, x=0.075, ha="left", y=0.94, weight="bold")
    fig.text(0.5, 0.035,
             f"single federate · model cost c ≈ {c * 1e3:.1f} ms (work {int(work):,}) · "
             f"20 ticks × 3 repeats · heavy_compute_dummy · zmq · local · "
             f"vertical lines = W-multiples (step points)",
             fontsize=10, color=MUTED, ha="center")
    out = os.path.join(HERE, "09_par_staircase.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)

print("\nPredicted instance-crossover M* (law): "
      + ", ".join(f"work {int(w):,}: M*={instance_crossover_M(w, 8)}"
                  for w in (2000, 25000)))
