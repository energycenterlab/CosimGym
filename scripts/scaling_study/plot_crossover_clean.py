#!/usr/bin/env python3
"""Publication-clean seq-vs-par crossover figure.

Single controlled setup (F=1, N=1, M=16, W=8, zmq, local, 20 ticks, 3 reps),
heavy_compute_dummy, per-instance compute cost swept via the `work` parameter.
Source: findings/bench_all.csv, filtered to the M=16/W=8 slice only (no pooling
across M — that was the flaw in the previous 01_crossover.png).
"""
import csv
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BENCH = "/media/space/rando/CODE/CosimGym/scripts/scaling_study/findings/bench_all.csv"
OUT = "/media/space/rando/CODE/CosimGym/scripts/scaling_study/findings/01_crossover.png"

# palette (dataviz reference, light print surface)
BLUE = "#2a78d6"   # slot 1 -> sequential
ORANGE = "#eb6834"  # slot 2 -> parallel
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# heavy_compute_dummy cost model (phase-1 self-consistent fit): c = a + b*work (s)
A, B = 2.65e-5, 1.337e-7
O_PAR = 0.0462  # phase-1 local self-consistent O_par (s/tick)
M, W = 16, 8

# ---- load clean slice -> per (mode, work) median/min/max of tick_mean_s ----
rows = [r for r in csv.DictReader(open(BENCH))
        if r["model"] == "heavy_compute_dummy" and r["M"] == str(M)
        and r["core_type"] == "zmq" and r["placement"] == "local"]
g = defaultdict(list)
for r in rows:
    g[(r["mode"], float(r["work"]))].append(float(r["tick_mean_s"]))

works = sorted({w for (m, w) in g if ("seq", w) in g and ("par", w) in g})  # shared grid


def series(mode):
    med = [st.median(g[(mode, w)]) * 1e3 for w in works]   # -> ms
    lo = [min(g[(mode, w)]) * 1e3 for w in works]
    hi = [max(g[(mode, w)]) * 1e3 for w in works]
    return med, lo, hi


seq_med, seq_lo, seq_hi = series("seq")
par_med, par_lo, par_hi = series("par")

# measured crossover: zero of (seq-par) on log-work between bracketing samples
diff = [(s - p) for s, p in zip(seq_med, par_med)]
xover = None
import math
for i in range(len(works) - 1):
    if diff[i] < 0 <= diff[i + 1]:
        f = (0 - diff[i]) / (diff[i + 1] - diff[i])
        lx = math.log10(works[i]) + f * (math.log10(works[i + 1]) - math.log10(works[i]))
        xover = 10 ** lx
        break
# law prediction: (M-ceil(M/W))*c* = O_par  ->  work*
c_star = O_PAR / (M - math.ceil(M / W))
work_law = (c_star - A) / B

# ---- plot ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})
fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=300)
fig.subplots_adjust(left=0.115, right=0.965, top=0.80, bottom=0.185)

# region shading
ax.axvspan(works[0] * 0.85, xover, color=BLUE, alpha=0.045, zorder=0)
ax.axvspan(xover, works[-1] * 1.18, color=ORANGE, alpha=0.05, zorder=0)

# min-max bands
ax.fill_between(works, seq_lo, seq_hi, color=BLUE, alpha=0.15, lw=0, zorder=1)
ax.fill_between(works, par_lo, par_hi, color=ORANGE, alpha=0.15, lw=0, zorder=1)

# lines + markers
ax.plot(works, seq_med, "-o", color=BLUE, lw=2.6, ms=9, mfc=SURFACE,
        mew=2.2, label="Sequential", zorder=4)
ax.plot(works, par_med, "-o", color=ORANGE, lw=2.6, ms=9, mfc=SURFACE,
        mew=2.2, label="Parallel  (W = 8 workers)", zorder=4)

# crossover marker
ax.axvline(xover, color=MUTED, ls="--", lw=1.6, zorder=2)
ax.annotate(
    f"crossover  ≈ {round(xover / 500) * 500:,.0f} work\n"
    f"(c ≈ {(A + B * xover) * 1e3:.1f} ms / instance)",
    xy=(xover, 150), xytext=(xover * 1.06, 150),
    fontsize=13.5, color=INK2, ha="left", va="center",
)

ax.set_xscale("log")
ax.set_xlim(works[0] * 0.85, works[-1] * 1.18)
ax.set_ylim(0, 232)

# x ticks: plain integers with thousands separators, no sci-notation
ax.set_xticks(works)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_yticks(range(0, 231, 40))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

ax.set_xlabel("Per-instance model work  (busy-loop iterations / step)",
              fontsize=16, color=INK, labelpad=9)
ax.set_ylabel("Mean wall time per tick  (ms)", fontsize=16, color=INK, labelpad=9)
ax.tick_params(axis="both", labelsize=13.5, colors=INK2, length=4)

# grid + spines
ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c3c2b7")

# twin top axis: per-instance step cost c (ms) = (a + b*work)*1000
axt = ax.twiny()
axt.set_xscale("log")
axt.set_xlim(ax.get_xlim())
axt.set_xticks(works)
axt.xaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{(A + B * v) * 1e3:.1f}"))
axt.set_xlabel("Per-instance step cost  c  (ms)", fontsize=15, color=INK2, labelpad=8)
axt.tick_params(axis="x", labelsize=12.5, colors=MUTED, length=0)
for s in axt.spines.values():
    s.set_visible(False)

leg = ax.legend(loc="upper left", fontsize=14.5, frameon=True, framealpha=0.95,
                edgecolor=GRID, borderpad=0.8, handlelength=1.8)
leg.get_frame().set_facecolor(SURFACE)

# region captions (placed low, clear of legend + crossover text)
ax.text(works[0] * 1.05, 20, "sequential faster", fontsize=13,
        color=BLUE, style="italic", va="center")
ax.text(xover * 1.35, 20, "parallel faster", fontsize=13, color=ORANGE,
        style="italic", va="center")

fig.suptitle("Sequential vs. parallel model-instance execution",
             fontsize=19.5, color=INK, x=0.115, ha="left", y=0.955, weight="bold")
fig.text(0.5, 0.028,
         f"single federate · M = {M} instances · W = {W} workers · "
         f"20 ticks × 3 repeats · heavy_compute_dummy · zmq · local",
         fontsize=10.5, color=MUTED, ha="center")

fig.savefig(OUT)
print("wrote", OUT)
print(f"measured crossover (log-interp) = {xover:,.0f} work  "
      f"(c = {(A + B * xover) * 1e3:.2f} ms)")
print(f"law-predicted crossover        = {work_law:,.0f} work  "
      f"(c* = {c_star * 1e3:.2f} ms)")
