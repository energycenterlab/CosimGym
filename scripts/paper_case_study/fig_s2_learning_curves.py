"""fig_s2_learning_curves + tab_s2_sample_eff.

Curve: episode return vs training step, one curve per reset mode
(full / rolling / none), mean with shaded +/-1 std across seeds.
Source: the RL agent's TRAIN storage `episode_rewards` / `episode_lengths`
(`results/<scenario>/<sim>/rl_federation/rl_agent_train_rl_storage.json`).

Sample efficiency: steps to first reach a threshold = FRAC (default 0.9) of the
best final return achieved across modes. "Best final" = max over modes of the
mean of each mode's last `TAIL` episodes. Returns are negative (penalty rewards),
so "90% of best" is taken on the shifted scale: thr = best - FRAC_GAP*(best-worst)
is avoided; instead we use the plain rule documented in the output.

Usage: python fig_s2_learning_curves.py [--seeds 42 43 44] [--frac 0.9]
"""
from __future__ import annotations
import argparse, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

OUT = Path("results/paper_case_study")
OUT.mkdir(parents=True, exist_ok=True)
MODES = {"full": "cs_s2_reset_full", "rolling": "cs_s2_reset_rolling", "none": "cs_s2_reset_none"}
TAIL = 5  # episodes averaged to define "final" performance


def _collect(names):
    runs = []
    for n in names:
        try:
            rew, lens = M.rl_episode_rewards(n, "train")
        except Exception:
            continue
        if not rew:
            continue
        steps, acc = [], 0
        for i, r in enumerate(rew):
            acc += lens[i] if i < len(lens) else 0
            steps.append(acc)
        runs.append((steps, rew))
    return runs


def load_mode(base, seeds):
    """Return list of (steps[], returns[]) — one per seed run found. Prefer seed
    variants; fall back to the un-suffixed run only if no seed variant exists."""
    runs = _collect([f"{base}_s{s}" for s in seeds])
    return runs or _collect([base])


def mean_std(runs):
    """Align seeds by episode index; return (steps, mean, std)."""
    if not runs:
        return [], [], []
    n = min(len(r[1]) for r in runs)
    steps = runs[0][0][:n]
    mean, std = [], []
    for i in range(n):
        vals = [r[1][i] for r in runs]
        mean.append(statistics.mean(vals))
        std.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return steps, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--frac", type=float, default=0.9)
    a = ap.parse_args()

    data, missing = {}, []
    for mode, base in MODES.items():
        runs = load_mode(base, a.seeds)
        if not runs:
            missing.append(mode)
            continue
        data[mode] = (mean_std(runs), len(runs))

    if not data:
        print("No S2 reset-mode training results found yet — run the S2 sweep first. "
              f"(missing: {', '.join(missing)})")
        return

    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {"full": "tab:blue", "rolling": "tab:orange", "none": "tab:green"}
    finals = {}
    for mode, ((steps, mean, std), nseeds) in data.items():
        ax.plot(steps, mean, lw=1.4, color=colors[mode], label=f"reset={mode} (n={nseeds})")
        ax.fill_between(steps, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], color=colors[mode], alpha=0.18)
        finals[mode] = statistics.mean(mean[-TAIL:]) if len(mean) >= TAIL else mean[-1]
    ax.set_xlabel("training step")
    ax.set_ylabel("episode return")
    ax.set_title("S2B — learning curves by reset strategy (mean ± 1 std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_s2_learning_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- sample efficiency ---------------------------------------------------
    best = max(finals.values())
    worst = min(finals.values())
    # returns are negative penalties: threshold = best minus frac-gap of the spread.
    thr = best - (1.0 - a.frac) * abs(best - worst) if best != worst else best
    rows = []
    for mode, ((steps, mean, _), nseeds) in data.items():
        hit = next((steps[i] for i, v in enumerate(mean) if v >= thr), None)
        rows.append((mode, f"{finals[mode]:.3f}", str(hit) if hit is not None else "not reached", str(nseeds)))
    for mode in missing:
        rows.append((mode, "MISSING", "MISSING", "0"))

    hdr = ("reset_mode", "final_return(mean last %d ep)" % TAIL, "steps_to_threshold", "n_seeds")
    (OUT / "tab_s2_sample_eff.csv").write_text(
        ",".join(hdr) + "\n" + "\n".join(",".join(f'"{c}"' for c in r) for r in rows) + "\n")
    md = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md.append("")
    md.append(f"_Threshold = {thr:.3f} — defined as best final return ({best:.3f}) minus "
              f"{(1-a.frac):.0%} of the spread across modes (returns are negative penalties). "
              f"'Final' = mean of last {TAIL} episodes._")
    (OUT / "tab_s2_sample_eff.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nfig_s2_learning_curves written. modes found: {list(data)} missing: {missing}")


if __name__ == "__main__":
    main()
