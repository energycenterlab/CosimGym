"""Generate per-seed copies of the RL case-study scenarios.

The framework takes ONE seed per scenario (`reinforcement_learning_config.seed`),
so an N-seed comparison = N scenario files. This writes
`<base>_s<seed>.yaml` for each seed, changing ONLY:
  - `name`                                        (so results/logs land in their own dir)
  - `reinforcement_learning_config.seed`
  - `experiment.checkpoint.best`                  (else seeds clobber each other's checkpoint)

Everything else is copied verbatim, so the seed sweep is a pure seed diff.

Usage:
    python make_seed_variants.py                       # default set, seeds 42 43 44
    python make_seed_variants.py --seeds 42 43 44 --scenarios cs_s2_sac cs_s2_dqn
    python make_seed_variants.py --clean               # delete generated variants
"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "src/scenarios"

DEFAULT_SCENARIOS = [
    "cs_s2_sac", "cs_s2_dqn",
    "cs_s2_reset_full", "cs_s2_reset_rolling", "cs_s2_reset_none",
    "cs_s3_fmu",
]
DEFAULT_SEEDS = [42, 43, 44]


def variant_name(base: str, seed: int) -> str:
    return f"{base}_s{seed}"


def make(base: str, seed: int) -> Path | None:
    src = SCEN / f"{base}.yaml"
    if not src.exists():
        print(f"  SKIP {base}: {src} not found")
        return None
    d = yaml.safe_load(src.read_text())
    name = variant_name(base, seed)
    d["name"] = name
    rl = d.get("reinforcement_learning_config")
    if rl is None:
        print(f"  SKIP {base}: no reinforcement_learning_config")
        return None
    rl["seed"] = seed
    exp = rl.setdefault("experiment", {}) or {}
    rl["experiment"] = exp
    ckpt = exp.setdefault("checkpoint", {}) or {}
    exp["checkpoint"] = ckpt
    ckpt["best"] = f"best_{name}.pth"
    out = SCEN / f"{name}.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    if a.clean:
        n = 0
        for base in a.scenarios:
            for seed in a.seeds:
                p = SCEN / f"{variant_name(base, seed)}.yaml"
                if p.exists():
                    p.unlink(); n += 1
        print(f"removed {n} seed-variant files")
        return

    made = []
    for base in a.scenarios:
        for seed in a.seeds:
            p = make(base, seed)
            if p:
                made.append(p.name)
                print(f"  wrote {p.name}")
    print(f"\n{len(made)} seed variants written to src/scenarios/")


if __name__ == "__main__":
    main()
