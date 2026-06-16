#!/usr/bin/env python3
"""
validate_scenarios.py — parse-gate for every scenario YAML.

Validates each src/scenarios/*.yaml against the Pydantic schema (ScenarioConfig). Exits
non-zero if any fails. Run after schema edits and after migrating scenario files.

Usage:
    python scripts/validate_scenarios.py                 # all src/scenarios/*.yaml
    python scripts/validate_scenarios.py a.yaml b.yaml    # specific files
"""
from __future__ import annotations

import glob
import os
import sys

# make `src` importable the same way the test scripts do
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from utils.config_reader import read_scenario_config  # noqa: E402


def main(argv):
    if argv:
        files = argv
    else:
        files = sorted(glob.glob(os.path.join(_SRC, "scenarios", "*.yaml")))

    ok, fail = 0, 0
    for path in files:
        name = os.path.basename(path)
        try:
            cfg = read_scenario_config(os.path.abspath(path))
            rl = "RL" if cfg.reinforcement_learning_config is not None else "base"
            print(f"  PASS [{rl:4}] {name}")
            ok += 1
        except Exception as e:
            print(f"  FAIL        {name}")
            print(f"      {str(e).splitlines()[0]}")
            for ln in str(e).splitlines()[1:8]:
                print(f"      {ln}")
            fail += 1
    print(f"\n{ok} passed, {fail} failed, {len(files)} total")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
