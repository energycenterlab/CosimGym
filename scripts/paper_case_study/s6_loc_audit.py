"""s6_loc_audit: engineering-effort audit (no simulation).

For each stage S1->S5, count YAML LOC of the stage's scenario file, the diff LOC
vs the previous stage's file (added+changed lines), and new hand-written Python
LOC for that stage (feeder/reward — analysis & plotting scripts EXCLUDED).

LOC rule: blank lines and pure-comment lines (whitespace then '#') excluded.
Diff LOC = added+modified lines reported by `git diff --no-index` (a/b), counting
'+' lines only (added or changed content on the new side).
"""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "src/scenarios"
OUT = ROOT / "results/paper_case_study"
OUT.mkdir(parents=True, exist_ok=True)

# (stage, capability, scenario file, previous file for diff, [new python files])
STAGES = [
    ("S1", "PID baseline co-sim",        "cs_s1_baseline.yaml", None, []),
    ("S2", "RL control (SAC) swap",      "cs_s2_sac.yaml",      "cs_s1_baseline.yaml", []),
    ("S3", "FMU formalism swap",         "cs_s3_fmu.yaml",      "cs_s2_sac.yaml", []),
    ("S4", "parallel scaling",           "cs_s4_vert_par_N20.yaml", "cs_s1_baseline.yaml", []),
    ("S5", "digital-twin interface",     "cs_s5_dt.yaml",       "cs_s1_baseline.yaml",
     ["scripts/paper_case_study/s5_external_feeder.py"]),
]


def yaml_loc(path: Path) -> int:
    if path is None or not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def py_loc(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def diff_loc(prev: Path, cur: Path) -> int:
    if prev is None or cur is None or not prev.exists() or not cur.exists():
        return 0
    r = subprocess.run(["git", "diff", "--no-index", "--numstat", str(prev), str(cur)],
                       cwd=ROOT, capture_output=True, text=True)
    # numstat: "added\tremoved\tpath"; use added as diff-LOC proxy
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            return int(parts[0])
    return 0


def main():
    rows = [("stage", "capability", "yaml_loc", "diff_loc_vs_prev", "new_python_loc")]
    for stage, cap, scen, prev, pys in STAGES:
        sp = (SCEN / scen) if scen else None
        pp = (SCEN / prev) if prev else None
        yl = yaml_loc(sp) if sp else 0
        dl = diff_loc(pp, sp) if (sp and pp) else 0
        pl = sum(py_loc(ROOT / p) for p in pys)
        note = " (DEFERRED)" if scen is None else ""
        rows.append((stage, cap + note, str(yl), str(dl), str(pl)))

    # framework-side proxy for the "hand-built HELICS->Gym wrapper" comparison
    rlf = py_loc(ROOT / "src/core/RL_Federate.py")
    schema = py_loc(ROOT / "src/utils/config_dataclasses.py")

    (OUT / "tab_s6_loc.csv").write_text(
        "\n".join(",".join(r) for r in rows) + "\n"
        f"# framework RL wrapper proxy (author to interpret): "
        f"RL_Federate.py={rlf} LOC, config_dataclasses.py={schema} LOC\n")
    md = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    md.append("")
    md.append(f"_Framework RL-wrapper proxy (author interprets vs a hand-built "
              f"HELICS→Gym wrapper): `RL_Federate.py` = {rlf} LOC, "
              f"`config_dataclasses.py` = {schema} LOC._")
    (OUT / "tab_s6_loc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
