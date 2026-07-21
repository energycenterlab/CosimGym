#!/usr/bin/env python

#TODO : make this safer and more solid ready to be expanded every time a new feature is addedd, add more combination of features 
#TODO: i do not want to test only  single features by themselves but also the combination of different features
"""Pre-merge regression suite for CosimGym.

Runs the unit tests plus one fast, dependency-light scenario per feature axis,
each in an isolated subprocess (so HELICS/Redis/broker state never bleeds between
runs), and reports a PASS/FAIL table. Exit code is non-zero if anything failed —
wire it into your merge checklist:

    conda run -n cosim_gym python tests/regression_suite.py

Prereqs (same as any run): docker services up (redis, mosquitto, minio) and, for
the two distributed scenarios, passwordless ssh to 127.0.0.1 (localhost-as-remote).
Run from the project root.

Add a scenario here whenever you add a feature — that is the whole point: the suite
is the living contract that every pre-existing feature still works after a change.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_SCENARIO_TIMEOUT = 300  # seconds

# feature axis -> representative fast scenario. Keep each one SHORT (smoke, not a
# real run) so the whole suite stays runnable on every merge.
SCENARIOS = [
    ("BASE co-sim",            "simple_test"),
    ("MULTIFED (hierarchy)",   "simple_test_multifederations"),
    ("DIST remote-ssh",        "distributed_demo"),
    ("DIST + MULTIFED",        "distributed_multifederation_test"),
    ("PARALLEL model exec",    "benchmark_parallel_par"),
    ("PARQUET sink",           "rc_building_parquet_test"),
    ("INTERFACE (dig-twin)",   "m0_interface_smoke_test"),
    ("STREAM (mqtt mirror)",   "m1_stream_smoke_test"),
    ("INTERFACE override",     "m4_interface_override_smoke_test"),
    ("BK4 sim-to-real",        "m5_bk4_demo_a_full_sim"),
    ("RL online train",        "smoke_rl_dqn"),
    # EnergyPlus FMUs require the run duration to be a whole multiple of 86400s
    # (1 day), so there is no sub-day "smoke" variant — bui0_fmu_test is already
    # the 1-day minimum. FMU instantiation dominates the runtime here.
    ("FMU",                    "bui0_fmu_test"),
]

RUN_ONE = (
    "import sys; sys.path.insert(0, 'src'); "
    "from core.ScenarioManager import main; main('{name}')"
)


def _run_once(name):
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, "-c", RUN_ONE.format(name=name)],
            cwd=ROOT, capture_output=True, text=True, timeout=PER_SCENARIO_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, PER_SCENARIO_TIMEOUT, "TIMEOUT", ""
    dt = time.time() - t0
    out = p.stdout + p.stderr
    if "Scenario execution did NOT complete successfully" in out:
        return False, dt, "reported failure", out
    if "Scenario execution completed successfully" in out:
        return True, dt, "", out
    return False, dt, f"no success marker (exit {p.returncode})", out


def run_scenario(name):
    """Run one scenario in a fresh interpreter. PASS iff it prints the success line
    and never prints the failure line.

    Retries ONCE on a transient broker port-bind clash ("Address already in use"):
    a broker from the previous scenario can still hold its port in TIME_WAIT when the
    next one hardcodes the same port. A short wait lets it clear. This is a suite-level
    safeguard only — it does not mask a genuinely broken scenario, which fails on retry too."""
    ok, dt, info, out = _run_once(name)
    if not ok and "Address already in use" in out:
        time.sleep(5)
        ok2, dt2, info2, _ = _run_once(name)
        return ok2, dt + dt2, (info2 if not ok2 else "(passed on retry)")
    return ok, dt, info


def run_pytest():
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "--ignore=tests/regression_suite.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    dt = time.time() - t0
    ok = p.returncode == 0
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return ok, dt, tail


def main():
    print("=" * 72)
    print("CosimGym pre-merge regression suite")
    print("=" * 72)
    results = []

    print("\n[unit] pytest tests/ ...", flush=True)
    ok, dt, info = run_pytest()
    results.append(("UNIT pytest", "tests/", ok, dt, info))
    print(f"  {'PASS' if ok else 'FAIL'}  ({dt:.1f}s)  {info}")

    for feature, name in SCENARIOS:
        print(f"\n[{feature}] {name} ...", flush=True)
        ok, dt, info = run_scenario(name)
        results.append((feature, name, ok, dt, info))
        print(f"  {'PASS' if ok else 'FAIL'}  ({dt:.1f}s)  {info}")

    print("\n" + "=" * 72)
    print(f"{'FEATURE':24s} {'SCENARIO':34s} {'RESULT':6s} TIME")
    print("-" * 72)
    n_fail = 0
    for feature, name, ok, dt, info in results:
        n_fail += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        line = f"{feature:24s} {name:34s} {mark:6s} {dt:5.1f}s"
        if not ok and info:
            line += f"  <- {info}"
        print(line)
    print("=" * 72)
    total = len(results)
    print(f"{total - n_fail}/{total} passed"
          + (f"  ({n_fail} FAILED)" if n_fail else "  — all green"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
