"""
E2E verification for distributed SSH federate spawning (T6).

Runs the remote-spawn demo (`distributed_demo`, `pv_federate` on `host: local_box`)
and its all-local physics twin (`pv_batt_test_base`), then asserts every recorded
federate timeseries matches within tolerance. A remote federate executes the same
model code over the same HELICS/Redis wiring, so a distributed run must reproduce
the all-local numbers exactly (bit-for-bit up to float noise).

Prerequisites (see docs/user_guide/distributed_deployment.md):
  1. Passwordless key-based ssh to 127.0.0.1 (the demo's `local_box` target).
  2. cosim Redis + Mosquitto reachable at 127.0.0.1 (docker compose -f src/docker-compose.yaml up -d).
  3. `cosim_gym` conda env on the ssh PATH, or set `python:` in distributed_demo.yaml.

Usage (from project root, cosim_gym env active):
  python src/verify_distributed_demo.py            # run both scenarios, then compare
  python src/verify_distributed_demo.py --no-run   # compare the latest existing runs only

Exit code 0 = match (PASS), 1 = mismatch or missing data (FAIL).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results"

sys.path.insert(0, str(REPO_ROOT / "src"))

REMOTE_SCENARIO = "distributed_demo"
LOCAL_TWIN = "pv_batt_test_base"
TOLERANCE = 1e-9


def _latest_sim_id(scenario: str) -> str | None:
    """Newest sim_id directory under results/<scenario>/ (lexical = chronological here)."""
    scen_dir = RESULTS_PATH / scenario
    if not scen_dir.exists():
        return None
    sim_dirs = sorted([p.name for p in scen_dir.iterdir() if p.is_dir()])
    return sim_dirs[-1] if sim_dirs else None


def _records_by_key(scenario: str, sim_id: str) -> dict:
    """Map (federate, model_instance, attribute, type, mode, time) -> value for a run."""
    from dashboard.dashboard_data import load_all_records

    records = load_all_records(scenario, sim_id, results_path=RESULTS_PATH)
    keyed = {}
    for r in records:
        key = (
            r["federate"], r["model_instance"], r["attribute"],
            r["type"], r["mode"], r["time"],
        )
        keyed[key] = r["value"]
    return keyed


def compare(remote_sim: str, local_sim: str) -> bool:
    """Compare the two runs record-by-record. Returns True on match."""
    remote = _records_by_key(REMOTE_SCENARIO, remote_sim)
    local = _records_by_key(LOCAL_TWIN, local_sim)

    if not remote:
        print(f"FAIL: no records for {REMOTE_SCENARIO}/{remote_sim}")
        return False
    if not local:
        print(f"FAIL: no records for {LOCAL_TWIN}/{local_sim}")
        return False

    # Compare on the intersection of keys (federate/instance/attribute/time overlap).
    common = set(remote) & set(local)
    if not common:
        print("FAIL: no overlapping (federate, attribute, time) keys between the two runs")
        return False

    mismatches = []
    for key in sorted(common):
        rv, lv = remote[key], local[key]
        try:
            if abs(float(rv) - float(lv)) > TOLERANCE:
                mismatches.append((key, rv, lv))
        except (TypeError, ValueError):
            if rv != lv:
                mismatches.append((key, rv, lv))

    print(f"Compared {len(common)} overlapping records "
          f"(remote had {len(remote)}, local twin had {len(local)}).")

    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatched records (showing up to 10):")
        for key, rv, lv in mismatches[:10]:
            print(f"  {key}: remote={rv} local={lv}")
        return False

    print("PASS: every overlapping record matches within tolerance "
          f"({TOLERANCE}). Remote federate reproduced the all-local result.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-run", action="store_true",
        help="skip running the scenarios; compare the latest existing results only",
    )
    args = parser.parse_args()

    if not args.no_run:
        from core.ScenarioManager import main as run_scenario
        print(f"=== Running all-local twin: {LOCAL_TWIN} ===")
        run_scenario(LOCAL_TWIN)
        print(f"=== Running remote-spawn demo: {REMOTE_SCENARIO} ===")
        run_scenario(REMOTE_SCENARIO)

    remote_sim = _latest_sim_id(REMOTE_SCENARIO)
    local_sim = _latest_sim_id(LOCAL_TWIN)
    if remote_sim is None or local_sim is None:
        print(f"FAIL: missing results (remote={remote_sim}, local={local_sim}). "
              "Run without --no-run, or run both scenarios first.")
        return 1

    print(f"Comparing {REMOTE_SCENARIO}/{remote_sim} vs {LOCAL_TWIN}/{local_sim}")
    return 0 if compare(remote_sim, local_sim) else 1


if __name__ == "__main__":
    sys.exit(main())
