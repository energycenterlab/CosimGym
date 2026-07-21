#!/usr/bin/env python
"""Pre-merge regression suite for CosimGym.

Runs `pytest` plus a broad set of scenarios — one or more per feature axis, and
explicit feature COMBINATIONS — each in an isolated subprocess (so HELICS / Redis /
broker state never bleeds between runs). Long scenarios are auto-shortened on a
throwaway temp copy (the real YAML is never touched), so the whole suite stays fast
enough to run on every merge.

    conda run -n cosim_gym python tests/regression_suite.py

Prereqs (same as any run): docker services up (redis, mosquitto, minio) and, for the
distributed scenarios, passwordless ssh to 127.0.0.1 (localhost-as-remote). Run from
the project root.

Three lists below:
  SCENARIOS      - single-feature coverage, expected to PASS.
  COMBOS         - feature intersections, expected to PASS.
  KNOWN_FAIL     - scenarios blocked by a *tracked framework/env bug* (not a scenario
                   bug). Run anyway: expected to fail, reported as xfail. If one starts
                   PASSING it is flagged UNEXPECTED-PASS (time to drop it from here).
  CLOUD_OPTIONAL - distributed scenarios that need the real remote machines; only run
                   when RUN_CLOUD=1 (they pass only if those hosts are reachable).

Add a scenario here whenever you add a feature or a feature combination — this suite is
the living contract that every pre-existing capability still works after a change.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCEN_DIR = ROOT / "src" / "scenarios"
PER_SCENARIO_TIMEOUT = 240  # seconds before we call a scenario SLOW

# --- single-feature coverage (expected PASS) ---------------------------------
SCENARIOS = [
    ("BASE co-sim",            "simple_test"),
    ("BASE + csv models",      "rc_building_test_base"),
    ("BASE + building csv",    "bui_hp_test_base"),
    ("BASE + pv/battery",      "pv_batt_test_base"),
    ("BASE scale",             "benchmark_scale_local"),
    ("MULTIFED (hierarchy)",   "simple_test_multifederations"),
    ("DIST remote-ssh",        "distributed_demo"),
    ("PARALLEL model exec",    "benchmark_parallel_par"),
    ("SEQUENTIAL (par twin)",  "benchmark_parallel_seq"),
    ("INTERFACE (dig-twin)",   "m0_interface_smoke_test"),
    ("INTERFACE outbound",     "m2_interface_outbound_smoke_test"),
    ("INTERFACE inbound",      "m3_interface_inbound_smoke_test"),
    ("INTERFACE override",     "m4_interface_override_smoke_test"),
    ("STREAM (mqtt mirror)",   "m1_stream_smoke_test"),
    ("STREAM + pandapipes",    "dh_district_jan_base"),
    ("BK4 sim-to-real (a)",    "m5_bk4_demo_a_full_sim"),
    ("BK4 digital-twin (b)",   "m5_bk4_demo_b_digital_twin"),
    ("GRID pandapower",        "pandapower_grid_test_base"),
    ("GRID pandapipes",        "pandapipes_grid_test_base"),
    ("STRESS json sink",       "stress_multi_building_json"),
    ("RL DQN (custom)",        "simple_DQN_test"),
    ("RL SAC (sb3)",           "simple_SACsb3_test"),
    ("RL PPO (rllib)",         "simple_rllib_test"),
    ("RL DQN smoke",           "smoke_rl_dqn"),
    ("RL building DQN",        "bui_hp_DQN"),
    ("RL building SAC",        "bui_hp_SAC"),
    ("RL rolling-reset DQN",   "bui_hp_DQN_rollingreset"),
    ("RL rolling-reset SAC",   "bui_hp_SAC_rollingreset"),
    ("FMU (EnergyPlus)",       "bui0_fmu_test"),
    ("FMU + RL DQN",           "bui0_setpoint_DQN"),
    ("FMU + RL SAC",           "bui0_setpoint_SAC"),
    ("FMU + RL heating DQN",   "bui0_heatingpower_DQN"),
]

# --- feature COMBINATIONS (expected PASS) ------------------------------------
COMBOS = [
    ("DIST + MULTIFED",        "distributed_multifederation_test"),
    ("DIST + PARALLEL",        "combo_dist_parallel"),
    ("MULTIFED + PARALLEL",    "combo_multifed_parallel"),
    ("PARALLEL + GRID",        "multi_building_grid_test"),
]

# --- tracked framework/env bugs: run, expect FAIL (xfail) --------------------
# Fixing these is a *framework code* change, out of scope for a scenario-only pass.
KNOWN_FAIL = {
    "rc_building_parquet_test":               "parquet sink -> native libstdc++ SIGSEGV (json twin passes)",
    "m2_interface_outbound_smoke_test_parquet": "parquet sink -> native libstdc++ SIGSEGV (json twin passes)",
    "fmu_feedthrough_test":                   "parquet sink -> native libstdc++ SIGSEGV (json twin passes)",
    "stress_multi_building_parquet":          "parquet sink -> native libstdc++ SIGSEGV (json twin passes)",
    "pv_batt_DQN":                            "zmq auto-port alloc ignores port+1 -> 2-broker RL collision",
    "pv_batt_SAC":                            "zmq auto-port alloc ignores port+1 -> 2-broker RL collision",
    "simple_test_rlagent":                    "RL_Simple_Agent catalog model is a non-functional skeleton (env_step missing)",
    "Adelaide_test":                          "missing MinIO FMU object PCMA_1_0_control_2022.fmu (data/infra)",
}

# --- need the real remote machines; only with RUN_CLOUD=1 --------------------
CLOUD_OPTIONAL = [
    ("DIST cloud multi-machine", "distributed_demo_multi"),
    ("DIST cloud scale",         "benchmark_scale_distributed"),
    ("DIST cloud scale debug",   "benchmark_scale_distributed_debug"),
]

RUN_ONE = ("import sys; sys.path.insert(0, 'src'); "
           "from core.ScenarioManager import main; main('{name}')")


def _shorten(d, name):
    """Trim a scenario for a fast functional check, on a copy. Returns the dict."""
    is_fmu = "fmu" in name.lower() or "fmu" in yaml.dump(d).lower()
    # broker ports -> auto-assign (isolates concurrent/back-to-back runs)
    for fed in (d.get("federations") or {}).values():
        (fed.get("broker_config") or {}).pop("port", None)
    # span > 6h -> 1h, but NOT for FMU (EnergyPlus needs whole-day multiples)
    try:
        st = datetime.fromisoformat(str(d.get("start_time")))
        en = datetime.fromisoformat(str(d.get("end_time")))
        if (en - st) > timedelta(hours=6) and not is_fmu:
            d["end_time"] = (st + timedelta(hours=1)).isoformat()
    except Exception:
        pass
    # RL: few short episodes. episode_length is only capped for NON-FMU scenarios —
    # an FMU-RL run needs real_period*episode_length to stay a whole-day multiple.
    rl = d.get("reinforcement_learning_config")
    if rl:
        run = rl.get("run") or {}
        for phase in ("train", "eval", "test"):
            blk = run.get(phase)
            if isinstance(blk, dict):
                if "episodes" in blk:
                    blk["episodes"] = 2 if phase == "train" else 1
                if not is_fmu and isinstance(blk.get("episode_length"), int) and blk["episode_length"] > 20:
                    blk["episode_length"] = 20
        exp = rl.setdefault("experiment", {}) or {}
        rl["experiment"] = exp
        ckpt = exp.setdefault("checkpoint", {}) or {}
        exp["checkpoint"] = ckpt
        # FORCE a throwaway checkpoint name (not setdefault): otherwise a scenario that
        # names a real checkpoint (e.g. best_dqn_model.pth) would have that file clobbered
        # by this 2-episode smoke run. Also prevents save_model(None) crashing.
        ckpt["best"] = f"best_regr_{name}.pth"
    return d


def _classify(out):
    low = out.lower()
    if "completed successfully" in out:
        return True, ""
    if "address already in use" in low:
        return False, "PORT"
    m = None
    import re
    mm = re.search(r"error[^\n]{0,90}", out, re.I)
    if mm:
        m = mm.group(0).strip()
    return False, (m or "no success marker")


def run_scenario(name):
    """Shorten on a temp copy, run in an isolated process group, classify. Retries
    once on a transient port clash."""
    src = SCEN_DIR / f"{name}.yaml"
    if not src.exists():
        return False, 0.0, "file-not-found"
    try:
        d = yaml.safe_load(src.read_text())
    except Exception as e:
        return False, 0.0, f"PARSEERR {str(e)[:60]}"
    tmp_name = f"_regr_{name}"
    d["name"] = tmp_name
    _shorten(d, name)
    tmp = SCEN_DIR / f"{tmp_name}.yaml"
    tmp.write_text(yaml.safe_dump(d, sort_keys=False))
    code = RUN_ONE.format(name=tmp_name)

    def _once():
        pr = subprocess.Popen([sys.executable, "-c", code], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd=ROOT, start_new_session=True)
        try:
            out, _ = pr.communicate(timeout=PER_SCENARIO_TIMEOUT)
            return _classify(out)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            pr.communicate()
            return False, f"SLOW >{PER_SCENARIO_TIMEOUT}s"

    t0 = time.time()
    try:
        ok, info = _once()
        if not ok and info == "PORT":
            time.sleep(4)
            ok, info = _once()
            info = "" if ok else (info if info != "PORT" else "port clash x2")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
        for base in ("results", "logs"):
            shutil.rmtree(ROOT / base / tmp_name, ignore_errors=True)
        # throwaway RL checkpoint this run may have written (see _shorten)
        (ROOT / "src/models/model_catalog/RL_agents/checkpoints" / f"best_regr_{name}.pth").unlink(missing_ok=True)
    return ok, time.time() - t0, info


def run_pytest():
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                        "--ignore=tests/regression_suite.py"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return p.returncode == 0, time.time() - t0, tail


def main():
    print("=" * 78)
    print("CosimGym pre-merge regression suite")
    print("=" * 78)
    rows = []  # (section, feature, name, ok, dt, info, expect_fail)

    print("\n[unit] pytest tests/ ...", flush=True)
    ok, dt, info = run_pytest()
    rows.append(("UNIT", "pytest", "tests/", ok, dt, info, False))
    print(f"  {'PASS' if ok else 'FAIL'}  ({dt:.1f}s)  {info}")

    plan = ([("SINGLE", f, n, False) for f, n in SCENARIOS]
            + [("COMBO", f, n, False) for f, n in COMBOS]
            + [("XFAIL", "known-bug", n, True) for n in KNOWN_FAIL])
    if os.getenv("RUN_CLOUD") == "1":
        plan += [("CLOUD", f, n, False) for f, n in CLOUD_OPTIONAL]

    for section, feature, name, expect_fail in plan:
        label = feature if section not in ("XFAIL",) else KNOWN_FAIL[name][:46]
        print(f"\n[{section}] {name} ...", flush=True)
        ok, dt, info = run_scenario(name)
        rows.append((section, label, name, ok, dt, info, expect_fail))
        if expect_fail:
            mark = "xfail" if not ok else "UNEXPECTED-PASS"
        else:
            mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  ({dt:.1f}s)  {info}")

    print("\n" + "=" * 78)
    print(f"{'SECTION':8s} {'NAME':40s} {'RESULT':16s} TIME")
    print("-" * 78)
    hard_fail = 0
    unexpected_pass = 0
    for section, feature, name, ok, dt, info, expect_fail in rows:
        if expect_fail:
            result = "xfail" if not ok else "UNEXPECTED-PASS"
            if ok:
                unexpected_pass += 1
        else:
            result = "PASS" if ok else "FAIL"
            if not ok:
                hard_fail += 1
        line = f"{section:8s} {name:40s} {result:16s} {dt:5.1f}s"
        if (not ok and not expect_fail) or (ok and expect_fail):
            line += f"  <- {info or KNOWN_FAIL.get(name, '')}"
        print(line)
    print("=" * 78)
    total = len(rows)
    xfails = sum(1 for r in rows if r[6])
    passed = sum(1 for r in rows if r[3] and not r[6])
    print(f"{passed} passed, {hard_fail} FAILED, {xfails - unexpected_pass} xfail (known bugs)"
          + (f", {unexpected_pass} UNEXPECTED-PASS (drop from KNOWN_FAIL)" if unexpected_pass else ""))
    # Green gate = no hard failures. Known bugs (xfail) don't break it; an unexpected
    # pass doesn't either but is worth acting on.
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
