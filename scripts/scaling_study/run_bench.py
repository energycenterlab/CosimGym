#!/usr/bin/env python3
"""Bench driver for the CosimGym scaling study (D2).

Sweeps a matrix of `gen_scenario.py` (D1) knob-cells x repeats, runs each
generated scenario as an isolated subprocess with `COSIM_PERF_LOG=1` (D3),
samples the process tree's RSS/CPU while it runs, reads back `perf.json`,
and appends one row per (cell, repeat) to a CSV -- incrementally, so a
mid-sweep crash still leaves every already-completed row on disk.

CLI (CONTRACT.md D2):
    python scripts/scaling_study/run_bench.py --matrix <matrix.yaml> \\
        [--repeats 3] [--out results/scaling/bench.csv]

Subprocess-launch mechanism is copied from `tests/regression_suite.py`
(`run_scenario`): `subprocess.Popen([sys.executable, "-c", code], ...,
start_new_session=True)` running `core.ScenarioManager.main(<path>)`, killed
via `os.killpg(os.getpgid(pid), SIGKILL)` on timeout. The one difference is
the scenario path: `read_scenario_config` (src/utils/config_reader.py)
accepts an absolute path directly, so generated scenarios live in a scratch
dir outside `src/scenarios/` rather than being copied in.

Matrix file format (see scripts/scaling_study/matrices/smoke.yaml):
    runs: [ {F: 1, N: 2, ...}, ... ]        # explicit list of knob-dicts, OR
    axes: {F: [1], N: [2, 4], ...}          # cartesian product of axis values
Knob keys are exactly the `gen_scenario.py` CLI flags minus the leading
`--` and with `-` -> `_` (F, N, M, mode, W, core_type, model, work,
placement, ticks, machine_b_host, machine_b_user, machine_b_workdir,
machine_b_python, manager_address). Omitted knobs fall back to
`gen_scenario.py` CLI defaults.

sim_id resolution: `ScenarioManager._write_perf_log` writes to
`results/<scenario_name>/<sim_id>/perf.json` where `<sim_id>` is a fresh
timestamp per run and `<scenario_name>` is deterministic from the knobs
(gen_scenario.py's `name:` field) -- so it repeats across `--repeats` and
even across cells that only differ in a knob not in the name (e.g. `work`).
Since runs are executed strictly one-at-a-time and harvested+cleaned before
the next one starts, "the newest subdir under results/<scenario_name>/ after
this run" is unambiguous -- see `_newest_subdir`.
"""
import argparse
import csv
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

try:
    import psutil
except ImportError:  # pragma: no cover - environment-dependent
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_SCRIPT = Path(__file__).resolve().parent / "gen_scenario.py"
RESULTS_ROOT = REPO_ROOT / "results"
LOGS_ROOT = REPO_ROOT / "logs"

# matrix knob key -> gen_scenario.py CLI flag (CONTRACT.md D1/D2 knob table).
KNOB_TO_FLAG = {
    "F": "--F", "N": "--N", "M": "--M", "mode": "--mode", "W": "--W",
    "core_type": "--core-type", "model": "--model", "work": "--work",
    "placement": "--placement", "ticks": "--ticks",
    "machine_b_host": "--machine-b-host", "machine_b_user": "--machine-b-user",
    "machine_b_workdir": "--machine-b-workdir", "machine_b_python": "--machine-b-python",
    "manager_address": "--manager-address",
}

# CONTRACT.md D2 row schema: all knobs + repeat + perf.json fields + derived metrics.
# perf.json's own "n_ticks" (ticks actually executed by the gating federate) is kept
# as "perf_n_ticks" to avoid colliding with the knob column "n_ticks" (ticks
# *configured*) -- a deliberate, documented deviation from a literal dict-merge:
# both numbers matter (configured vs. actually-reached, which differ on a
# partial/failed run) and CONTRACT.md's two schemas happen to reuse the same key.
CSV_FIELDS = [
    "F", "N", "M", "mode", "W", "core_type", "model", "work", "placement",
    "n_machines", "n_ticks", "repeat",
    "scenario_name", "sim_id",
    "setup_s", "broker_setup_s", "federate_spawn_s", "sim_wall_s",
    "perf_n_ticks", "tick_mean_s", "tick_median_s", "tick_p95_s",
    "failure_mode", "peak_rss_mb", "cpu_util_pct", "throughput_inst_steps_s",
]

RUN_CODE = ("import sys; sys.path.insert(0, 'src'); "
            "from core.ScenarioManager import main; main({path!r})")

_HELICS_ERRCODE_RE = re.compile(r"helics[a-z_]*error[^\n]{0,40}?(-\d+)", re.I)
_ERR_LINE_RE = re.compile(r"error[^\n]{0,120}", re.I)


def expand_matrix(spec):
    """`runs:` list -> as-is. `axes:` mapping -> cartesian product of knob-dicts."""
    if "runs" in spec:
        return list(spec["runs"])
    if "axes" in spec:
        axes = spec["axes"]
        keys = list(axes.keys())
        return [dict(zip(keys, values))
                for values in itertools.product(*(axes[k] for k in keys))]
    raise ValueError("matrix file must have a top-level 'runs:' list or 'axes:' mapping")


def gen_scenario(cell, out_path):
    """Call gen_scenario.py (D1) for one matrix cell; return (spec_dict, scenario_name)."""
    cmd = [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)]
    for k, v in cell.items():
        if v is None:
            continue
        flag = KNOB_TO_FLAG.get(k)
        if flag is None:
            raise ValueError(f"unknown matrix knob {k!r} (not a gen_scenario.py flag)")
        cmd += [flag, str(v)]
    pr = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if pr.returncode != 0:
        raise RuntimeError(f"gen_scenario.py failed (rc={pr.returncode}): "
                            f"{pr.stdout}\n{pr.stderr}"[:500])
    spec = json.loads(Path(str(out_path) + ".spec.json").read_text())
    scenario_name = yaml.safe_load(out_path.read_text())["name"]
    return spec, scenario_name


def _newest_subdir(path):
    if not path.is_dir():
        return None
    subs = [p for p in path.iterdir() if p.is_dir()]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def _classify_failure(stdout_text):
    """Best-effort failure_mode string when perf.json has none (crashed before/
    without writing one). HELICS error codes look like '...helics_error...-101',
    matching the CONTRACT.md example 'lost_comms_-101'."""
    m = _HELICS_ERRCODE_RE.search(stdout_text)
    if m:
        return f"lost_comms_{m.group(1)}"
    m = _ERR_LINE_RE.search(stdout_text)
    if m:
        return m.group(0).strip()[:120]
    return "crash"


def _poll_tree(root_pid, tracked, cpu_samples, rss_samples):
    """One psutil sample of the process tree rooted at root_pid. `tracked` is a
    pid -> psutil.Process cache across calls (needed because cpu_percent(None)
    measures since-the-*previous*-call-on-that-object, so a pid must be primed
    once before its number means anything)."""
    try:
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    cpu_total = 0.0
    rss_total = 0
    got_reading = False
    live_pids = set()
    for p in procs:
        live_pids.add(p.pid)
        if p.pid not in tracked:
            try:
                p.cpu_percent(None)  # prime; first read is meaningless, skip it
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            tracked[p.pid] = p
            continue
        try:
            cpu_total += tracked[p.pid].cpu_percent(None)
            rss_total += tracked[p.pid].memory_info().rss
            got_reading = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            tracked.pop(p.pid, None)
    for pid in list(tracked):
        if pid not in live_pids:
            tracked.pop(pid, None)
    if got_reading:
        cpu_samples.append(cpu_total)
        rss_samples.append(rss_total / (1024 * 1024))


def run_once(scenario_path, timeout, poll_interval=0.3):
    """Run one generated scenario in an isolated subprocess (regression_suite's
    launch mechanism: Popen(['-c', code], start_new_session=True), SIGKILL the
    whole process group on timeout), sampling RSS/CPU at ~1/poll_interval Hz."""
    env = os.environ.copy()
    env["COSIM_PERF_LOG"] = "1"
    code = RUN_CODE.format(path=str(scenario_path))
    pr = subprocess.Popen([sys.executable, "-c", code], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=REPO_ROOT, start_new_session=True, env=env, bufsize=1)

    out_chunks = []

    def _drain():
        try:
            for line in pr.stdout:
                out_chunks.append(line)
        except Exception:
            pass

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    tracked, cpu_samples, rss_samples = {}, [], []
    t0 = time.time()
    timed_out = False
    while True:
        if pr.poll() is not None:
            break
        if time.time() - t0 > timeout:
            timed_out = True
            break
        if psutil is not None:
            _poll_tree(pr.pid, tracked, cpu_samples, rss_samples)
        time.sleep(poll_interval)

    if timed_out:
        try:
            os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            pr.wait(timeout=5)
        except Exception:
            pass

    reader.join(timeout=5)
    return {
        "timed_out": timed_out,
        "returncode": pr.returncode,
        "stdout": "".join(out_chunks),
        "wall_s": time.time() - t0,
        "cpu_samples": cpu_samples,
        "rss_samples": rss_samples,
    }


def _empty_row():
    return {k: None for k in CSV_FIELDS}


def bench_one(cell, repeat, scratch_dir, timeout, keep_scratch):
    """Generate + run one (cell, repeat); always returns a filled-in CSV row dict,
    even on gen_scenario/timeout/crash failure."""
    row = _empty_row()
    row["repeat"] = repeat
    tag = uuid.uuid4().hex[:8]
    scen_path = scratch_dir / f"bench_{tag}.yaml"

    try:
        spec_knobs, scenario_name = gen_scenario(cell, scen_path)
    except Exception as e:
        row["failure_mode"] = f"gen_scenario_error: {e}"[:200]
        return row
    row.update(spec_knobs)
    row["scenario_name"] = scenario_name

    result = run_once(scen_path, timeout=timeout)

    results_scen_dir = RESULTS_ROOT / scenario_name
    sim_dir = _newest_subdir(results_scen_dir)
    perf = {}
    if sim_dir is not None:
        perf_path = sim_dir / "perf.json"
        if perf_path.exists():
            try:
                perf = json.loads(perf_path.read_text())
            except Exception:
                perf = {}
        row["sim_id"] = perf.get("sim_id") or sim_dir.name

    row["setup_s"] = perf.get("setup_s")
    row["broker_setup_s"] = perf.get("broker_setup_s")
    row["federate_spawn_s"] = perf.get("federate_spawn_s")
    row["sim_wall_s"] = perf.get("sim_wall_s")
    row["perf_n_ticks"] = perf.get("n_ticks")
    row["tick_mean_s"] = perf.get("tick_mean_s")
    row["tick_median_s"] = perf.get("tick_median_s")
    row["tick_p95_s"] = perf.get("tick_p95_s")

    if result["timed_out"]:
        row["failure_mode"] = "timeout"
    elif perf.get("failure_mode"):
        row["failure_mode"] = perf["failure_mode"]
    elif "completed successfully" not in result["stdout"].lower():
        row["failure_mode"] = _classify_failure(result["stdout"])
    else:
        row["failure_mode"] = None

    row["peak_rss_mb"] = max(result["rss_samples"]) if result["rss_samples"] else None
    row["cpu_util_pct"] = (sum(result["cpu_samples"]) / len(result["cpu_samples"])
                           if result["cpu_samples"] else None)

    sim_wall = perf.get("sim_wall_s")
    try:
        if sim_wall and sim_wall > 0:
            row["throughput_inst_steps_s"] = (
                row["F"] * row["N"] * row["M"] * row["n_ticks"] / sim_wall)
        else:
            row["throughput_inst_steps_s"] = None
    except Exception:
        row["throughput_inst_steps_s"] = None

    if not keep_scratch:
        scen_path.unlink(missing_ok=True)
        Path(str(scen_path) + ".spec.json").unlink(missing_ok=True)
        if sim_dir is not None:
            shutil.rmtree(sim_dir, ignore_errors=True)
            try:
                if results_scen_dir.is_dir() and not any(results_scen_dir.iterdir()):
                    results_scen_dir.rmdir()
            except OSError:
                pass
            # logs/<scenario_name>/<sim_id>/ shares the same leaf name as
            # results/<scenario_name>/<sim_id>/ (both derive from the same
            # run_timestamp -- see utils/logging_config.py), so the same sim_id
            # cleans up the matching log tree.
            log_dir = LOGS_ROOT / scenario_name / sim_dir.name
            shutil.rmtree(log_dir, ignore_errors=True)
            log_scen_dir = LOGS_ROOT / scenario_name
            try:
                if log_scen_dir.is_dir() and not any(log_scen_dir.iterdir()):
                    log_scen_dir.rmdir()
            except OSError:
                pass

    return row


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, help="matrix YAML (runs: list or axes: mapping)")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per matrix cell")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "scaling" / "bench.csv"),
                    help="output CSV (appended to; header written once)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-run wall-clock timeout in seconds (default 180)")
    ap.add_argument("--scratch-dir", default=None,
                    help="dir for generated scenario YAMLs (default: results/scaling/_scratch)")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep generated scenario/spec files and results/logs dirs (debugging)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if psutil is None:
        print("WARNING: psutil not importable in this interpreter -- peak_rss_mb/"
              "cpu_util_pct will be left empty for every row. Install it in the "
              "cosim_gym env: `conda run -n cosim_gym pip install psutil`.",
              file=sys.stderr)

    spec = yaml.safe_load(Path(args.matrix).read_text())
    cells = expand_matrix(spec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = (Path(args.scratch_dir) if args.scratch_dir
                   else REPO_ROOT / "results" / "scaling" / "_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    f = out_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        f.flush()

    total = len(cells) * args.repeats
    done = 0
    try:
        for cell in cells:
            for repeat in range(args.repeats):
                done += 1
                print(f"[{done}/{total}] cell={cell} repeat={repeat}", flush=True)
                row = bench_one(cell, repeat, scratch_dir, args.timeout, args.keep_scratch)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                mark = "FAIL" if row["failure_mode"] else "ok"
                print(f"    -> {mark}  sim_wall_s={row['sim_wall_s']}  "
                      f"failure_mode={row['failure_mode']}", flush=True)
    finally:
        f.close()
        if not args.keep_scratch:
            try:
                if scratch_dir.is_dir() and not any(scratch_dir.iterdir()):
                    scratch_dir.rmdir()
            except OSError:
                pass

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
