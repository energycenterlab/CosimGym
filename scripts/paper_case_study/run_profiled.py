"""Profiled scenario runner: wall-clock + peak RSS per scenario, N repetitions.

Runs a scenario by name in an isolated subprocess (same launch pattern as
tests/regression_suite.py), samples peak RSS of the WHOLE process tree (parent
broker/manager + federate children) via psutil, and reads the authoritative
simulation timing from the run's logs/<name>/<sim_id>/execution_metrics.json.

Appends one row per scenario to results/paper_case_study/exec_metrics.csv:
    scenario, reps, wall_median_s, wall_min_s, wall_max_s,
    sim_median_s, peak_rss_mb_median, git_commit, marker

Usage:
    python run_profiled.py <scenario> [<scenario> ...] --reps 3
    python run_profiled.py --list-from S4A          # convenience group
"""
from __future__ import annotations
import argparse, json, os, signal, statistics, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_case_study"
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "exec_metrics.csv"

RUN_ONE = ("import sys; sys.path.insert(0, 'src'); "
           "from core.ScenarioManager import main; main('{name}')")

try:
    import psutil
except ImportError:
    psutil = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _peak_rss_sampler(pid, stop_evt, out):
    """Sample max RSS (MB) of the process tree until stop_evt is set."""
    if psutil is None:
        return
    peak = 0
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    while not stop_evt.is_set():
        try:
            procs = [parent] + parent.children(recursive=True)
            rss = sum(p.memory_info().rss for p in procs if p.is_running())
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.2)
    out["peak_rss_mb"] = peak / (1024 * 1024)


def _sim_duration(name) -> float | None:
    """Latest logs/<name>/<sim_id>/execution_metrics.json -> simulation_duration."""
    base = ROOT / "logs" / name
    if not base.exists():
        return None
    subs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not subs:
        return None
    em = subs[-1] / "execution_metrics.json"
    if not em.exists():
        return None
    try:
        d = json.loads(em.read_text())
        return d.get("phase_durations", {}).get("simulation_duration") or d.get("total_duration")
    except Exception:
        return None


def run_once(name, timeout=3600):
    code = RUN_ONE.format(name=name)
    stop = threading.Event()
    out = {"peak_rss_mb": float("nan")}
    t0 = time.time()
    pr = subprocess.Popen([sys.executable, "-c", code], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=ROOT, start_new_session=True)
    sampler = threading.Thread(target=_peak_rss_sampler, args=(pr.pid, stop, out), daemon=True)
    sampler.start()
    try:
        stdout, _ = pr.communicate(timeout=timeout)
        ok = "completed successfully" in stdout
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        pr.communicate()
        stdout, ok = f"TIMEOUT >{timeout}s", False
    finally:
        stop.set()
        sampler.join(timeout=2)
    wall = time.time() - t0
    sim = _sim_duration(name)
    tail = "" if ok else " | " + stdout.strip().splitlines()[-1][:120] if stdout.strip() else ""
    return ok, wall, sim, out["peak_rss_mb"], tail


def profile(name, reps=3, timeout=3600):
    walls, sims, rss = [], [], []
    marker = "PASS"
    for i in range(reps):
        ok, wall, sim, peak, tail = run_once(name, timeout)
        if not ok:  # one retry for a transient broker/port clash
            time.sleep(4)
            ok, wall, sim, peak, tail = run_once(name, timeout)
        print(f"  [{name}] rep {i+1}/{reps}: ok={ok} wall={wall:.2f}s sim={sim} rss={peak:.0f}MB{tail}")
        if not ok:
            marker = "FAIL" + tail
            walls.append(wall)
            break
        walls.append(wall); rss.append(peak)
        if sim is not None:
            sims.append(sim)
    med = statistics.median
    wall_med = f"{med(walls):.3f}" if walls else "NA"
    wall_min = f"{min(walls):.3f}" if walls else "NA"
    wall_max = f"{max(walls):.3f}" if walls else "NA"
    sim_med = f"{med(sims):.3f}" if sims else "NA"
    rss_med = f"{med(rss):.1f}" if rss else "NA"
    return (f"{name},{reps},{wall_med},{wall_min},{wall_max},"
            f"{sim_med},{rss_med},{_git_commit()},\"{marker}\"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if not CSV.exists():
        CSV.write_text("scenario,reps,wall_median_s,wall_min_s,wall_max_s,"
                       "sim_median_s,peak_rss_mb_median,git_commit,marker\n")
    for name in a.scenarios:
        print(f"[profile] {name} x{a.reps}")
        row = profile(name, a.reps, a.timeout)
        with CSV.open("a") as f:
            f.write(row)
        print(f"  -> {row.strip()}")


if __name__ == "__main__":
    main()
