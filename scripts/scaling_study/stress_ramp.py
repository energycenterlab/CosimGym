#!/usr/bin/env python3
"""Stress driver: ramp one axis until CosimGym actually breaks, then stop (D6).

`run_bench.py` executes a FIXED matrix and keeps going after a failure. That is
wrong for a stress test on a shared machine: once a configuration falls over,
every larger cell in the matrix is both uninformative (we already know the
ceiling is below it) and hostile to co-users (it is the biggest, heaviest run in
the file). This driver instead climbs a ladder one rung at a time, re-checks the
machine's headroom before every rung, and stops at the FIRST hard failure --
recording *what* broke, which is the actual deliverable (plan Sec.3 Phase F:
"push until first failure; record the failure mode").

It reuses `run_bench.bench_one` verbatim, so a stress row and a matrix row are
the same schema (CONTRACT.md D2) and land in the same CSV shape.

Guards (both checked immediately before each rung, per plan Sec.8.7's "check
`uptime`/`free -g` immediately before scaling up"):
  --guard-free-pct : abort if free RAM falls below this share of total (default
                     40, the budget named in the plan).
  --guard-load     : abort if the 1-minute load average exceeds this (default
                     = core count, i.e. the box is already fully committed).
A guard abort is NOT a framework failure and is recorded as such -- conflating
"we ran out of permission to keep going" with "CosimGym broke" is exactly how
Phase 4 produced a confounded result.

Usage:
    python scripts/scaling_study/stress_ramp.py --axis M --start 64 --factor 2 \\
        --steps 6 --N 4 --exchange on --out results/scaling/stress_M.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bench import CSV_FIELDS, bench_one  # noqa: E402  (path set above)

REPO_ROOT = Path(__file__).resolve().parents[2]


def machine_headroom():
    """(free_pct, load1) sampled fresh -- never cached, the point is the value NOW."""
    total = free = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                free = float(line.split()[1])
            if total and free:
                break
    load1 = os.getloadavg()[0]
    return (100.0 * free / total if total else 0.0), load1


def check_guards(args):
    """None if clear to proceed, else a human-readable abort reason."""
    free_pct, load1 = machine_headroom()
    if free_pct < args.guard_free_pct:
        return (f"guard_ram: only {free_pct:.1f}% RAM free, below the "
                f"{args.guard_free_pct:.0f}% budget")
    if load1 > args.guard_load:
        return (f"guard_load: 1-min load {load1:.1f} exceeds {args.guard_load:.1f}")
    return None


def ladder(args):
    """The rung values for the chosen axis: geometric growth from --start."""
    vals, v = [], args.start
    for _ in range(args.steps):
        vals.append(int(v))
        v *= args.factor
    return vals


def cell_for(args, value):
    """One matrix cell (run_bench knob dict) with `value` on the ramped axis."""
    cell = {
        "F": args.F, "N": args.N, "M": args.M,
        "mode": "seq", "core_type": args.core_type,
        "model": args.model, "placement": args.placement,
        "ticks": args.ticks,
    }
    if args.axis == "M":
        cell["M"] = value
    elif args.axis == "N":
        cell["N"] = value
    if args.exchange == "on":
        cell.update({
            "exchange": "on", "distance": args.distance, "fanout": args.fanout,
            "msg_width": args.msg_width, "freq": args.freq,
            "causality": "same_step",
        })
    else:
        cell["exchange"] = "none"
    return cell


def main(argv=None):
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    scratch = Path(args.scratch_dir or (REPO_ROOT / "results" / "scaling" / "_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)

    rungs = ladder(args)
    print(f"stress ramp: axis={args.axis} rungs={rungs} exchange={args.exchange} "
          f"(stop at first failure; guards: free>={args.guard_free_pct}% "
          f"load<={args.guard_load})")

    results = []
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for value in rungs:
            reason = check_guards(args)
            if reason:
                print(f"  ABORT before {args.axis}={value}: {reason}")
                results.append((value, reason, None))
                break

            free_pct, load1 = machine_headroom()
            print(f"  {args.axis}={value:<6} (free {free_pct:.0f}%, load {load1:.1f}) ... ",
                  end="", flush=True)
            t0 = time.time()
            row = bench_one(cell_for(args, value), repeat=0, scratch_dir=scratch,
                            timeout=args.timeout, keep_scratch=False)
            writer.writerow(row)
            fh.flush()

            fail = row.get("failure_mode")
            tick = row.get("tick_mean_s")
            wall = time.time() - t0
            if fail:
                print(f"FAILED [{fail}] after {wall:.0f}s")
                results.append((value, fail, row))
                break
            print(f"ok  tick_mean={float(tick) * 1e6:.0f}us  "
                  f"rss={row.get('peak_rss_mb')}MB  ({wall:.0f}s)")
            results.append((value, None, row))

    print("\n=== ladder summary ===")
    last_ok = None
    for value, fail, row in results:
        if fail:
            print(f"  {args.axis}={value}: STOP -- {fail}")
        else:
            last_ok = value
    if last_ok is not None:
        print(f"  max stable {args.axis} reached: {last_ok} "
              f"(F={args.F}, N={args.N}, M={args.M}, exchange={args.exchange})")
    print(f"  rows appended to {out_path}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=("M", "N"), required=True,
                    help="which knob to ramp (M = instances/federate, N = federates/federation)")
    ap.add_argument("--start", type=float, required=True, help="first rung value")
    ap.add_argument("--factor", type=float, default=2.0, help="geometric growth per rung")
    ap.add_argument("--steps", type=int, default=6, help="max rungs to attempt")
    ap.add_argument("--F", type=int, default=2)
    ap.add_argument("--N", type=int, default=4, help="held fixed unless --axis N")
    ap.add_argument("--M", type=int, default=4, help="held fixed unless --axis M")
    ap.add_argument("--core-type", dest="core_type", default="zmq")
    ap.add_argument("--model", default="exchange_dummy")
    ap.add_argument("--placement", default="local")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--exchange", choices=("none", "on"), default="on")
    ap.add_argument("--distance", default="cross_fed")
    ap.add_argument("--fanout", default="all2all")
    ap.add_argument("--msg-width", dest="msg_width", type=int, default=1)
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-run wall-clock timeout (s); a timeout counts as a failure")
    ap.add_argument("--out", required=True, help="CSV to append rows to")
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--guard-free-pct", dest="guard_free_pct", type=float, default=40.0,
                    help="abort if free RAM drops below this %% of total (plan Sec.8.7 budget)")
    ap.add_argument("--guard-load", dest="guard_load", type=float,
                    default=float(os.cpu_count() or 8),
                    help="abort if 1-min load average exceeds this (default: core count)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
