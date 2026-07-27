#!/usr/bin/env python3
"""Parametric scenario generator for the CosimGym scaling study (D1).

One CLI, one spec -> a valid CosimGym scenario YAML for ANY point in the
3-axis sweep (F federations x N federates x M model instances/federate),
sequential vs parallel model execution, HELICS core_type, and placement
(local / distributed behind NAT / distributed direct). Generalizes
`src/scenarios/generate_scale_benchmark.py` (single fixed building+PV
topology, local-vs-3-machine-NAT only) across all three axes described in
`docs/future_and_TODOs/scaling_study_plan.md` and pinned in
`scripts/scaling_study/CONTRACT.md` (D1 section) so the bench driver (D2)
and cost-model fitter (D4) can rely on an exact, stable knob vocabulary.

Topology (deliberately simple — see CONTRACT.md's explicit guidance to pick
the "simplest correct" multi-federation wiring):
    Each federation is SELF-CONTAINED. It has N federates, each stepping M
    instances of `--model`. Federates have NO subscriptions, in this or any
    other federation — publishes are derived from the catalog's declared
    `outputs` for the chosen model, so parsing/registration stays valid for
    ANY catalog model without hand-wiring per-model inputs. There is
    therefore no cross-federate/cross-federation key to get wrong, no
    causality cycle, and no need for the causality-loop-break trick used in
    generate_scale_benchmark.py's pid->hp->bldg loop — F federations are
    just N-federate blocks replicated F times. `heavy_compute_dummy`
    (the primary model for this study; see scaling_study_plan.md Phase 0/1)
    has no required inputs at all, so this is a correct, general default.

Placement:
    - local              : no `deployment:` block, no `host:` on any federate.
    - distributed_nat     : forces zmq_ss (NAT-safe, single-socket) + the
                            3-machine NAT deployment block from
                            generate_scale_benchmark.py (manager 130.192.177.14
                            ipazia/112c, machine_a 130.192.238.9/32c,
                            machine_b 130.192.238.13/32c). Federates placed
                            core-proportionally across the flattened F*N list,
                            exactly like generate_scale_benchmark.py's
                            placement() (weather stays local there because
                            it's a single shared federate; here there is no
                            shared federate, so proportional placement runs
                            over the full federate list).
    - distributed_direct  : 2-machine deployment block (manager + machine_b),
                            machine_b's connection params come from
                            --machine-b-host/--machine-b-user/--machine-b-workdir
                            /--machine-b-python (defaults are visible TBD
                            placeholders, per CONTRACT.md D1 -- Config B in
                            scaling_study_plan.md Section 5 is explicitly
                            "TBD, provide host/user/workdir"). core_type is
                            NOT forced to *_ss (no NAT to cross); whatever
                            --core-type was passed is used as-is. Federates
                            split evenly manager/machine_b (no core-count
                            data is available for an arbitrary machine_b).

Usage:
    python scripts/scaling_study/gen_scenario.py \\
        --F 1 --N 2 --M 2 --mode seq --core-type zmq \\
        --model heavy_compute_dummy --work 1 --placement local \\
        --ticks 10 --out /path/to/scenario.yaml

Also writes `<out>.spec.json` = the exact canonical knob dict (CONTRACT.md
D1 table: F, N, M, mode, W, core_type, model, work, placement, n_machines,
n_ticks) so D2/D4 can round-trip a generated scenario back to its knobs
without re-parsing YAML.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "src" / "models" / "model_catalog" / "catalog.yaml"

# Core counts measured on the 3-machine NAT rig (see generate_scale_benchmark.py
# and scaling_study_plan.md Section 5) -- placement weights derive from these.
NAT_CORES = {"manager": 112, "machine_a": 32, "machine_b": 32}

NAT_DEPLOYMENT = {
    "manager_address": "130.192.177.14",
    "machines": {
        "machine_a": {
            "host": "130.192.238.9",
            "user": "eclabuser",
            "ssh_port": 22,
            "workdir": "/home/eclabuser/CosimGym",
            "conda_env": "cosim_gym",
            "python": "/home/eclabuser/miniconda3/envs/cosim_gym/bin/python",
        },
        "machine_b": {
            "host": "130.192.238.13",
            "user": "eclabuser",
            "ssh_port": 22,
            "workdir": "/home/eclabuser/CosimGym",
            "conda_env": "cosim_gym",
            "python": "/home/eclabuser/miniconda3/envs/cosim_gym/bin/python",
        },
    },
}

# python type (as declared in catalog.yaml parameters/outputs) -> HELICS wire type
_TYPE_MAP = {
    "float": "double",
    "double": "double",
    "int": "int",
    "bool": "boolean",
    "boolean": "boolean",
    "str": "string",
    "string": "string",
}


def helics_type(py_type: str) -> str:
    return _TYPE_MAP.get(str(py_type).lower(), "double")


def load_catalog_outputs(model_name: str):
    """Best-effort: (key, helics_type, unit) list for a model's declared catalog outputs.

    Falls back to a single generic 'result' double publication if the model
    isn't found or declares no outputs, so the generator degrades gracefully
    for any catalog entry rather than hand-wiring per-model knowledge.
    """
    try:
        catalog = yaml.safe_load(CATALOG_PATH.read_text())
        outputs = catalog["models"][model_name].get("outputs") or {}
    except Exception:
        outputs = {}
    if not outputs:
        return [("result", "double", "-")]
    return [(k, helics_type(v.get("type", "float")), v.get("unit", "-"))
            for k, v in outputs.items()]


def flatten_placement(n_total: int, distributed: bool):
    """index (0-based, across ALL federations flattened) -> machine alias or None (local)."""
    if not distributed:
        return {i: None for i in range(n_total)}
    total_cores = sum(NAT_CORES.values())
    n_a = round(n_total * NAT_CORES["machine_a"] / total_cores)
    n_b = round(n_total * NAT_CORES["machine_b"] / total_cores)
    out = {}
    for i in range(n_total):
        if i < n_a:
            out[i] = "machine_a"
        elif i < n_a + n_b:
            out[i] = "machine_b"
        else:
            out[i] = None  # manager
    return out


def direct_placement(n_total: int):
    """index -> None (manager) or 'machine_b', split evenly (no core-count data available)."""
    half = n_total // 2
    return {i: ("machine_b" if i < half else None) for i in range(n_total)}


def build_federate(fed_idx: int, alias, args, outputs):
    """One compute federate: M instances of --model, self-contained (no subscriptions)."""
    fed_name = f"compute_{fed_idx:03d}"
    core_type = args.core_type

    instantiation = {
        "model_name": args.model,
        "n_instances": args.M,
        "prefix": f"m{fed_idx:03d}",
        "parallel_execution": args.mode == "par",
    }
    if args.mode == "par" and args.W is not None:
        instantiation["max_parallel_workers"] = args.W

    parameters = {}
    if args.model == "heavy_compute_dummy" and args.work is not None:
        parameters["iterations"] = args.work

    federate = {
        "name": fed_name,
        "type": "base",
        "core_type": core_type,
        "log_level": "ERROR",
        "timing_configs": {"real_period": 1},
        "flags": {"terminate_on_error": True},
        "connections": {
            "endpoints": [],
            "subscribes": [],
            "publishes": [
                {"key": k, "type": t, "units": u} for (k, t, u) in outputs
            ],
        },
        "model_configs": {
            "instantiation": instantiation,
            "parameters": parameters,
            "init_state": {},
        },
        "memory_config": {"sink": "none", "attrs": "all"},
    }
    if alias:
        federate["host"] = alias
    return fed_name, federate


SYNC_BLOCK = {
    "auto_offset": {
        "enabled": True,
        "offset_step": 0.1,
        "override_existing_offsets": False,
    },
    "default_subscription_causality": "same_step",
    "validate_causality_cycles": True,
}


def build_scenario(args):
    outputs = load_catalog_outputs(args.model)
    n_total = args.F * args.N
    distributed = args.placement != "local"

    if args.placement == "distributed_nat":
        place = flatten_placement(n_total, distributed=True)
    elif args.placement == "distributed_direct":
        place = direct_placement(n_total)
    else:
        place = flatten_placement(n_total, distributed=False)

    start = datetime(2024, 1, 1)
    end = start + timedelta(seconds=args.ticks)  # real_period=1s/tick

    scenario = {
        "version": "1.0.0",
        "name": f"scaling_F{args.F}_N{args.N}_M{args.M}_{args.mode}_{args.core_type}_{args.placement}",
        "scenario_description": (
            f"D1-generated scaling-study scenario: F={args.F} federations, "
            f"N={args.N} federates/federation, M={args.M} instances/federate, "
            f"mode={args.mode}, W={args.W}, core_type={args.core_type}, "
            f"model={args.model}, work={args.work}, placement={args.placement}, "
            f"ticks={args.ticks}. Generated by scripts/scaling_study/gen_scenario.py "
            f"-- do not hand-edit; regenerate from the CLI args instead."
        ),
        "start_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "log_level": "ERROR",
        "memory_config": {"sink": "none", "attrs": "all"},
        "synchronization": SYNC_BLOCK,
    }

    if args.placement == "distributed_nat":
        scenario["deployment"] = NAT_DEPLOYMENT
    elif args.placement == "distributed_direct":
        scenario["deployment"] = {
            "manager_address": args.manager_address,
            "machines": {
                "machine_b": {
                    "host": args.machine_b_host,
                    "user": args.machine_b_user,
                    "ssh_port": 22,
                    "workdir": args.machine_b_workdir,
                    "conda_env": "cosim_gym",
                    **({"python": args.machine_b_python} if args.machine_b_python else {}),
                }
            },
        }

    federations = {}
    idx = 0
    for f in range(1, args.F + 1):
        fed_name = f"federation_{f}"
        federate_configs = {}
        for n in range(args.N):
            name, fed_cfg = build_federate(idx, place.get(idx), args, outputs)
            federate_configs[name] = fed_cfg
            idx += 1
        federations[fed_name] = {
            "broker_config": {
                "core_type": args.core_type,
                # Stride by 10, not 1: ScenarioManager._broker_ports() reserves BOTH
                # `port` and `port + 1` for plain zmq (the paired reply socket), so a
                # 1-apart stride made federation_f's `port+1` collide with
                # federation_(f+1)'s `port` for F >= 2 zmq scenarios ("port already in
                # use" bind failure). zmq_ss is unaffected (no port+1 reservation) but
                # the same stride is used for every core_type for simplicity.
                "port": 23500 + 10 * f,
                "federates": args.N,
            },
            "federate_configs": federate_configs,
        }
    scenario["federations"] = federations
    return scenario


def spec_dict(args):
    """Canonical knob dict — names/types exactly as CONTRACT.md D1 table."""
    n_machines = {"local": 1, "distributed_nat": 3, "distributed_direct": 2}[args.placement]
    return {
        "F": args.F,
        "N": args.N,
        "M": args.M,
        "mode": args.mode,
        "W": args.W if args.mode == "par" else None,
        "core_type": args.core_type,
        "model": args.model,
        "work": args.work,
        "placement": args.placement,
        "n_machines": n_machines,
        "n_ticks": args.ticks,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--F", type=int, default=1, help="number of federations")
    ap.add_argument("--N", type=int, default=1, help="federates per federation")
    ap.add_argument("--M", type=int, default=1, help="model instances per federate")
    ap.add_argument("--mode", choices=("seq", "par"), default="seq",
                    help="seq -> sequential instance stepping; par -> parallel_execution: true")
    ap.add_argument("--W", type=int, default=None,
                    help="max_parallel_workers (only used when --mode par; omitted -> runtime default)")
    ap.add_argument("--core-type", dest="core_type", choices=("zmq", "tcp", "zmq_ss", "tcp_ss"),
                    default="zmq", help="HELICS core_type applied to broker(s) and federates")
    ap.add_argument("--model", default="heavy_compute_dummy", help="catalog model_name")
    ap.add_argument("--work", type=float, default=None,
                    help="heavy_compute_dummy work/cost param (iterations); ignored for other models")
    ap.add_argument("--placement", choices=("local", "distributed_nat", "distributed_direct"),
                    default="local")
    ap.add_argument("--ticks", type=int, default=100, help="horizon in ticks (real_period=1s/tick)")
    ap.add_argument("--out", required=True, help="output scenario YAML path")
    # distributed_direct machine_b params (Config B in scaling_study_plan.md Sec.5 is TBD)
    ap.add_argument("--machine-b-host", dest="machine_b_host", default="TBD_MACHINE_B_HOST",
                    help="distributed_direct only: machine_b SSH host (plan Sec.5 Config B is TBD)")
    ap.add_argument("--machine-b-user", dest="machine_b_user", default="TBD_USER",
                    help="distributed_direct only: machine_b SSH user")
    ap.add_argument("--machine-b-workdir", dest="machine_b_workdir", default="TBD_WORKDIR",
                    help="distributed_direct only: machine_b remote repo root")
    ap.add_argument("--machine-b-python", dest="machine_b_python", default=None,
                    help="distributed_direct only: machine_b explicit python interpreter (optional)")
    ap.add_argument("--manager-address", dest="manager_address", default="TBD_MANAGER_LAN_IP",
                    help="distributed_direct only: LAN IP the remote reaches this manager at")
    args = ap.parse_args(argv)

    if args.placement == "distributed_nat" and args.core_type not in ("zmq_ss", "tcp_ss"):
        print(f"NOTE: placement=distributed_nat forces core_type zmq_ss (NAT requires a single-socket "
              f"core); overriding requested core_type={args.core_type!r}.", file=sys.stderr)
        args.core_type = "zmq_ss"
    if args.mode == "seq" and args.W is not None:
        print("NOTE: --W is ignored when --mode seq.", file=sys.stderr)
    if args.placement == "distributed_direct" and args.machine_b_host.startswith("TBD"):
        print("WARNING: distributed_direct placement using TBD placeholder machine_b connection "
              "params -- pass --machine-b-host/--machine-b-user/--machine-b-workdir for a real run.",
              file=sys.stderr)
    return args


def main(argv=None):
    args = parse_args(argv)
    scenario = build_scenario(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# ============================================================================\n")
        f.write("# GENERATED FILE -- DO NOT EDIT BY HAND.\n")
        f.write("# Source: scripts/scaling_study/gen_scenario.py (regenerate with the same CLI args)\n")
        f.write("# ============================================================================\n")
        yaml.safe_dump(scenario, f, sort_keys=False, default_flow_style=False)

    spec_path = Path(str(out_path) + ".spec.json")
    spec_path.write_text(json.dumps(spec_dict(args), indent=2) + "\n")

    n_feds = args.F * args.N
    print(f"wrote {out_path}  ({args.F} federations x {args.N} federates = {n_feds} federates, "
          f"{args.M} instances/federate, mode={args.mode}, placement={args.placement})")
    print(f"wrote {spec_path}")


if __name__ == "__main__":
    main()
