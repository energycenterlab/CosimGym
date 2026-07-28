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
    By default (`--exchange none`, the default) each federation is
    SELF-CONTAINED. It has N federates, each stepping M instances of
    `--model`. Federates have NO subscriptions, in this or any other
    federation — publishes are derived from the catalog's declared `outputs`
    for the chosen model, so parsing/registration stays valid for ANY
    catalog model without hand-wiring per-model inputs. There is therefore
    no cross-federate/cross-federation key to get wrong, no causality cycle,
    and no need for the causality-loop-break trick used in
    generate_scale_benchmark.py's pid->hp->bldg loop — F federations are
    just N-federate blocks replicated F times. `heavy_compute_dummy`
    (the primary model for this study; see scaling_study_plan.md Phase 0/1)
    has no required inputs at all, so this is a correct, general default.

Data-exchange wiring (`--exchange on`, Phase D; CONTRACT.md "Part-B /
Phase-D additions — data-exchange wiring knobs"):
    Optionally layers a bipartite pub/sub graph on top of the same federate
    layout, so cross-federate/cross-machine data exchange becomes a
    first-class, measurable cost instead of being absent by construction.
    `--distance` picks where subscribers sit relative to publishers
    (intra_fed/cross_fed/cross_machine), `--fanout` picks the federate-level
    edge pattern (1to1/1toN/Nto1/all2all), and `--msg-width`/`--freq`/
    `--causality` size and pace the exchanged payload. The wiring is always
    bipartite — a federate is never both publisher and subscriber — which
    keeps every pattern acyclic; `ScenarioManager._validate_causality_cycles()`
    would otherwise raise on a cyclic `same_step` graph before tick 1. This
    layer targets the catalog's `exchange_dummy` model (`payload` output /
    `payload_in` input, both HELICS `vector`); the derived edge count
    (`n_edges`) is written to the `.spec.json` sidecar for the bench driver
    and cost-model fitter to consume.

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

# Base of the per-federation HELICS port blocks (see build_scenario for the block
# sizing). Kept inside the 20000-30000 range src/.env reserves for HELICS.
PORT_BASE = 23500

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
    "vector": "vector",  # exchange_dummy's payload/payload_in (Phase-D)
    "list": "vector",
}


def helics_type(py_type: str) -> str:
    return _TYPE_MAP.get(str(py_type).lower(), "double")


def load_catalog_outputs(model_name: str):
    """Best-effort: (key, helics_type, unit) list for a model's declared catalog outputs.

    Falls back to a single generic 'result' double publication if the model
    isn't found or declares no outputs, so the generator degrades gracefully
    for any catalog entry rather than hand-wiring per-model knowledge.
    """
    if model_name == "exchange_dummy":
        # exchange_dummy's catalog entry declares TWO outputs -- `payload` and a
        # diagnostic `n_received` -- but the generated federate must publish ONLY
        # `payload`. Publishing `n_received` too would add an uncounted HELICS
        # edge per federate and pollute the comms accounting (n_edges / bench
        # CSV only track the wiring this generator itself creates). Hardcoded
        # here rather than read from the catalog so this holds regardless of
        # exactly how the (separately-built) catalog entry is authored.
        return [("payload", "vector", "-")]
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


# ==============================================================================
# Phase-D: optional data-exchange wiring (CONTRACT.md "Part-B / Phase-D
# additions"). Everything below is inert when --exchange none: build_scenario
# never calls into this section, so the byte-for-byte Part-A output is
# untouched by construction rather than by careful no-op branching.
# ==============================================================================

def build_federate_names(F: int, N: int):
    """Federate names grouped by federation: names_by_fed[f-1][n] ==
    'compute_{idx:03d}' with idx the GLOBAL 0-based index across all
    federations -- i.e. the exact numbering build_scenario's own idx counter
    produces. Wiring needs the full name list up front (it can reference
    federates in other federations, e.g. cross_fed), before per-federate
    configs are built.
    """
    names_by_fed = []
    idx = 0
    for _f in range(1, F + 1):
        names = []
        for _n in range(N):
            names.append(f"compute_{idx:03d}")
            idx += 1
        names_by_fed.append(names)
    return names_by_fed


def compute_exchange_edges(names_by_fed, args):
    """Bipartite publisher/subscriber wiring at federate granularity.

    Returns (edges, n_edges, pub_names, sub_names):
      edges     : subscriber federate name -> ordered list of publisher
                  federate names it targets (instance-level pairing happens
                  later, in build_subscribe_block).
      n_edges   : M * (total per-instance target links), CONTRACT.md's
                  derived knob -- computed here once so nobody downstream
                  needs to recompute it from the YAML.
      pub_names, sub_names : the full publisher-side / subscriber-side
                  federate-name sets, unioned across every wired federation
                  pair. Only consumed by the cross_machine placement
                  override below.

    Bipartite by construction: every (P, S) pair below is disjoint and
    acyclic (intra_fed splits one federation in half; cross_fed/cross_machine
    only ever point from federation f+1 back to federation f, never wrapping).
    This matters beyond style -- ScenarioManager._validate_causality_cycles()
    raises RuntimeError on any same_step dependency cycle, so a cyclic wiring
    would abort the run outright, and would force next_step everywhere,
    confounding --causality as an independent knob. --F/--N preconditions
    (F>=2 for cross_fed/cross_machine, N>=2 for intra_fed) are validated in
    parse_args, before this function ever runs.
    """
    if args.exchange == "none":
        return {}, 0, set(), set()

    F = len(names_by_fed)
    pairs = []  # list of (P, S) federate-name lists
    if args.distance == "intra_fed":
        for names in names_by_fed:
            half = len(names) // 2
            pairs.append((names[:half], names[half:]))
    else:  # cross_fed / cross_machine share the same federate-level wiring
        for f in range(F - 1):
            pairs.append((names_by_fed[f + 1], names_by_fed[f]))  # (P, S)

    edges = {}
    pub_names, sub_names = set(), set()
    for P, S in pairs:
        p = len(P)
        if args.fanout == "1to1":
            for k, sub in enumerate(S):
                edges[sub] = [P[k % p]]
        elif args.fanout == "1toN":
            for sub in S:
                edges[sub] = [P[0]]
        elif args.fanout == "Nto1":
            edges[S[0]] = list(P)
        elif args.fanout == "all2all":
            for sub in S:
                edges[sub] = list(P)
        pub_names.update(P)
        sub_names.update(S)

    n_edges = args.M * sum(len(targets) for targets in edges.values())
    return edges, n_edges, pub_names, sub_names


def max_federate_links(edges, M: int):
    """Per-federate peak inbound/outbound link counts -> (max_fed_in, max_fed_out).

    THE regressors that matter. T_tick = max over machines/federates of their
    cost (plan Sec.2), so a scenario-wide total like n_edges cannot predict it:
    the same 16 links spread over 4 subscriber federates and piled onto 1
    subscriber federate cost very different amounts, because only the busiest
    federate gates the tick. Measured Phase D, cross_fed, M-adjusted to equal
    n_edges AND equal n_subs: spread = +87 us/tick, concentrated = +157 us/tick.

    in  = M * (targets on the busiest subscriber federate) -- inbound values it
          must poll and deserialise every tick.
    out = M * (number of subscriber federates pointing at the busiest publisher)
          -- outbound fan-out its core must service.
    Kept as two numbers, not summed: receiving costs more than sending (a
    publisher fans out once inside the core, a subscriber polls every handle),
    so a fit that lumps them cannot express the asymmetry.
    """
    if not edges:
        return 0, 0
    max_in = max(len(targets) for targets in edges.values())
    fanout = {}
    for targets in edges.values():
        for pub in targets:
            fanout[pub] = fanout.get(pub, 0) + 1
    max_out = max(fanout.values()) if fanout else 0
    return M * max_in, M * max_out


def count_subscriptions(edges, M: int) -> int:
    """CONTRACT.md's derived `n_subs`: total HELICS input HANDLES registered.

    One handle per (subscriber federate x instance) -- the targets attached to a
    handle do NOT multiply it. It therefore diverges from n_edges exactly when a
    handle carries several targets (Nto1, all2all), which is the point: n_subs
    scores per-subscription polling cost (BaseFederate._receive_inputs walks
    every handle every tick, updated or not) while n_edges scores per-link
    transfer cost. Fitting both separates the two.
    """
    return M * len(edges)


def build_subscribe_block(pub_fed_names, M: int, args):
    """One `connections.subscribes` entry for exchange_dummy's `payload_in`.

    Instance-paired: subscriber instance j targets publisher instance j from
    EVERY federate in pub_fed_names (>1 only for Nto1/all2all), so edge count
    scales linearly in M rather than M^2. Target strings use the flat global
    HELICS namespace '<federate>.<instance>/<key>' -- CosimGym registers every
    publication globally (register_global_publication), with no federation
    prefix, for both intra- and cross-federation targets (confirmed by
    src/scenarios/simple_test_multifederations.yaml).
    """
    targets = {
        str(j): [f"{pub}.{j}/payload" for pub in pub_fed_names]
        for j in range(M)
    }
    sub = {
        "key": "payload_in",
        "type": "vector",
        "units": "-",
        "causality": args.causality,
    }
    if len(pub_fed_names) > 1:
        # >1 target per instance (Nto1/all2all aggregation) -- set explicitly;
        # otherwise BaseFederate logs a warning about an unset combine policy.
        sub["multi_input_handling"] = "sum"
    sub["targets"] = targets
    return sub


def apply_cross_machine_placement(place, names_by_fed, pub_names, sub_names, args):
    """cross_machine wiring requires publisher-side and subscriber-side
    federates to land on DIFFERENT machines, which the core-proportional
    (flatten_placement) / even-split (direct_placement) placement functions
    above know nothing about. Kept as one small, clearly-contained override
    applied AFTER the general placement is computed: simplest correct rule is
    subscriber-side stays on the manager (alias None), publisher-side is
    pinned to the first non-manager machine alias available for this
    placement. Only called when --distance cross_machine (parse_args already
    rejects that combination with --placement local).
    """
    alias = {"distributed_nat": "machine_a", "distributed_direct": "machine_b"}[args.placement]
    flat_names = [name for names in names_by_fed for name in names]
    for idx, name in enumerate(flat_names):
        if name in sub_names:
            place[idx] = None
        elif name in pub_names:
            place[idx] = alias
    return place


def build_federate(fed_idx: int, alias, args, outputs, pub_fed_names=None):
    """One compute federate: M instances of --model.

    pub_fed_names is None for a self-contained federate (--exchange none, or
    this federate isn't on the subscriber side of the wiring); otherwise it's
    the list of publisher federate names this federate subscribes to (see
    compute_exchange_edges / build_subscribe_block).
    """
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
    elif args.model == "exchange_dummy":
        # exchange_dummy's catalog parameters (Phase-D): msg_width/publish_every
        # are its core knobs; iterations is optional, mirroring
        # heavy_compute_dummy's --work handling for cross-model comparability.
        parameters["msg_width"] = args.msg_width
        parameters["publish_every"] = args.freq
        if args.work is not None:
            parameters["iterations"] = args.work

    subscribes = [build_subscribe_block(pub_fed_names, args.M, args)] if pub_fed_names else []

    federate = {
        "name": fed_name,
        "type": "base",
        "core_type": core_type,
        "log_level": "ERROR",
        "timing_configs": {"real_period": 1},
        "flags": {"terminate_on_error": True},
        "connections": {
            "endpoints": [],
            "subscribes": subscribes,
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

    # Phase-D: optional data-exchange wiring. names_by_fed/edges/n_edges are
    # all no-ops (empty/zero) when --exchange none -- see compute_exchange_edges.
    names_by_fed = build_federate_names(args.F, args.N)
    edges, n_edges, pub_names, sub_names = compute_exchange_edges(names_by_fed, args)
    n_subs = count_subscriptions(edges, args.M)
    max_fed_in, max_fed_out = max_federate_links(edges, args.M)
    if args.distance == "cross_machine" and edges:
        place = apply_cross_machine_placement(place, names_by_fed, pub_names, sub_names, args)

    start = datetime(2024, 1, 1)
    end = start + timedelta(seconds=args.ticks)  # real_period=1s/tick

    scenario_name = f"scaling_F{args.F}_N{args.N}_M{args.M}_{args.mode}_{args.core_type}_{args.placement}"
    description = (
        f"D1-generated scaling-study scenario: F={args.F} federations, "
        f"N={args.N} federates/federation, M={args.M} instances/federate, "
        f"mode={args.mode}, W={args.W}, core_type={args.core_type}, "
        f"model={args.model}, work={args.work}, placement={args.placement}, "
        f"ticks={args.ticks}."
    )
    if args.exchange == "on":
        # Byte-identical Part-A output is a hard requirement, so exchange
        # knobs are only ever appended (never inserted) into name/description.
        scenario_name += (
            f"_x{args.distance}_{args.fanout}_w{args.msg_width}_f{args.freq}_{args.causality}"
        )
        description += (
            f" Data-exchange wiring: distance={args.distance}, fanout={args.fanout}, "
            f"msg_width={args.msg_width}, freq={args.freq}, causality={args.causality}, "
            f"n_edges={n_edges}."
        )
    description += (
        " Generated by scripts/scaling_study/gen_scenario.py "
        "-- do not hand-edit; regenerate from the CLI args instead."
    )

    scenario = {
        "version": "1.0.0",
        "name": scenario_name,
        "scenario_description": description,
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
            name = names_by_fed[f - 1][n]
            pub_fed_names = edges.get(name)
            name, fed_cfg = build_federate(idx, place.get(idx), args, outputs, pub_fed_names)
            federate_configs[name] = fed_cfg
            idx += 1
        federations[fed_name] = {
            "broker_config": {
                "core_type": args.core_type,
                # Port BLOCK per federation, sized from N -- a fixed stride is wrong.
                # Two things live above a federation's broker port with plain zmq:
                #   1. the broker's paired reply socket at `port + 1`
                #      (ScenarioManager._broker_ports()), and
                #   2. EVERY federate core's own inbound listener, which HELICS binds
                #      at `port + 10 + n` for n in 0..N-1 (ScenarioManager:1640).
                # (2) is why the earlier fixed stride of 10 -- itself the fix for an
                # even earlier stride of 1 -- was still only safe up to N ~ 8. At N=8,
                # federation 1's cores climbed to 23527 while federation 2's broker sat
                # at 23520, and federates died with "Unable to bind zmq pull socket
                # giving up tcp://127.0.0.1:23521" -> "unable to register federate".
                # It fails as UNWIRED controls too, so it is a port-planning bug, not a
                # data-exchange limit. Block = 10 (core offset) + N (one per core) + 12
                # (broker port+1 and headroom).
                "port": PORT_BASE + f * (args.N + 22),
                "federates": args.N,
            },
            "federate_configs": federate_configs,
        }
    scenario["federations"] = federations
    return scenario, n_edges, n_subs, max_fed_in, max_fed_out


def spec_dict(args, n_edges: int = 0, n_subs: int = 0,
              max_fed_in: int = 0, max_fed_out: int = 0):
    """Canonical knob dict — names/types exactly as CONTRACT.md D1 table,
    plus the Part-B/Phase-D data-exchange knobs. n_edges/n_subs are passed in
    (computed once by compute_exchange_edges/count_subscriptions via
    build_scenario) rather than recomputed here, so there is a single source
    of truth for them.
    """
    n_machines = {"local": 1, "distributed_nat": 3, "distributed_direct": 2}[args.placement]
    spec = {
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
    if args.exchange == "on":
        spec.update({
            "exchange": "on",
            "distance": args.distance,
            "fanout": args.fanout,
            "msg_width": args.msg_width,
            "freq": args.freq,
            "causality": args.causality,
            "n_edges": n_edges,
            "n_subs": n_subs,
            "max_fed_in": max_fed_in,
            "max_fed_out": max_fed_out,
        })
    else:
        # Part-A-style row (CONTRACT.md): exchange="none",,,1,1,,0,0 -- knobs are
        # inert when off, so their spec values are pinned regardless of
        # whatever (unused) --msg-width/--freq/--causality were passed.
        spec.update({
            "exchange": "none",
            "distance": "",
            "fanout": "",
            "msg_width": 1,
            "freq": 1,
            "causality": "",
            "n_edges": 0,
            "n_subs": 0,
            "max_fed_in": 0,
            "max_fed_out": 0,
        })
    return spec


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
    # Phase-D data-exchange wiring knobs (CONTRACT.md "Part-B / Phase-D additions").
    # All inert when --exchange none (the default) -- byte-identical Part-A output.
    ap.add_argument("--exchange", choices=("none", "on"), default="none",
                    help="none (default) -> self-contained federations, no subscriptions (Part-A "
                         "behaviour, byte-identical output); on -> wire a bipartite pub/sub graph "
                         "using --model exchange_dummy")
    ap.add_argument("--distance", choices=("intra_fed", "cross_fed", "cross_machine"), default="intra_fed",
                    help="only used with --exchange on: where subscribers sit relative to publishers "
                         "-- intra_fed (within each federation), cross_fed (adjacent federations), "
                         "cross_machine (like cross_fed, but P/S pinned to different machines)")
    ap.add_argument("--fanout", choices=("1to1", "1toN", "Nto1", "all2all"), default="1to1",
                    help="only used with --exchange on: federate-level edge pattern between the "
                         "publisher side P and subscriber side S")
    ap.add_argument("--msg-width", dest="msg_width", type=int, default=1,
                    help="only used with --exchange on: exchange_dummy published payload vector "
                         "length (1 = scalar-equivalent)")
    ap.add_argument("--freq", type=int, default=1,
                    help="only used with --exchange on: exchange_dummy publishes every Nth tick "
                         "(1 = every tick)")
    ap.add_argument("--causality", choices=("same_step", "next_step"), default="same_step",
                    help="only used with --exchange on: subscription causality on the wired edges")
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

    if args.exchange == "on":
        if args.distance == "intra_fed" and args.N < 2:
            sys.exit("error: --distance intra_fed requires --N >= 2 (needs a publisher half and a "
                      "subscriber half within each federation)")
        if args.distance in ("cross_fed", "cross_machine") and args.F < 2:
            sys.exit(f"error: --distance {args.distance} requires --F >= 2 (wiring connects each "
                      f"federation to the next)")
        if args.distance == "cross_machine" and args.placement == "local":
            sys.exit("error: --distance cross_machine requires a distributed --placement "
                      "(distributed_nat or distributed_direct) -- it has no meaning for --placement local")
    return args


def main(argv=None):
    args = parse_args(argv)
    scenario, n_edges, n_subs, max_fed_in, max_fed_out = build_scenario(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# ============================================================================\n")
        f.write("# GENERATED FILE -- DO NOT EDIT BY HAND.\n")
        f.write("# Source: scripts/scaling_study/gen_scenario.py (regenerate with the same CLI args)\n")
        f.write("# ============================================================================\n")
        yaml.safe_dump(scenario, f, sort_keys=False, default_flow_style=False)

    spec_path = Path(str(out_path) + ".spec.json")
    spec_path.write_text(json.dumps(
        spec_dict(args, n_edges, n_subs, max_fed_in, max_fed_out), indent=2) + "\n")

    n_feds = args.F * args.N
    print(f"wrote {out_path}  ({args.F} federations x {args.N} federates = {n_feds} federates, "
          f"{args.M} instances/federate, mode={args.mode}, placement={args.placement})")
    print(f"wrote {spec_path}")


if __name__ == "__main__":
    main()
