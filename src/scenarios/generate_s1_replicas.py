#!/usr/bin/env python3
"""Replicate the cs_s1_models topology N times as MODEL INSTANCES (not federates).

`cs_s1_models.yaml` is the template: ONE federation, one shared weather CSV
reader, and one "site" made of six federates

    weather ──DryBulb────────────┬─> heatpump  ──Q_heat──> building
            ──GloHorzRad/DifHorzRad─> pv        ──PV_power─┬─> rb_controller
                                                           └─> battery
    pid ──modulation──> heatpump ──P_elec──┬─> rb_controller ──Battery_power──> battery
    building ──T_indoor──> pid (causality: next_step, breaks the loop)
    battery ──SOC──> rb_controller (causality: next_step, breaks the loop)

This generator keeps EXACTLY those six federates and that wiring, and scales the
district by raising `model_configs.instantiation.n_instances` to N on every
federate except weather (which stays a single shared instance). Federate count
is therefore constant at 7 for any N — the scaling axis here is model instances
per federate, not processes.

Connections are matched index-by-index: instance i of every federate talks only
to instance i of the others (pid.i <-> building.i <-> heatpump.i <-> pv.i <->
rb_controller.i <-> battery.i). That is expressed with the per-instance
`targets:` dict form (`{'0': [...], '1': [...]}`), which `BaseFederate._register_subs`
resolves by the instance index of the subscribing model. Weather subscriptions
point every instance at the single `weather_federate.0/...` key.

Per-instance parametrisation: every scalar parameter becomes a LIST of length N,
which `BaseModel._resolve_parameter_value` indexes by instance number
(`mod_num`). Values are drawn from physically sensible uniform ranges kept
inside the catalog's declared min/max, from a seeded RNG so a given
(N, seed) pair always regenerates the identical file.

Usage:
    python src/scenarios/generate_s1_replicas.py --N 10
    python src/scenarios/generate_s1_replicas.py --N 50 --seed 7 --parallel \
        --end 2024-02-01 --sink parquet --out src/scenarios/cs_s1_N50.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

# --- weather file (Beijing) --------------------------------------------------
# The PV model's lat/long must match the weather CSV's location, so they are NOT
# randomised; only the array geometry/sizing is.
WEATHER_CSV = "model_catalog/physical_models/resources/weather_data_bj.csv"
SITE_LAT, SITE_LONG = 39.8, 116.467


# --- federate names & the short aliases accepted by --parallel/--max-workers --
# `weather_federate` is listed for completeness but always stays sequential:
# it has a single instance, so a worker pool would only add process overhead.
FEDERATES = {
    "pid": "pid_federate",
    "heatpump": "heatpump_federate",
    "hp": "heatpump_federate",
    "building": "building_federate",
    "bldg": "building_federate",
    "pv": "pv_federate",
    "bems": "rb_controller",
    "rb_controller": "rb_controller",
    "battery": "battery_federate",
    "batt": "battery_federate",
    "weather": "weather_federate",
}
REPLICATED = ["pid_federate", "heatpump_federate", "building_federate",
              "pv_federate", "rb_controller", "battery_federate"]


def resolve_federate(token):
    """Short alias or full federate name -> full federate name."""
    t = token.strip()
    if t in FEDERATES:
        return FEDERATES[t]
    if t in FEDERATES.values():
        return t
    raise ValueError(
        f"unknown federate '{token}'. Use one of: {', '.join(sorted(set(FEDERATES)))} "
        f"(or 'all' / 'none')")


def parse_parallel(spec):
    """--parallel value -> set of federate names to run with parallel_execution.

    'all' (the bare flag) = every replicated federate, 'none'/'' = nobody,
    otherwise a comma-separated list of federate names/aliases:
        --parallel building,heatpump          # only those two
        --parallel bldg,pv,batt
    """
    if not spec or spec.lower() == "none":
        return set()
    if spec.lower() == "all":
        return set(REPLICATED)
    out = set()
    for tok in spec.split(","):
        if not tok.strip():
            continue
        name = resolve_federate(tok)
        if name == "weather_federate":
            print("note: weather_federate has 1 instance; parallel_execution ignored for it")
            continue
        out.add(name)
    return out


def parse_max_workers(spec):
    """--max-workers value -> {federate_name: n}. Accepts a bare int (applied to
    every parallel federate) or per-federate `fed=n` pairs:
        --max-workers 8
        --max-workers building=16,pv=4
    """
    if not spec:
        return {}
    spec = str(spec).strip()
    if "=" not in spec:
        n = int(spec)
        if n < 1:
            raise ValueError("--max-workers must be >= 1")
        return {f: n for f in REPLICATED}
    out = {}
    for tok in spec.split(","):
        if not tok.strip():
            continue
        fed_tok, _, n = tok.partition("=")
        n = int(n)
        if n < 1:
            raise ValueError("--max-workers must be >= 1")
        out[resolve_federate(fed_tok)] = n
    return out


def _u(rng, lo, hi, nd=6):
    return round(rng.uniform(lo, hi), nd)


def draw_params(n, seed):
    """Draw N per-instance parameter sets. Every value stays within the catalog
    min/max for its parameter; ranges are narrowed to physically plausible
    residential-scale values so the district is diverse but not absurd."""
    rng = random.Random(seed)
    p = {
        # simple_building (1R1C)
        "bldg": {
            "thermal_capacitance": [_u(rng, 0.8e6, 3.0e6, 0) for _ in range(n)],
            "thermal_resistance": [_u(rng, 0.003, 0.012) for _ in range(n)],
            "T_initial": [_u(rng, 17.0, 21.0, 2) for _ in range(n)],
        },
        # simple_heatpump
        "hp": {
            "P_rated": [_u(rng, 3000.0, 12000.0, 0) for _ in range(n)],
            "eta_carnot": [_u(rng, 0.35, 0.55, 3) for _ in range(n)],
            "T_supply": [_u(rng, 35.0, 55.0, 1) for _ in range(n)],
            "COP_min": [1.5] * n,
            "COP_max": [6.0] * n,
        },
        # simple_pid_controller
        "pid": {
            "T_setpoint": [_u(rng, 19.0, 22.0, 1) for _ in range(n)],
            "Kp": [_u(rng, 0.02, 0.08, 4) for _ in range(n)],
            "Ki": [_u(rng, 5.0e-6, 5.0e-5, 8) for _ in range(n)],
            "Kd": [_u(rng, 0.4, 1.5, 3) for _ in range(n)],
        },
        # pv_dest — lat/long pinned to the weather file's site
        "pv": {
            "lat": [SITE_LAT] * n,
            "long": [SITE_LONG] * n,
            "calc_area": [_u(rng, 20.0, 120.0, 1) for _ in range(n)],
            "Tilt_angle": [_u(rng, 0.35, 0.75, 4) for _ in range(n)],
            "Azimuth_angle": [_u(rng, 0.5, 1.3, 3) for _ in range(n)],
            "area_ratio": [1] * n,
            "SVF_hori": [_u(rng, 0.85, 1.0, 3) for _ in range(n)],
            "Reflectance": [0.2] * n,
            "NOCT": [45] * n,
            "Power_rated_pv": [410] * n,
            "length": [2.05] * n,
            "width": [1.02] * n,
            "solar_constant": [1353] * n,
            "std_long": [120] * n,
        },
        # battery_dest — charge/discharge power sized as C/5 of the drawn capacity
        "batt": {},
        # rb_bems
        "bems": {},
    }

    cap = [float(round(rng.uniform(20000.0, 80000.0), -2)) for _ in range(n)]
    soc0 = [_u(rng, 0.40, 0.80, 3) for _ in range(n)]
    p["batt"] = {
        "rated_capacity": cap,
        "maximum_charge_power": [round(c / 5.0, 0) for c in cap],
        "maximum_discharge_power": [round(c / 5.0, 0) for c in cap],
        "SOC_upper_limit": [0.95] * n,
        "SOC_lower_limit": [0.25] * n,
        "charge_efficiency": [_u(rng, 0.92, 1.0, 3) for _ in range(n)],
        "discharge_efficiency": [_u(rng, 0.92, 1.0, 3) for _ in range(n)],
        "self_discharge_rate": [0] * n,
        "SOC": soc0,
    }
    soc_min = [_u(rng, 0.30, 0.45, 3) for _ in range(n)]
    p["bems"] = {
        "SOC_min": soc_min,
        "SOC_max": [round(lo + rng.uniform(0.20, 0.35), 3) for lo in soc_min],
    }
    p["_soc0"] = soc0
    p["_T_init"] = p["bldg"]["T_initial"]
    return p


def per_instance(n, fmt):
    """{'0': ['<fmt with 0>'], '1': [...], ...} — index-matched targets."""
    return {str(i): [fmt.format(i=i)] for i in range(n)}


def shared(key):
    """Plain-list targets apply to EVERY instance (BaseFederate._register_subs uses
    the list as-is when `targets` is not a dict) — used for the shared weather feed."""
    return [key]


def build(n, seed, start, end, sink, batch_size, log_level, parallel, max_workers):
    """`parallel` is the SET of federate names to run with parallel_execution
    (see parse_parallel), `max_workers` a {federate: n} map (see parse_max_workers)."""
    p = draw_params(n, seed)

    def inst(fed_key, model_name, prefix=None, params=None, init=None, user=None):
        par = fed_key in parallel
        block = {
            "model_name": model_name,
            "n_instances": n,
            "parallel_execution": par,
        }
        if prefix:
            block["prefix"] = prefix
        if par and max_workers.get(fed_key):
            block["max_parallel_workers"] = max_workers[fed_key]
        out = {"instantiation": block, "parameters": params or {}, "init_state": init or {}}
        out["user_defined"] = user or {}
        return out

    def fed(name, core_name, offset, subs, pubs, model_cfgs, log=None, extra=None):
        f = {
            "name": name,
            "type": "base",
            "log_level": log or log_level,
            "core_name": core_name,
            "core_type": "tcp",
            "timing_configs": {"real_period": 900},
            "flags": {"terminate_on_error": True, "wait_for_current_time_update": False},
            "connections": {"endpoints": [], "subscribes": subs, "publishes": pubs},
            "model_configs": model_cfgs,
        }
        if offset is not None:
            f["timing_configs"]["time_offset"] = offset
        if extra:
            f.update(extra)
        return f

    def pub(key, type_, units):
        return {"key": key, "type": type_, "units": units}

    def sub(key, type_, units, targets, causality=None, multi=None):
        s = {"key": key, "type": type_, "units": units, "targets": targets}
        if causality:
            s["causality"] = causality
        if multi:
            s["multi_input_handling"] = multi
        return s

    feds = {}

    # --- weather: ONE shared instance, no subscriptions ----------------------
    feds["weather_federate"] = fed(
        "weather_federate", "fed1", None, [],
        [pub("DryBulb", "double", "°C"),
         pub("GloHorzRad", "double", "W/m²"),
         pub("DifHorzRad", "double", "W/m²")],
        {
            "instantiation": {"model_name": "base_csv_reader", "n_instances": 1,
                              "prefix": "weather", "parallel_execution": False},
            "parameters": {"csv_path": WEATHER_CSV, "skip_rows": 0},
            "init_state": {"DryBulb": 3.0, "GloHorzRad": 0.0, "DifHorzRad": 0.0},
            "user_defined": {},
        },
        log="ERROR",
    )

    # --- pid_i <- building_i -------------------------------------------------
    feds["pid_federate"] = fed(
        "pid_federate", "bldgs_fed2", 0.1,
        [sub("T_indoor", "double", "°C",
             per_instance(n, "building_federate.{i}/T_indoor"), causality="next_step")],
        [pub("modulation", "double", "-")],
        inst("pid_federate", "simple_pid_controller", "pid", p["pid"],
             {"T_indoor": list(p["_T_init"]), "modulation": 0.0}),
    )

    # --- heatpump_i <- weather, pid_i ---------------------------------------
    feds["heatpump_federate"] = fed(
        "heatpump_federate", "bldgs_fed3", 0.2,
        [sub("T_ext", "double", "°C", shared("weather_federate.0/DryBulb")),
         sub("modulation", "double", "-", per_instance(n, "pid_federate.{i}/modulation"))],
        [pub("Q_heat", "double", "W"), pub("P_elec", "double", "W"), pub("COP", "double", "-")],
        inst("heatpump_federate", "simple_heatpump", "hp", p["hp"],
             {"T_ext": 3.0, "modulation": 0.0, "Q_heat": 0.0, "P_elec": 0.0, "COP": 0.0},
             user={"integrator": "fixed-step"}),
    )

    # --- building_i <- weather, heatpump_i -----------------------------------
    feds["building_federate"] = fed(
        "building_federate", "bldgs_fed4", 0.3,
        [sub("T_ext", "double", "°C", shared("weather_federate.0/DryBulb")),
         sub("Q_heat", "double", "W", per_instance(n, "heatpump_federate.{i}/Q_heat"))],
        [pub("T_indoor", "double", "°C")],
        inst("building_federate", "simple_building", "bldg", p["bldg"],
             {"T_ext": 3.0, "Q_heat": 0.0, "T_indoor": list(p["_T_init"])},
             user={"integrator": "euler"}),
    )

    # --- pv_i <- weather -----------------------------------------------------
    feds["pv_federate"] = fed(
        "pv_federate", "gen_fed2", 0.2,
        [sub("T_ext", "double", "°C", shared("weather_federate.0/DryBulb")),
         sub("GHI", "double", "W/m²", shared("weather_federate.0/GloHorzRad")),
         sub("DHI", "double", "W/m²", shared("weather_federate.0/DifHorzRad"))],
        [pub("PV_power", "double", "W")],
        inst("pv_federate", "pv_dest", "pv", p["pv"], {}, user={"integrator": "fixed-step"}),
    )

    # --- rb_controller_i <- pv_i, heatpump_i, battery_i ----------------------
    feds["rb_controller"] = fed(
        "rb_controller", "gen_fed3", 0.3,
        [sub("P_gen", "double", "W", per_instance(n, "pv_federate.{i}/PV_power")),
         sub("P_load", "double", "W", per_instance(n, "heatpump_federate.{i}/P_elec")),
         sub("SOC", "double", "-", per_instance(n, "battery_federate.{i}/SOC"),
             causality="next_step")],
        [pub("Battery_power", "double", "W")],
        inst("rb_controller", "rb_bems", None, p["bems"], {"Battery_power": 0.0}),
    )

    # --- battery_i <- rb_controller_i, heatpump_i, pv_i ----------------------
    feds["battery_federate"] = fed(
        "battery_federate", "gen_fed4", 0.4,
        [sub("Battery_power", "double", "W",
             per_instance(n, "rb_controller.{i}/Battery_power")),
         sub("P_load", "double", "W", per_instance(n, "heatpump_federate.{i}/P_elec")),
         sub("PV_power", "double", "W", per_instance(n, "pv_federate.{i}/PV_power"))],
        [pub("SOC", "double", "-"), pub("P_net", "double", "W"),
         pub("P_clipped", "double", "W")],
        inst("battery_federate", "battery_dest", None, p["batt"], {"SOC": list(p["_soc0"])},
             user={"integrator": "euler"}),
        extra={"startup_sync": {"required_inputs": ["Battery_power", "P_load", "PV_power"],
                                "missing_inputs_policy": "warn"}},
    )

    memory = {"batch_size": batch_size, "attrs": "all"}
    if sink != "json":
        memory["sink"] = sink

    scenario = {
        "version": "1.0.0",
        "name": None,  # filled by caller
        "scenario_description": (
            f"S1 case study replicated to N={n} model instances per federate. "
            f"One federation, 7 federates (1 shared weather + 6 replicated), "
            f"index-matched connections, per-instance randomised parameters (seed {seed})."
        ),
        "start_time": f"{start}T00:00:00",
        "end_time": f"{end}T00:00:00",
        "log_level": log_level,
        "memory_config": memory,
        "synchronization": {
            "auto_offset": {"enabled": True, "offset_step": 0.1,
                            "override_existing_offsets": False},
            "default_subscription_causality": "same_step",
            "validate_causality_cycles": True,
            "default_startup_sync": {
                "enabled": True,
                "force_read_all_subscriptions": True,
                "require_updated_inputs": True,
                "require_finite_numeric": True,
                "invalid_numeric_sentinels": [-1.0e49],
                "missing_inputs_policy": "warn",
                "invalid_inputs_policy": "warn",
            },
        },
        "federations": {
            "building": {
                "broker_config": {"core_type": "tcp", "federates": len(feds),
                                  "log_level": log_level},
                "federate_configs": feds,
            }
        },
    }
    return scenario


HEADER = """\
# {name}.yaml — GENERATED FILE, DO NOT EDIT BY HAND.
#
# Produced by src/scenarios/generate_s1_replicas.py:
#     python src/scenarios/generate_s1_replicas.py --N {n} --seed {seed}
#
# cs_s1_models.yaml replicated to N={n} model instances per federate. The
# federation, the 7 federates and the wiring are the template's; only
# n_instances, the per-instance target maps and the per-instance parameter
# lists change with N. Instance i of every federate is connected only to
# instance i of the others; the single weather instance feeds all of them.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, required=True,
                    help="model instances per replicated federate (weather stays 1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the per-instance parameter draw (default 42)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-01-02")
    ap.add_argument("--sink", default="json", choices=("json", "parquet", "none"))
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--log-level", default="ERROR",
                    choices=("ERROR", "WARNING", "INFO", "DEBUG"))
    ap.add_argument("--parallel", nargs="?", const="all", default="none",
                    metavar="all|none|FED[,FED...]",
                    help="which federates get instantiation.parallel_execution: bare "
                         "--parallel (or 'all') = every replicated federate, 'none' = "
                         "sequential (default), or a comma-separated list of federate "
                         "names/aliases (pid, heatpump/hp, building/bldg, pv, bems, "
                         "battery/batt), e.g. --parallel building,pv")
    ap.add_argument("--max-workers", default=None, metavar="N|FED=N[,FED=N...]",
                    help="max_parallel_workers: a bare int applied to every parallel "
                         "federate, or per-federate 'fed=n' pairs (e.g. building=16,pv=4). "
                         "Only emitted for federates that are actually parallel; omitted "
                         "=> the framework default min(n_instances, cpu_count).")
    ap.add_argument("--name", default=None,
                    help="scenario name (default cs_s1_models_N<N>)")
    ap.add_argument("--out", default=None,
                    help="output path (default src/scenarios/<name>.yaml)")
    args = ap.parse_args()

    if args.N < 1:
        ap.error("--N must be >= 1")

    try:
        parallel = parse_parallel(args.parallel)
        max_workers = parse_max_workers(args.max_workers)
    except ValueError as e:
        ap.error(str(e))

    name = args.name or f"cs_s1_models_N{args.N}"
    scenario = build(args.N, args.seed, args.start, args.end, args.sink,
                     args.batch_size, args.log_level, parallel, max_workers)
    scenario["name"] = name

    class NoAlias(yaml.SafeDumper):
        def ignore_aliases(self, data):  # never emit &anchors/*aliases in a scenario
            return True

    body = yaml.dump(scenario, Dumper=NoAlias, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=100)
    text = HEADER.format(name=name, n=args.N, seed=args.seed) + "\n" + body

    out = Path(args.out) if args.out else Path(__file__).parent / f"{name}.yaml"
    out.write_text(text, encoding="utf-8")
    par = ", ".join(sorted(parallel)) if parallel else "none (all sequential)"
    print(f"wrote {out}  (N={args.N} instances x 6 federates + 1 weather, seed={args.seed})")
    print(f"  parallel_execution: {par}")
    if parallel and max_workers:
        mw = ", ".join(f"{f}={max_workers[f]}" for f in sorted(parallel) if f in max_workers)
        if mw:
            print(f"  max_parallel_workers: {mw}")


if __name__ == "__main__":
    main()
