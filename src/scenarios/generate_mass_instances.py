#!/usr/bin/env python3
"""Generate mass-instance-scaling scenarios ("biggest possible district").

Companion to generate_scale_sharded.py, but a DIFFERENT scaling axis. That
generator scales BUILDING COUNT by adding more FEDERATES (4 federates/site) —
which hits the ~73-81-federate-per-broker zmq_ss ceiling almost immediately
(documented in generate_scale_sharded.py / results/paper_case_study/
s4c_real_analysis.md). THIS generator instead keeps federate COUNT fixed at
exactly 4 (weather, building, heatpump, pid) and scales the NUMBER OF MODEL
INSTANCES (n_instances=K) inside each of the 3 non-weather federates. Buildings
achieved = K per federation/machine. This sidesteps the federate-count ceiling
entirely (a single broker only ever sees 4 federates, regardless of K) and
pushes the OTHER scaling axis: HELICS-interface-registration count and the
serial per-tick Python loops in BaseFederate (see
results/paper_case_study/mass_scale_bottleneck_analysis.md for the exact
file:line bottleneck citations this generator's sweep results feed into).

Topology (single shard/federation, K = n_instances of building/heatpump/pid):

    weather.0 --DryBulb--> ALL building.i, heatpump.i (list-target broadcast,
                            same value for every instance, no per-instance
                            dict targets needed for this one)
    heatpump.i --Q_heat--> building.i   (1:1, needs a DICT target: {'i': [...]})
    pid.i      --modulation--> heatpump.i (1:1, DICT target)
    building.i --T_indoor--> pid.i        (1:1, DICT target, causality:
                                            next_step — breaks the
                                            bldg->pid->hp->bldg algebraic loop,
                                            same trick as generate_scale_sharded.py)

3 dict-target subscriptions total (building.Q_heat, heatpump.modulation,
pid.T_indoor), each with K entries — this is the "dict-target YAML size grows
linearly with K" cost characterized in the bottleneck analysis.

Two modes:

  1. Single-machine / single-federation sweep (part A — the K-ceiling sweep):
         python generate_mass_instances.py --k 1000 5000 20000 50000 100000 \\
             [--core-type tcp] [--sink none] [--start ...] [--end ...]
     Writes cs_mass_k{K}.yaml for each K, ALL LOCAL (no deployment block),
     core_type defaults to plain "tcp" (local-only sweep, no NAT concern per
     the task brief — zmq_ss is reserved for the real multi-machine run).

  2. Multi-machine sharded mode (part B/C — the real "biggest district" run):
         python generate_mass_instances.py --shard-machines 1 2 3 --k 50000 \\
             [--sink none] [--start ...] [--end ...]
     Writes cs_mass_shard_{n}m.yaml: n self-contained federations (1/machine,
     SAME sharding pattern and machine aliases as generate_scale_sharded.py —
     shard_1 local/manager, shard_2 -> machine_a, shard_3 -> machine_b), each
     with its own weather + building + heatpump + pid federates at
     n_instances=K, core_type zmq_ss (mandatory for the real NAT'd remotes).
     Total buildings achieved = n_machines * K.
"""
import argparse
from pathlib import Path

HEADER = """\
# ============================================================================
# GENERATED FILE — DO NOT EDIT BY HAND.
# Source: src/scenarios/generate_mass_instances.py   (re-run it to regenerate)
#
# {title}
#
# {placement}
#
# SCALING AXIS: this scenario keeps federate COUNT fixed at 4/federation
# (weather, building, heatpump, pid) and scales via n_instances=K inside
# building/heatpump/pid instead of adding one federate per site. This avoids
# the ~73-81-federate-per-broker zmq_ss ceiling (see generate_scale_sharded.py)
# entirely — the bottleneck on THIS axis is HELICS interface-registration count
# and BaseFederate's serial per-tick Python loops, not broker federate count.
# See results/paper_case_study/mass_scale_bottleneck_analysis.md.
# ============================================================================

version: "1.0.0"
name: "{name}"
scenario_description: "{desc}"

start_time: "{start}T00:00:00"
end_time: "{end}"
log_level: ERROR          # quiet: {k} instances/federate would otherwise flood the log

memory_config:
  batch_size: 1000
  attrs: ["T_indoor", "Q_heat"]
  sink: {sink}
"""

DEPLOYMENT = """
deployment:
  manager_address: "130.192.177.14"
  machines:
    machine_a:
      host: "130.192.238.9"
      user: "eclabuser"
      ssh_port: 22
      workdir: "/home/eclabuser/CosimGym"
      conda_env: "cosim_gym"
      python: "/home/eclabuser/miniconda3/envs/cosim_gym/bin/python"
    machine_b:
      host: "130.192.238.13"
      user: "eclabuser"
      ssh_port: 22
      workdir: "/home/eclabuser/CosimGym"
      conda_env: "cosim_gym"
      python: "/home/eclabuser/miniconda3/envs/cosim_gym/bin/python"
"""

SYNC = """
synchronization:
  auto_offset:
    enabled: true
    offset_step: 0.1
    override_existing_offsets: false
  default_subscription_causality: "same_step"
  validate_causality_cycles: true
  default_startup_sync:
    enabled: true
    force_read_all_subscriptions: true
    require_updated_inputs: false
    require_finite_numeric: true
    invalid_numeric_sentinels: [-1.0e49]
    missing_inputs_policy: "warn"
    invalid_inputs_policy: "warn"
"""

# shard -> machine alias (None = local/manager). shard_1 always local. Same
# aliases/machines as generate_scale_sharded.py so both generators target the
# identical, already-verified real infrastructure.
SHARD_ALIAS = {1: None, 2: "machine_a", 3: "machine_b"}


def host_line(alias):
    return f'        host: "{alias}"\n' if alias else ""


def dict_targets_block(key_indent, count, target_fn):
    """Build a fully-indented `targets:` dict block (key line + `count` entry
    lines, one per model instance) as a single ready-to-splice multi-line
    string. `key_indent` = indentation of the `targets:` line itself (entries
    get +2). Built via list + one join — avoids O(K^2) string concatenation
    at large K."""
    pad_k = " " * key_indent
    pad_e = " " * (key_indent + 2)
    lines = [f"{pad_e}'{i}': [{target_fn(i)}]" for i in range(count)]
    return pad_k + "targets:\n" + "\n".join(lines)


def weather_block(fname, core_type, alias):
    h = host_line(alias)
    return f'''
      {fname}:
        type: "base"
{h}        core_type: "{core_type}"
        log_level: ERROR
        timing_configs:
          real_period: 3600
        flags:
          terminate_on_error: true
        connections:
          endpoints: []
          subscribes: []
          publishes:
            - key: "DryBulb"
              type: "double"
              units: "°C"
            - key: "GloHorzRad"
              type: "double"
              units: "W/m²"
            - key: "DifHorzRad"
              type: "double"
              units: "W/m²"
        model_configs:
          instantiation:
            model_name: "base_csv_reader"
            n_instances: 1
            prefix: "{fname}"
            parallel_execution: false
          parameters:
            csv_path: "model_catalog/physical_models/resources/weather_data_bj.csv"
            skip_rows: 0
          init_state:
            DryBulb: 3.0
            GloHorzRad: 0.0
            DifHorzRad: 0.0
'''


def building_block(fname, k, core_type, alias, weather_name, hp_name):
    h = host_line(alias)
    q_heat_targets = dict_targets_block(14, k, lambda i: f"{hp_name}.{i}/Q_heat")
    return f'''
      {fname}:
        type: "base"
{h}        core_type: "{core_type}"
        log_level: ERROR
        timing_configs:
          real_period: 3600
        flags:
          terminate_on_error: true
        connections:
          endpoints: []
          subscribes:
            - key: "T_ext"
              type: "double"
              units: "°C"
              targets: [{weather_name}.0/DryBulb]
            - key: "Q_heat"
              type: "double"
              units: "W"
{q_heat_targets}
          publishes:
            - key: "T_indoor"
              type: "double"
              units: "°C"
        model_configs:
          instantiation:
            model_name: "simple_building"
            n_instances: {k}
            prefix: "{fname}"
            parallel_execution: false
          parameters:
            thermal_capacitance: 1000000.0
            thermal_resistance: 0.005
            T_initial: 18.0
          init_state:
            T_ext: 3.0
            Q_heat: 0.0
            T_indoor: 18.0
'''


def heatpump_block(fname, k, core_type, alias, weather_name, pid_name):
    h = host_line(alias)
    mod_targets = dict_targets_block(14, k, lambda i: f"{pid_name}.{i}/modulation")
    return f'''
      {fname}:
        type: "base"
{h}        core_type: "{core_type}"
        log_level: ERROR
        timing_configs:
          real_period: 3600
        flags:
          terminate_on_error: true
        connections:
          endpoints: []
          subscribes:
            - key: "T_ext"
              type: "double"
              units: "°C"
              targets: [{weather_name}.0/DryBulb]
            - key: "modulation"
              type: "double"
              units: "-"
{mod_targets}
          publishes:
            - key: "Q_heat"
              type: "double"
              units: "W"
            - key: "P_elec"
              type: "double"
              units: "W"
            - key: "COP"
              type: "double"
              units: "-"
        model_configs:
          instantiation:
            model_name: "simple_heatpump"
            n_instances: {k}
            prefix: "{fname}"
            parallel_execution: false
          parameters:
            P_rated: 5000.0
            eta_carnot: 0.45
            T_supply: 45.0
            COP_min: 1.5
            COP_max: 6.0
          init_state:
            T_ext: 3.0
            modulation: 0.0
'''


def pid_block(fname, k, core_type, alias, bldg_name):
    h = host_line(alias)
    t_indoor_targets = dict_targets_block(14, k, lambda i: f"{bldg_name}.{i}/T_indoor")
    return f'''
      {fname}:
        type: "base"
{h}        core_type: "{core_type}"
        log_level: ERROR
        timing_configs:
          real_period: 3600
        flags:
          terminate_on_error: true
        connections:
          endpoints: []
          subscribes:
            - key: "T_indoor"
              type: "double"
              units: "°C"
{t_indoor_targets}
              causality: "next_step"     # breaks the bldg->pid->hp->bldg loop
          publishes:
            - key: "modulation"
              type: "double"
              units: "-"
        model_configs:
          instantiation:
            model_name: "simple_pid_controller"
            n_instances: {k}
            prefix: "{fname}"
            parallel_execution: false
          parameters:
            T_setpoint: 20.0
            Kp: 0.05
            Ki: 0.001
            Kd: 0.0
          init_state:
            T_indoor: 18.0
            modulation: 0.0
'''




def build_shard_federation(shard_label, k, core_type, alias):
    """One `federations.<shard_label>:` entry — 4 federates total
    (weather n=1, building/heatpump/pid n=K each)."""
    w = f"weather_{shard_label}"
    bldg = f"building_{shard_label}"
    hp = f"heatpump_{shard_label}"
    pid = f"pid_{shard_label}"
    out = f'''
  {shard_label}:
    broker_config:
      core_type: "{core_type}"
      federates: 4

    federate_configs:
'''
    out += weather_block(w, core_type, alias)
    out += building_block(bldg, k, core_type, alias, w, hp)
    out += heatpump_block(hp, k, core_type, alias, w, pid)
    out += pid_block(pid, k, core_type, alias, bldg)
    return out


def build_single(k, start, end, sink, core_type):
    """Part A: single machine, single federation, 4 federates, n_instances=K."""
    name = f"cs_mass_k{k}"
    desc = (f"Mass-instance-scaling sweep (part A) — 1 federation, 4 federates "
            f"(weather + building/heatpump/pid @ n_instances={k}), local, core_type={core_type}.")
    title = f"MASS-SCALE K-SWEEP — n_instances={k}, 4 federates, single broker, local {core_type}"
    placement_txt = f"PLACEMENT: everything local (manager), 1 federation, 4 federates, K={k} buildings"
    out = HEADER.format(title=title, placement=placement_txt, name=name, desc=desc,
                         start=start, end=end, sink=sink, k=k)
    out += SYNC
    out += "\nfederations:"
    out += build_shard_federation("district", k, core_type, None)
    return name, out


def build_sharded(n_machines, k, start, end, sink):
    """Parts B/C: n_machines self-contained federations, each K instances/type.
    core_type is always zmq_ss (mandatory for the real NAT'd remotes)."""
    core_type = "zmq_ss"
    total = n_machines * k
    name = f"cs_mass_shard_{n_machines}m_k{k}"
    desc = (f"Mass-instance-scaling REAL multi-machine run (parts B/C) — "
            f"{n_machines} machine(s), K={k} buildings/machine, {total} total buildings, "
            f"4 federates/machine (fixed), zmq_ss.")
    any_remote = n_machines > 1
    lines_pl = [f"PLACEMENT ({n_machines} machine(s), K={k} buildings/machine, {total} total):"]
    for shard in range(1, n_machines + 1):
        alias = SHARD_ALIAS[shard]
        where = "manager (local)" if alias is None else f"{alias} (remote, SSH)"
        lines_pl.append(f"#   shard_{shard} -> {where}: weather + building/heatpump/pid @ K={k} = 4 federates")
    placement_txt = "\n".join(lines_pl)
    title = f"MASS-SCALE REAL SHARDED RUN — {n_machines} machine(s), K={k}/machine, {total} total buildings"
    out = HEADER.format(title=title, placement=placement_txt, name=name, desc=desc,
                         start=start, end=end, sink=sink, k=k)
    if any_remote:
        out += DEPLOYMENT
    out += SYNC
    out += "\nfederations:"
    for shard in range(1, n_machines + 1):
        out += build_shard_federation(f"shard_{shard}", k, core_type, SHARD_ALIAS[shard])
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=None,
                     help="n_instances value(s) for building/heatpump/pid — single-machine sweep mode (part A)")
    ap.add_argument("--shard-machines", type=int, nargs="+", default=None, choices=(1, 2, 3),
                     help="generate the n-machine sharded scenario at a single --k value (parts B/C)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-01-01T06:00:00",
                     help="short smoke horizon by default (6 hourly ticks) — this sweep "
                          "characterizes per-tick/registration cost, not a long simulation")
    ap.add_argument("--sink", default="none", choices=("json", "parquet", "none"))
    ap.add_argument("--core-type", default="tcp", help="local-sweep transport (part A); ignored in --shard-machines mode (always zmq_ss)")
    args = ap.parse_args()

    here = Path(__file__).parent
    end = args.end if "T" in args.end else args.end + "T00:00:00"

    if args.shard_machines:
        if not args.k or len(args.k) != 1:
            ap.error("--shard-machines requires exactly one --k value")
        k = args.k[0]
        for n in args.shard_machines:
            name, text = build_sharded(n, k, args.start, end, args.sink)
            fn = here / f"{name}.yaml"
            fn.write_text(text)
            print(f"wrote {fn}  ({n} machine(s), K={k}/machine, {n*k} total buildings, {n*4} federates)")
        return

    if not args.k:
        ap.error("provide --k (single-machine sweep) or --shard-machines + --k (real multi-machine run)")

    for k in args.k:
        name, text = build_single(k, args.start, end, args.sink, args.core_type)
        fn = here / f"{name}.yaml"
        fn.write_text(text)
        print(f"wrote {fn}  (K={k}, 4 federates, core_type={args.core_type})")


if __name__ == "__main__":
    main()
