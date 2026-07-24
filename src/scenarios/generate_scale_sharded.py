#!/usr/bin/env python3
"""Generate the S4c REAL-MACHINE sharded-capacity scenario trio.

Companion/contrast to generate_scale_benchmark.py (single federation spread
across machines). THIS generator's whole point is different: it shows that
raw wall-clock speedup is NOT the multi-machine story on this hardware/network
(generate_scale_benchmark.py already measured that: distributing one shared
workload was ~1.47x SLOWER, sync-bound). The honest benefit is CAPACITY via
FEDERATION SHARDING.

Why sharding, not one big federation
-------------------------------------
zmq_ss (required for NAT'd remotes — single outbound socket, see
docs/user_guide/multi_machine_test_walkthrough.md A0) has a measured PER-BROKER
federate ceiling on this LAN of ~33 (see generate_scale_benchmark.py docstring:
17 OK, 33 OK, 49 flaky, 65 fails with "[-101] lost comms" after ~52s). That
ceiling is per HELICS broker == per FEDERATION, regardless of how many
physical machines sit under it — spreading one federation over 3 machines does
NOT lift it (evidence: benchmark_scale_distributed.yaml's ~201-federate single
federation was noted flaky in commit e9cee8c).

So: each MACHINE gets its OWN federation (its own broker, its own ~33-federate
ceiling), self-contained (own weather federate + N sites, no cross-federation
pub/sub — keeps each shard's broker load identical across configs, so timing
differences are attributable to distribution alone, not extra hierarchy-broker
traffic). ScenarioManager auto-inserts the hierarchy broker across federations
regardless (docs/user_guide/scenario_configuration/federation.md "Multi-federation
scenarios"), already validated combined with remote `host:` federates by
src/scenarios/distributed_multifederation_test.yaml and
tests/regression_suite.py's multifed+dist combo.

Topology per shard (identical to generate_scale_benchmark.py's per-site wiring,
federate names given a shard-unique prefix `s{shard}` — HELICS federate names
must be unique across the WHOLE broker hierarchy, not just per-federation):

    weather_s{shard} --DryBulb/GloHorzRad/DifHorzRad--> pv_s{shard}_NN (pv_dest)
                                                          bldg_s{shard}_NN (simple_building)
                                                          hp_s{shard}_NN   (simple_heatpump)
    pid_s{shard}_NN -modulation-> hp -Q_heat-> bldg -T_indoor-> pid
      (loop broken by causality: next_step on pid's T_indoor sub, same trick
       as bui_hp_test_base.yaml / generate_scale_benchmark.py)

Federate count per shard = 4*sites_per_shard + 1 (weather). Total federate
count = n_machines * (4*sites_per_shard + 1).

Placement:
    n_machines=1: ONE federation ("shard_1"), all local (manager). Uses
                  zmq_ss even though local-only could use plain zmq/tcp —
                  keeping the protocol IDENTICAL across the 1/2/3-machine
                  configs isolates "does sharding+more machines help" from
                  "did we also change the wire protocol".
    n_machines=2: shard_1 local (manager), shard_2 -> host: machine_a.
    n_machines=3: shard_1 local (manager), shard_2 -> host: machine_a,
                  shard_3 -> host: machine_b.

Each shard is deliberately identical size (sites_per_shard, default 8 = 33
federates/shard, the known-safe zmq_ss ceiling) so per-shard broker load never
changes as n_machines grows — only the number of PARALLEL shards grows. The
capacity claim: total sites achieved scales ~linearly with machine count
(n_machines * sites_per_shard) while per-shard wall-clock stays roughly flat,
because each shard's own broker sees the same load regardless of how many
other shards exist elsewhere.

Real infrastructure (same 3 machines as generate_scale_benchmark.py):
    manager   (this machine) 112 cores, LAN IP 130.192.177.14
    machine_a (eclab-cloud1)  32 cores, 130.192.238.9
    machine_b (eclab-cloud5)  32 cores, 130.192.238.13

Usage:
    python src/scenarios/generate_scale_sharded.py \\
        [--sites-per-shard N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

    # ceiling-characterization throwaway scenarios (single machine, single
    # federation, local, zmq_ss) at a list of shard sizes:
    python src/scenarios/generate_scale_sharded.py --ceiling 8 10 12 14 16 \\
        [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
from pathlib import Path

HEADER = """\
# ============================================================================
# GENERATED FILE — DO NOT EDIT BY HAND.
# Source: src/scenarios/generate_scale_sharded.py   (re-run it to regenerate)
#
# {title}
#
# {placement}
#
# CEILING RATIONALE: zmq_ss (required for NAT'd remotes, single outbound
# socket) has a measured per-broker federate ceiling on this LAN of ~33
# (17 OK, 33 OK, 49 flaky, 65 fails — see generate_scale_benchmark.py
# docstring). That ceiling is PER FEDERATION (= per broker), not per machine,
# so one federation spread over many machines does NOT lift it. This scenario
# therefore SHARDS the workload: one self-contained federation per machine,
# each sized at the known-safe {sites}-site / {n_feds_shard}-federate ceiling.
# Total federates in this file: {n_machines} shard(s) x {n_feds_shard} = {n_feds_total}.
# ============================================================================

version: "1.0.0"
name: "{name}"
scenario_description: "{desc}"

start_time: "{start}T00:00:00"
end_time: "{end}T00:00:00"
log_level: ERROR          # quiet: {n_feds_total} federates would otherwise flood the log

memory_config:
  batch_size: 1000
  attrs: ["PV_power", "T_indoor", "Q_heat", "P_elec"]
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

# shard -> machine alias (None = local/manager). shard_1 always local.
SHARD_ALIAS = {1: None, 2: "machine_a", 3: "machine_b"}


def host_line(alias):
    return f'        host: "{alias}"\n' if alias else ""


def weather_block(shard):
    w = f"weather_s{shard}"
    return f'''
      {w}:
        type: "base"
        core_type: "zmq_ss"
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
            prefix: "{w}"
            parallel_execution: false
          parameters:
            csv_path: "model_catalog/physical_models/resources/weather_data_bj.csv"
            skip_rows: 0
          init_state:
            DryBulb: 3.0
            GloHorzRad: 0.0
            DifHorzRad: 0.0
'''


def site_blocks(shard, i, alias):
    """The 4 federates of site i within shard `shard`. Names shard-prefixed
    (s{shard}) so they stay unique across the whole broker hierarchy."""
    h = host_line(alias)
    s = f"{i:02d}"
    w = f"weather_s{shard}"
    pv, bldg, hp, pid = (f"pv_s{shard}_{s}", f"bldg_s{shard}_{s}",
                         f"hp_s{shard}_{s}", f"pid_s{shard}_{s}")
    return f'''
      {pv}:
        type: "base"
{h}        core_type: "zmq_ss"
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
              targets:
                '0': [{w}.0/DryBulb]
            - key: "GHI"
              type: "double"
              units: "W/m²"
              targets:
                '0': [{w}.0/GloHorzRad]
            - key: "DHI"
              type: "double"
              units: "W/m²"
              targets:
                '0': [{w}.0/DifHorzRad]
          publishes:
            - key: "PV_power"
              type: "double"
              units: "W"
        model_configs:
          instantiation:
            model_name: "pv_dest"
            n_instances: 1
            prefix: "{pv}"
            parallel_execution: false
          parameters:
            lat: [39.8]
            long: [116.467]
            calc_area: [50]
            Tilt_angle: [0.6981317010]
            Azimuth_angle: [0.9]
            area_ratio: [1]
            SVF_hori: [1]
            Reflectance: [0.2]
            NOCT: [45]
            Power_rated_pv: [410]
            length: [2.05]
            width: [1.02]
            solar_constant: [1353]
            std_long: [120]
          init_state: {{}}

      {bldg}:
        type: "base"
{h}        core_type: "zmq_ss"
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
              targets:
                '0': [{w}.0/DryBulb]
            - key: "Q_heat"
              type: "double"
              units: "W"
              targets:
                '0': [{hp}.0/Q_heat]
          publishes:
            - key: "T_indoor"
              type: "double"
              units: "°C"
        model_configs:
          instantiation:
            model_name: "simple_building"
            n_instances: 1
            prefix: "{bldg}"
            parallel_execution: false
          parameters:
            thermal_capacitance: 1000000.0
            thermal_resistance: 0.005
            T_initial: 18.0
          init_state:
            T_ext: 3.0
            Q_heat: 0.0
            T_indoor: 18.0

      {hp}:
        type: "base"
{h}        core_type: "zmq_ss"
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
              targets:
                '0': [{w}.0/DryBulb]
            - key: "modulation"
              type: "double"
              units: "-"
              targets:
                '0': [{pid}.0/modulation]
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
            n_instances: 1
            prefix: "{hp}"
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

      {pid}:
        type: "base"
{h}        core_type: "zmq_ss"
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
              targets:
                '0': [{bldg}.0/T_indoor]
              causality: "next_step"     # breaks the bldg->pid->hp->bldg loop
          publishes:
            - key: "modulation"
              type: "double"
              units: "-"
        model_configs:
          instantiation:
            model_name: "simple_pid_controller"
            n_instances: 1
            prefix: "{pid}"
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


def build_shard_federation(shard, sites_per_shard, alias):
    """One `federations.shard_N:` entry — self-contained (own weather + sites,
    no cross-federation pub/sub) so every shard's broker load is identical
    regardless of n_machines."""
    n_feds = 4 * sites_per_shard + 1
    out = f'''
  shard_{shard}:
    broker_config:
      core_type: "zmq_ss"        # single socket: NAT-proof, no inbound rules on remotes
      federates: {n_feds}

    federate_configs:
'''
    out += weather_block(shard)
    for i in range(1, sites_per_shard + 1):
        out += site_blocks(shard, i, alias)
    return out


def build(n_machines, sites_per_shard, start, end, sink="json"):
    n_feds_shard = 4 * sites_per_shard + 1
    n_feds_total = n_machines * n_feds_shard
    name = f"cs_s4c_shard_{n_machines}m"
    desc = (f"Paper case study S4c-real — federation-sharded capacity scaling, "
            f"{n_machines} machine(s), {sites_per_shard} sites/shard "
            f"({n_feds_shard} federates/shard), {n_feds_total} federates total.")

    any_remote = n_machines > 1
    lines_pl = [f"PLACEMENT ({n_machines} machine(s), {sites_per_shard} sites/shard):"]
    for shard in range(1, n_machines + 1):
        alias = SHARD_ALIAS[shard]
        where = "manager (local)" if alias is None else f"{alias} (remote, SSH)"
        lines_pl.append(f"#   shard_{shard} -> {where}: weather + {sites_per_shard} sites = {n_feds_shard} federates")
    placement_txt = "\n".join(lines_pl)
    title = f"S4c-REAL — SHARDED CAPACITY ({n_machines} machine(s), {n_feds_total} federates total)"

    out = HEADER.format(title=title, placement=placement_txt, name=name, desc=desc,
                         start=start, end=end, sink=sink, sites=sites_per_shard,
                         n_feds_shard=n_feds_shard, n_machines=n_machines,
                         n_feds_total=n_feds_total)
    if any_remote:
        out += DEPLOYMENT
    out += SYNC
    out += "\nfederations:"
    for shard in range(1, n_machines + 1):
        out += build_shard_federation(shard, sites_per_shard, SHARD_ALIAS[shard])
    return out


def build_ceiling(sites_per_shard, start, end, sink="none"):
    """Single-machine, single-federation, local, zmq_ss — throwaway scenario
    for the ceiling-characterization sweep (task D)."""
    n_feds = 4 * sites_per_shard + 1
    name = f"cs_s4c_ceiling_{sites_per_shard}"
    desc = (f"Ceiling-characterization throwaway: 1 federation, 1 machine (local), "
            f"{sites_per_shard} sites, {n_feds} federates, zmq_ss.")
    title = f"S4c-REAL CEILING SWEEP — {n_feds} federates, single broker, local zmq_ss"
    placement_txt = f"PLACEMENT: everything local (manager), 1 federation, {sites_per_shard} sites = {n_feds} federates"
    out = HEADER.format(title=title, placement=placement_txt, name=name, desc=desc,
                         start=start, end=end, sink=sink, sites=sites_per_shard,
                         n_feds_shard=n_feds, n_machines=1, n_feds_total=n_feds)
    out += SYNC
    out += "\nfederations:"
    out += build_shard_federation(1, sites_per_shard, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-per-shard", type=int, default=8,
                     help="sites per shard/federation; 4 federates each + 1 weather "
                          "(default 8 -> 33 federates/shard, the known-safe zmq_ss ceiling)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-01-08", help="default 1 week of hourly steps (168 ticks)")
    ap.add_argument("--sink", default="json", choices=("json", "parquet", "none"))
    ap.add_argument("--ceiling", type=int, nargs="+", default=None,
                     help="generate throwaway single-machine/single-federation scenarios "
                          "at each given sites-per-shard size instead of the 1/2/3-machine trio")
    args = ap.parse_args()

    here = Path(__file__).parent

    if args.ceiling:
        for n in args.ceiling:
            text = build_ceiling(n, args.start, args.end, args.sink)
            fn = here / f"cs_s4c_ceiling_{n}.yaml"
            fn.write_text(text)
            print(f"wrote {fn}  ({4*n+1} federates)")
        return

    for n_machines in (1, 2, 3):
        text = build(n_machines, args.sites_per_shard, args.start, args.end, args.sink)
        fn = here / f"cs_s4c_shard_{n_machines}m.yaml"
        fn.write_text(text)
        n_feds = n_machines * (4 * args.sites_per_shard + 1)
        print(f"wrote {fn}  ({n_feds} federates total, {n_machines} shard(s) x "
              f"{4*args.sites_per_shard+1})")


if __name__ == "__main__":
    main()
