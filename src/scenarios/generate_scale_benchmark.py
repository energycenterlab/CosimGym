#!/usr/bin/env python3
"""Generate the local/distributed scaling benchmark scenario pair.

Both scenarios are emitted from ONE spec so they are byte-identical except for
placement (`deployment:` + `host:`). That is the whole point: if the two files
drifted in federate count, instance count, horizon or wiring, the timing
comparison would measure the drift instead of the distribution.

Topology — a district of N "sites", each a building with rooftop PV, all sharing
one weather federate (4 federates per site + 1 weather = 4N+1 total):

    weather ──DryBulb/GloHorzRad/DifHorzRad──┬─> pv_i    (pv_dest)      -> PV_power
                                             ├─> bldg_i  (simple_building)
                                             └─> hp_i    (simple_heatpump)

    pid_i ──modulation──> hp_i ──Q_heat──> bldg_i ──T_indoor──> pid_i
      ^ the bldg->pid->hp->bldg loop is broken by causality: next_step on pid's
        T_indoor subscription (same trick as bui_hp_test_base.yaml).

PLACEMENT (distributed variant) is proportional to CORE COUNT, not federate count:
    manager ipazia      112 cores -> 64% of sites
    cloud1  (machine_a)  32 cores -> 18%
    cloud5  (machine_b)  32 cores -> 18%
HELICS is lockstep, so the slowest machine gates every timestep. An even 1/3
split would hand a 32-core box the work a 112-core box does ~3.5x faster and the
distributed run would LOSE. Weather stays on the manager (every site reads it).

MEASURED (8 sites / 33 federates, 720 steps, 3 repeats each, --sink none):
    local        setup ~0.58s   sim ~5.25s
    distributed  setup ~2.44s   sim ~7.72s      => distribution ~1.47x SLOWER
  (--sink json adds the same result-writing cost to BOTH scenarios, so the
   comparison stays fair; it just shifts both numbers up.)

That is the expected, honest result and NOT a bug. Two reasons, both structural:
  1. SYNC-BOUND: these models are analytic (~µs/step) while HELICS sync costs
     ~ms/step, so the run measures per-tick synchronisation, and distributing it
     only adds LAN latency to every tick. To show a SPEEDUP the per-step compute
     must dominate the sync cost — use heavy_compute_dummy (benchmark_parallel_*).
  2. HARDWARE: the manager has 112 cores; the two remotes have 32 each. Even a
     perfectly compute-bound job could only gain 176/112 = 1.57x, and only with
     the core-proportional split used here.

!! zmq_ss FEDERATE CEILING — SUPERSEDED, see scaling_study/findings/README.md !!
  HISTORICAL observation (kept for provenance), measured on this LAN on some days:
    zmq_ss  17 federates: OK OK OK        zmq  65 federates: OK OK OK
    zmq_ss  33 federates: OK OK OK        zmq  89 federates: OK OK OK
    zmq_ss  49 federates: FAIL FAIL OK   <-- flaky; single socket saturates
    zmq_ss  65 federates: FAIL
  Failures appeared as every federate dying with "[-101] lost comms" after a ~52s
  timeout.
  UPDATE (2026-07-24, Phase 2/5): this ceiling does NOT reproduce reliably. A
  calibrated sweep passed zmq_ss/distributed to N=89, and N=200 on ONE broker
  passed cleanly — i.e. the failures above were a TRANSIENT LAN condition, not an
  intrinsic zmq_ss/HELICS limit (this file's git history also flags the behaviour
  as non-deterministic). The only DETERMINISTIC failure found was an unrelated SSH
  ControlPath / AF_UNIX 108-byte path-length bug at N>=112, since FIXED. Do not
  treat "~33 federates" as a hard cap. Full analysis + numbers:
  scripts/scaling_study/findings/phase2_ceiling.md.
  zmq_ss is still REQUIRED when remotes are behind NAT (see A0 in
  docs/user_guide/multi_machine_test_walkthrough.md). Both scenarios must use the
  SAME core_type to stay comparable, which is why the local twin uses zmq_ss too
  even though plain zmq would scale to 89 there.

Usage:
    python src/scenarios/generate_scale_benchmark.py [--sites N] [--end YYYY-MM-DD]
"""
import argparse
from pathlib import Path

# Core counts measured on the three machines; placement weights derive from these.
CORES = {"manager": 112, "machine_a": 32, "machine_b": 32}

HEADER = """\
# ============================================================================
# GENERATED FILE — DO NOT EDIT BY HAND.
# Source: src/scenarios/generate_scale_benchmark.py   (re-run it to regenerate)
#
# {title}
#
# {placement}
#
# Paired with: {twin}
# The two scenarios are identical except for placement, so any timing difference
# is attributable to distribution alone.
#
# NOTE: analytic models (~µs/step) vs HELICS sync (~ms/step) => this pair is
# SYNC-BOUND. It measures the COST of distribution, not a speedup. For a
# compute-bound comparison use heavy_compute_dummy (see benchmark_parallel_*).
# ============================================================================

version: "1.0.0"
name: "{name}"
scenario_description: "{desc}"

start_time: "{start}T00:00:00"
end_time: "{end}T00:00:00"
log_level: ERROR          # quiet: {nfeds} federates would otherwise flood the log

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


def host_line(alias):
    """`host:` only in the distributed variant; None keeps the federate local."""
    return f'        host: "{alias}"\n' if alias else ""


def weather_block():
    return '''
      weather_federate:
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
            prefix: "weather"
            parallel_execution: false
          parameters:
            csv_path: "model_catalog/physical_models/resources/weather_data_bj.csv"
            skip_rows: 0
          init_state:
            DryBulb: 3.0
            GloHorzRad: 0.0
            DifHorzRad: 0.0
'''


def site_blocks(i, alias):
    """The 4 federates of site i. `alias` is None (local) or a machine alias."""
    h = host_line(alias)
    s = f"{i:02d}"
    return f'''
      pv_{s}:
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
                '0': [weather_federate.0/DryBulb]
            - key: "GHI"
              type: "double"
              units: "W/m²"
              targets:
                '0': [weather_federate.0/GloHorzRad]
            - key: "DHI"
              type: "double"
              units: "W/m²"
              targets:
                '0': [weather_federate.0/DifHorzRad]
          publishes:
            - key: "PV_power"
              type: "double"
              units: "W"
        model_configs:
          instantiation:
            model_name: "pv_dest"
            n_instances: 1
            prefix: "pv"
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

      bldg_{s}:
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
                '0': [weather_federate.0/DryBulb]
            - key: "Q_heat"
              type: "double"
              units: "W"
              targets:
                '0': [hp_{s}.0/Q_heat]
          publishes:
            - key: "T_indoor"
              type: "double"
              units: "°C"
        model_configs:
          instantiation:
            model_name: "simple_building"
            n_instances: 1
            prefix: "bldg"
            parallel_execution: false
          parameters:
            thermal_capacitance: 1000000.0
            thermal_resistance: 0.005
            T_initial: 18.0
          init_state:
            T_ext: 3.0
            Q_heat: 0.0
            T_indoor: 18.0

      hp_{s}:
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
                '0': [weather_federate.0/DryBulb]
            - key: "modulation"
              type: "double"
              units: "-"
              targets:
                '0': [pid_{s}.0/modulation]
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
            prefix: "hp"
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

      pid_{s}:
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
                '0': [bldg_{s}.0/T_indoor]
              causality: "next_step"     # breaks the bldg->pid->hp->bldg loop
          publishes:
            - key: "modulation"
              type: "double"
              units: "-"
        model_configs:
          instantiation:
            model_name: "simple_pid_controller"
            n_instances: 1
            prefix: "pid"
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


def placement(n_sites, distributed):
    """Site -> machine alias. Proportional to cores; weather always on manager."""
    if not distributed:
        return {i: None for i in range(1, n_sites + 1)}
    total = sum(CORES.values())
    n_a = round(n_sites * CORES["machine_a"] / total)
    n_b = round(n_sites * CORES["machine_b"] / total)
    out = {}
    for i in range(1, n_sites + 1):
        if i <= n_a:
            out[i] = "machine_a"
        elif i <= n_a + n_b:
            out[i] = "machine_b"
        else:
            out[i] = None          # manager
    return out


def build(n_sites, distributed, start, end, port, sink):
    place = placement(n_sites, distributed)
    n_feds = 4 * n_sites + 1
    name = "benchmark_scale_distributed" if distributed else "benchmark_scale_local"
    twin = "benchmark_scale_local.yaml" if distributed else "benchmark_scale_distributed.yaml"

    if distributed:
        n_mgr = sum(1 for v in place.values() if v is None)
        n_a = sum(1 for v in place.values() if v == "machine_a")
        n_b = sum(1 for v in place.values() if v == "machine_b")
        desc = f"Scaling benchmark: {n_sites} building+PV sites, {n_feds} federates, distributed over 3 hosts"
        pl = (f"PLACEMENT (proportional to cores 112:32:32):\n"
              f"#   manager (112 cores): weather + {n_mgr} sites = {4*n_mgr+1} federates\n"
              f"#   machine_a (32)     : {n_a} sites = {4*n_a} federates\n"
              f"#   machine_b (32)     : {n_b} sites = {4*n_b} federates")
        title = f"SCALING BENCHMARK — DISTRIBUTED ({n_feds} federates over 3 machines)"
    else:
        desc = f"Scaling benchmark: {n_sites} building+PV sites, {n_feds} federates, all local"
        pl = f"PLACEMENT: everything on the manager ({n_feds} federates, 112 cores)"
        title = f"SCALING BENCHMARK — ALL LOCAL ({n_feds} federates, single machine)"

    out = HEADER.format(title=title, placement=pl, twin=twin, name=name,
                        desc=desc, start=start, end=end, nfeds=n_feds, sink=sink)
    if distributed:
        out += DEPLOYMENT
    out += SYNC
    out += f'''

federations:
  federation_1:
    broker_config:
      core_type: "zmq_ss"        # single socket: NAT-proof, no inbound rules on remotes
      port: {port}
      federates: {n_feds}

    federate_configs:
'''
    out += weather_block()
    for i in range(1, n_sites + 1):
        out += site_blocks(i, place[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=8,
                    help="building+PV sites; 4 federates each (default 8 -> 33 federates, "
                         "the largest size zmq_ss runs reliably here — see module docstring)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-01-02", help="default 2024-01-31 => 720 hourly steps")
    ap.add_argument("--sink", default="json", choices=("json", "parquet", "none"),
                    help="memory_config.sink. Default 'json' => results land in "
                         "results/<scenario>/<sim_id>/ (and are rsynced back from the remotes). "
                         "Use 'none' for a pure timing run with no disk I/O in the measurement — "
                         "both scenarios do the same I/O either way, so the comparison stays fair.")
    args = ap.parse_args()

    if 4 * args.sites + 1 > 40:
        print(f"WARNING: {4*args.sites+1} federates is above the ~33 that zmq_ss runs reliably\n"
              f"         on this LAN; expect intermittent '[-101] lost comms' after ~52s.\n"
              f"         See the module docstring for the measured ceiling.\n")

    here = Path(__file__).parent
    for dist, port in ((False, 23404), (True, 23404)):
        text = build(args.sites, dist, args.start, args.end, port, args.sink)
        fn = here / (f"benchmark_scale_{'distributed' if dist else 'local'}.yaml")
        fn.write_text(text)
        print(f"wrote {fn}  ({4*args.sites+1} federates)")

    p = placement(args.sites, True)
    print(f"\ndistributed placement of {args.sites} sites:")
    print(f"  manager  : {sum(1 for v in p.values() if v is None)} sites + weather")
    print(f"  machine_a: {sum(1 for v in p.values() if v == 'machine_a')} sites")
    print(f"  machine_b: {sum(1 for v in p.values() if v == 'machine_b')} sites")


if __name__ == "__main__":
    main()
