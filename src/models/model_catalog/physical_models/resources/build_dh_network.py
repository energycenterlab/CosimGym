"""
build_dh_network.py

One-shot builder for a parametric district-heating (DH) water network used by the
`dh_district_jan_base` co-simulation scenario. Emits a pandapipes JSON case file
that `pandapipes_grid` loads via `case_file_format: "json"`.

Topology (N heat consumers, default N=10):

    j_flow (0) ──circ_pump_const_pressure──┐            supply manifold @ t_flow_k
        │                                  │
        ├── heat_consumer i ──> jc_i ── return_pipe_i ──┐
        │   (qext_w, controlled_mdot)                   │
        └── ... (N consumers in parallel) ...           │
                                                        v
    j_return (1) <──────────────────────────────────────┘  return manifold

Junction index convention (deterministic, so the scenario YAML can reference them):
    0        -> j_flow    (supply manifold, ~ t_flow_k)
    1        -> j_return  (return manifold, mixed return temperature)
    2 .. 2+N-1 -> jc_i    (consumer i outlet temperature)

heat_consumer index i (0-based) carries column ``qext_w`` — this is the input
overwritten each co-sim step by building i's ``Q_heating`` (W). Each consumer is
created with exactly two set-points (``qext_w`` + ``controlled_mdot_kg_per_s``) as
pandapipes requires; the mass flow is held fixed and the return temperature floats
with the load.

Run:  python build_dh_network.py            # writes dh_network_10.json + self-test
"""

import os

import pandapipes as ppipe


# --- network design parameters --------------------------------------------------
N_CONSUMERS = 10
T_FLOW_K = 353.15        # 80 degC supply temperature at the plant
P_FLOW_BAR = 6.0         # absolute pressure at the supply manifold
PLIFT_BAR = 3.0          # pump pressure lift
PN_BAR = 6.0             # nominal/initial junction pressure
TFLUID_INIT_K = 333.15   # 60 degC initial fluid temperature (solver seed)

# per-consumer nominal mass flow (kg/s); heterogeneous to match varied buildings
NOMINAL_MDOT = [0.40, 0.30, 0.50, 0.35, 0.60, 0.25, 0.45, 0.32, 0.55, 0.28]
RETURN_PIPE_LENGTH_KM = 0.05      # 50 m return branch per consumer
RETURN_PIPE_DIAMETER_M = 0.05     # 50 mm
INIT_QEXT_W = 5000.0              # seed load (overwritten live by the buildings)

OUT_NAME = "dh_network_10.json"


def build_network(n=N_CONSUMERS, mdots=None):
    if mdots is None:
        mdots = NOMINAL_MDOT
    if len(mdots) < n:
        # fall back to a constant if not enough heterogeneous values provided
        mdots = (mdots + [0.4] * n)[:n]

    net = ppipe.create_empty_network(fluid="water")

    # --- manifold junctions: indices 0 (supply) and 1 (return) -----------------
    j_flow = ppipe.create_junction(net, pn_bar=PN_BAR, tfluid_k=TFLUID_INIT_K,
                                   name="supply_manifold")      # -> index 0
    j_return = ppipe.create_junction(net, pn_bar=PN_BAR, tfluid_k=TFLUID_INIT_K,
                                     name="return_manifold")    # -> index 1

    # --- heat plant: circulation pump sets pressure, lift and supply temp ------
    ppipe.create_circ_pump_const_pressure(
        net, return_junction=j_return, flow_junction=j_flow,
        p_flow_bar=P_FLOW_BAR, plift_bar=PLIFT_BAR, t_flow_k=T_FLOW_K,
        name="dh_plant")

    # --- N consumers in parallel: heat_consumer + return pipe ------------------
    for i in range(n):
        jc = ppipe.create_junction(net, pn_bar=PN_BAR, tfluid_k=TFLUID_INIT_K,
                                   name=f"consumer_{i}_out")     # -> index 2+i
        # heat_consumer index i (0-based, matches creation order)
        ppipe.create_heat_consumer(
            net, from_junction=j_flow, to_junction=jc,
            qext_w=INIT_QEXT_W, controlled_mdot_kg_per_s=float(mdots[i]),
            name=f"consumer_{i}")
        ppipe.create_pipe_from_parameters(
            net, from_junction=jc, to_junction=j_return,
            length_km=RETURN_PIPE_LENGTH_KM, diameter_m=RETURN_PIPE_DIAMETER_M,
            k_mm=0.05, name=f"return_pipe_{i}")

    return net


def self_test(net):
    ppipe.pipeflow(net, mode="bidirectional", max_iter_hyd=20, max_iter_therm=20,
                   friction_model="nikuradse")
    print("Pipeflow converged.")
    rj = net.res_junction
    print(f"  supply manifold (j0) t = {rj.t_k.iloc[0] - 273.15:6.2f} degC")
    print(f"  return manifold (j1) t = {rj.t_k.iloc[1] - 273.15:6.2f} degC")
    print(f"  consumer outlet temps  = "
          f"{[round(t - 273.15, 1) for t in rj.t_k.iloc[2:].tolist()]} degC")
    print(f"  res_heat_consumer cols = {list(net.res_heat_consumer.columns)}")
    print(f"  total demand qext      = "
          f"{net.heat_consumer.qext_w.sum() / 1000:.1f} kW")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, OUT_NAME)
    net = build_network()
    print(f"Built DH network: {len(net.junction)} junctions, "
          f"{len(net.heat_consumer)} heat consumers, {len(net.pipe)} pipes.")
    self_test(net)
    ppipe.to_json(net, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
