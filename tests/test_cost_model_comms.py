"""
test_cost_model_comms.py — Phase-D `comms` term of scripts/scaling_study/cost_model.py.

Covers (2026-07-28 pooled-in_per_link_s revision -- CONTRACT.md "comms cost
term"):
  1. Synthetic round-trip: fit() recovers a known per-distance per_edge_s, a
     single POOLED SCALAR in_per_link_s, and per_byte_s from a paired
     control/wired CSV within 10%, with n_edges and max_fed_in varying
     INDEPENDENTLY (the whole point -- a narrow matrix that lets them move
     together reproduces the collinearity that made the (now-superseded)
     per-distance in_per_link_s split unidentifiable).
  2. Backward compatibility: a Part-A-only CSV (no exchange columns at all)
     still fits without raising and yields an all-zero `comms` block, with
     in_per_link_s as the scalar 0.0.
  3. predict() adds the comms term correctly (in_per_link_s as a pooled
     scalar), and a params dict with no "comms" key predicts identically to
     one with an explicit all-zero block.
  4. A params file written in an OLDER (superseded) comms shape --
     fixed_per_tick_s/out_per_link_s from the per-federate-only iteration,
     PLUS a per-distance in_per_link_s dict (the shape this revision
     replaces) -- still loads and predicts without raising: predict()
     accepts both the new scalar and the old per-distance-dict shape for
     in_per_link_s.

Why in_per_link_s was pooled into a single scalar (was previously a
per-distance dict, mirroring per_edge_s): `distance` describes an edge's
ROUTING path (which broker chain carries it), whereas `max_fed_in` prices the
subscriber federate's own per-tick poll-and-deserialise loop -- local CPU
work that cannot know where a value came from, so splitting it per distance
is not physically meaningful. It was also unidentifiable in practice: within
a distance stratum, n_edges and max_fed_in correlate ~0.4-0.5, and the
per-distance split produced a 9.5x-too-high intra_fed coefficient and a
negative cross_fed one, while pooling recovers coefficients matching
hand-computed values. `per_edge_s` remains a per-distance dict -- unchanged.

`scripts/scaling_study/` is not a package (no __init__.py) and lives outside
`src/`, so it is loaded by absolute path via importlib, per project convention
for non-package modules referenced from tests/.

Run: pytest tests/test_cost_model_comms.py -q
"""
import csv
import importlib.util
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_MODEL_PATH = os.path.join(REPO_ROOT, "scripts", "scaling_study", "cost_model.py")

_spec = importlib.util.spec_from_file_location("cost_model", COST_MODEL_PATH)
cost_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cost_model)


# ---------------------------------------------------------------------------
# shared row-building helpers
# ---------------------------------------------------------------------------

# Common non-exchange knobs held constant across every row in these tests so
# a single control key covers every wired row (the paired-delta match key is
# F,N,M,mode,W,core_type,model,work,placement,n_ticks -- none of which vary
# here; only the exchange columns vary between rows).
BASE_ROW = {
    "F": 1, "N": 2, "M": 1, "mode": "seq", "W": "", "core_type": "zmq",
    "model": "heavy_compute_dummy", "work": 1.0, "placement": "local",
    "n_machines": 1, "n_ticks": 100, "repeat": 0,
    "scenario_name": "synthetic", "sim_id": "sim0",
    "setup_s": 0.1, "broker_setup_s": 0.05, "federate_spawn_s": 0.05,
    "sim_wall_s": 0.0, "perf_n_ticks": 100,
    "tick_mean_s": 0.0, "tick_median_s": 0.0, "tick_p95_s": 0.0,
    "failure_mode": "", "peak_rss_mb": 50.0, "cpu_util_pct": 10.0,
    "throughput_inst_steps_s": 100.0,
}

CONTROL_TICK_S = 0.0025  # known constant compute+sync baseline (no wiring)

# Ground-truth comms coefficients this test injects and expects fit() to
# recover (within 10% relative error -- the assertion this task must not
# weaken). per_edge_s covers two distances so the fit must separate distance
# for the edge-routing mechanism; in_per_link_s is a single POOLED SCALAR
# (per-federate inbound-polling cost -- see module docstring for why it is
# not split per distance).
TRUE_PER_EDGE_S = {"intra_fed": 3.9e-6, "cross_fed": 7.5e-6}
TRUE_IN_PER_LINK_S = 2.1e-6
TRUE_PER_BYTE_S = 1.5e-9


def _control_row(**overrides):
    row = dict(BASE_ROW)
    row.update({
        "tick_mean_s": CONTROL_TICK_S,
        "exchange": "none", "distance": "", "fanout": "", "msg_width": 1,
        "freq": 1, "causality": "", "n_edges": 0, "n_subs": 0,
        "max_fed_in": 0, "max_fed_out": 0,
    })
    row.update(overrides)
    return row


def _wired_row(distance, n_edges, max_fed_in, msg_width, freq, **overrides):
    delta = (
        TRUE_PER_EDGE_S[distance] * n_edges
        + TRUE_IN_PER_LINK_S * max_fed_in
        + TRUE_PER_BYTE_S * 8 * msg_width * n_edges / freq
    )
    row = dict(BASE_ROW)
    row.update({
        "tick_mean_s": CONTROL_TICK_S + delta,
        "exchange": "on", "distance": distance, "fanout": "1to1",
        "msg_width": msg_width, "freq": freq, "causality": "same_step",
        # max_fed_out/n_subs are no longer regressed on -- set to something
        # plausible but uncorrelated-in-a-load-bearing-way so the test can't
        # accidentally pass via a stale/unused column.
        "n_edges": n_edges, "n_subs": max_fed_in,
        "max_fed_in": max_fed_in, "max_fed_out": n_edges + max_fed_in,
    })
    row.update(overrides)
    return row


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# 1. synthetic round-trip
# ---------------------------------------------------------------------------

def test_fit_recovers_known_comms_coefficients(tmp_path):
    rows = [_control_row(), _control_row(sim_id="sim0b")]

    # Vary distance x n_edges x max_fed_in x msg_width x freq. Critically,
    # n_edges and max_fed_in must NOT move together in every row -- this is
    # exactly the collinearity that made the (now-superseded) per-distance
    # in_per_link_s split unidentifiable: within a distance stratum,
    # n_edges/max_fed_in correlated ~0.4-0.5 in the real data, which produced
    # a 9.5x-too-high intra_fed coefficient and a negative cross_fed one.
    # Pooling in_per_link_s into a single scalar removes the need to identify
    # it per distance, but this test still varies the two independently so it
    # can't accidentally pass via a reintroduced correlation. Deltas here span
    # from ~tens of us
    # (small n_edges/max_fed_in) to > 10ms (n_edges=2000+), i.e. more than
    # two orders of magnitude, so the weighted fit is meaningfully exercised.
    combos = [
        # (distance, n_edges, max_fed_in, msg_width, freq)
        ("intra_fed", 4, 1, 1, 1),
        ("intra_fed", 8, 2, 1, 1),
        ("intra_fed", 64, 2, 1, 1),      # n_edges up, max_fed_in unchanged
        ("intra_fed", 64, 32, 1, 1),     # n_edges unchanged, max_fed_in up
        ("intra_fed", 1024, 8, 1, 1),    # large n_edges, small max_fed_in
        ("intra_fed", 16, 64, 1, 1),     # small n_edges, large max_fed_in
        ("intra_fed", 500, 20, 4, 1),
        ("intra_fed", 2000, 10, 1, 1),
        ("intra_fed", 100, 100, 2, 2),
        ("cross_fed", 4, 1, 1, 1),
        ("cross_fed", 8, 2, 1, 1),
        ("cross_fed", 64, 2, 1, 1),
        ("cross_fed", 64, 32, 1, 1),
        ("cross_fed", 1024, 8, 1, 1),
        ("cross_fed", 16, 64, 1, 1),
        ("cross_fed", 500, 20, 4, 1),
        ("cross_fed", 2000, 10, 1, 1),
        ("cross_fed", 100, 100, 2, 2),
    ]
    for i, (distance, n_edges, max_fed_in, msg_width, freq) in enumerate(combos):
        rows.append(_wired_row(distance, n_edges, max_fed_in, msg_width, freq,
                                sim_id=f"wired{i}"))

    csv_path = tmp_path / "bench_synthetic.csv"
    _write_csv(csv_path, rows, cost_model.CSV_FIELDS)

    params = cost_model.fit(str(csv_path))

    comms = params["comms"]
    assert set(comms["per_edge_s"].keys()) == {"intra_fed", "cross_fed", "cross_machine"}
    assert isinstance(comms["in_per_link_s"], (int, float))

    def rel_err(got, expected):
        return abs(got - expected) / abs(expected)

    assert rel_err(comms["per_edge_s"]["intra_fed"], TRUE_PER_EDGE_S["intra_fed"]) < 0.10, comms
    assert rel_err(comms["per_edge_s"]["cross_fed"], TRUE_PER_EDGE_S["cross_fed"]) < 0.10, comms
    assert rel_err(comms["in_per_link_s"], TRUE_IN_PER_LINK_S) < 0.10, comms
    assert rel_err(comms["per_byte_s"], TRUE_PER_BYTE_S) < 0.10, comms
    # cross_machine had no data in this csv -- must default to 0.0.
    assert comms["per_edge_s"]["cross_machine"] == 0.0

    print("\nrecovered vs injected comms coefficients:")
    print(f"  per_edge_s.intra_fed: injected={TRUE_PER_EDGE_S['intra_fed']:.4e}  "
          f"recovered={comms['per_edge_s']['intra_fed']:.4e}")
    print(f"  per_edge_s.cross_fed: injected={TRUE_PER_EDGE_S['cross_fed']:.4e}  "
          f"recovered={comms['per_edge_s']['cross_fed']:.4e}")
    print(f"  in_per_link_s (pooled): injected={TRUE_IN_PER_LINK_S:.4e}  "
          f"recovered={comms['in_per_link_s']:.4e}")
    print(f"  per_byte_s:           injected={TRUE_PER_BYTE_S:.4e}  "
          f"recovered={comms['per_byte_s']:.4e}")
    print(f"  notes: {params['notes']}")


# ---------------------------------------------------------------------------
# 2. backward compatibility: Part-A-only CSV (no exchange columns at all)
# ---------------------------------------------------------------------------

def test_fit_backward_compatible_with_part_a_csv(tmp_path):
    part_a_fields = [f for f in cost_model.CSV_FIELDS if f not in
                      ("exchange", "distance", "fanout", "msg_width", "freq",
                       "causality", "n_edges", "n_subs", "max_fed_in", "max_fed_out")]
    rows = []
    for n in (2, 4):
        row = dict(BASE_ROW)
        row.pop("exchange", None); row.pop("distance", None)
        row["N"] = n
        row["tick_mean_s"] = 0.001 + 0.0001 * n
        rows.append({k: v for k, v in row.items() if k in part_a_fields})

    csv_path = tmp_path / "bench_part_a.csv"
    _write_csv(csv_path, rows, part_a_fields)

    params = cost_model.fit(str(csv_path))  # must not raise

    assert params["comms"] == {
        "per_edge_s": {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
        "in_per_link_s": 0.0,
        "per_byte_s": 0.0,
    }


# ---------------------------------------------------------------------------
# 3. predict() wiring
# ---------------------------------------------------------------------------

def _base_config(**overrides):
    config = {
        "F": 1, "N": 2, "M": 1, "mode": "seq", "W": None, "core_type": "zmq",
        "model": "heavy_compute_dummy", "work": 1.0, "placement": "local",
        "n_ticks": 100,
    }
    config.update(overrides)
    return config


def test_predict_adds_comms_term_matching_analytic_formula():
    params = {
        "c": {"heavy_compute_dummy": {"a": 1e-5, "b": 0.0}},
        "s": {"zmq": {"s0": 5e-5, "s1": 1e-6}},
        "O_par": 0.0,
        "rss_per_instance_mb": {},
        "rtt_s": 0.0,
        "comms": {
            "per_edge_s": {"intra_fed": 3.9e-6, "cross_fed": 7.5e-6, "cross_machine": 0.0},
            "in_per_link_s": 2.1e-6,
            "per_byte_s": 1.5e-9,
        },
        "notes": "",
    }

    off_config = _base_config(exchange="none")
    on_config = _base_config(exchange="on", distance="intra_fed", n_edges=64,
                              max_fed_in=5, msg_width=4, freq=2)

    off_result = cost_model.predict(off_config, params)
    on_result = cost_model.predict(on_config, params)

    expected_comms_term = (
        3.9e-6 * 64
        + 2.1e-6 * 5
        + 1.5e-9 * 8 * 4 * 64 / 2
    )
    assert on_result["comms_s"] - off_result["comms_s"] == pytest.approx(expected_comms_term, rel=1e-9)
    assert on_result["T_tick_s"] - off_result["T_tick_s"] == pytest.approx(expected_comms_term, rel=1e-9)


def test_predict_without_comms_key_matches_all_zero_comms_block():
    params_no_comms = {
        "c": {"heavy_compute_dummy": {"a": 1e-5, "b": 0.0}},
        "s": {"zmq": {"s0": 5e-5, "s1": 1e-6}},
        "O_par": 0.0,
        "rss_per_instance_mb": {},
        "rtt_s": 0.0,
        "notes": "",
    }
    params_zero_comms = dict(params_no_comms)
    params_zero_comms["comms"] = {
        "per_edge_s": {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
        "in_per_link_s": 0.0,
        "per_byte_s": 0.0,
    }

    on_config = _base_config(exchange="on", distance="intra_fed", n_edges=64,
                              max_fed_in=5, msg_width=4, freq=2)

    result_no_comms = cost_model.predict(on_config, params_no_comms)
    result_zero_comms = cost_model.predict(on_config, params_zero_comms)

    assert result_no_comms == result_zero_comms


# ---------------------------------------------------------------------------
# 4. old (superseded) comms shape still loads and predicts without raising
# ---------------------------------------------------------------------------

def test_predict_with_old_comms_shape_does_not_raise():
    """A params file fitted before the wide-matrix revision (per-federate-only
    shape: fixed_per_tick_s + in/out_per_link_s[distance]*max_fed_in/out) is
    simply missing the current keys (per_edge_s) and must predict a
    zero/partial comms contribution -- never raise. This also doubles as the
    in_per_link_s dict-vs-scalar backward-compat test: this old shape still
    stores in_per_link_s as a per-distance dict (the shape superseded by the
    pooled scalar), and predict() must still index it by distance when it
    sees a dict, rather than treating it as the new scalar."""
    params_old_shape = {
        "c": {"heavy_compute_dummy": {"a": 1e-5, "b": 0.0}},
        "s": {"zmq": {"s0": 5e-5, "s1": 1e-6}},
        "O_par": 0.0,
        "rss_per_instance_mb": {},
        "rtt_s": 0.0,
        "comms": {
            "fixed_per_tick_s": 4.0e-6,
            "in_per_link_s": {"intra_fed": 2.0e-5, "cross_fed": 6.0e-5, "cross_machine": 0.0},
            "out_per_link_s": {"intra_fed": 1.2e-5, "cross_fed": 3.0e-5, "cross_machine": 0.0},
            "per_byte_s": 1.5e-9,
        },
        "notes": "",
    }

    off_config = _base_config(exchange="none")
    on_config = _base_config(exchange="on", distance="intra_fed", n_edges=64,
                              max_fed_in=5, max_fed_out=3, msg_width=4, freq=2)

    off_result = cost_model.predict(off_config, params_old_shape)
    on_result = cost_model.predict(on_config, params_old_shape)  # must not raise

    # per_edge_s is absent -> contributes 0; in_per_link_s IS present under
    # the same key name and is still applied against max_fed_in (unchanged
    # meaning across both shapes); per_byte_s is present and IS applied (now
    # against n_edges instead of max_fed_in) -- a "partial" contribution.
    expected_comms_term = (
        2.0e-5 * 5
        + 1.5e-9 * 8 * 4 * 64 / 2
    )
    assert on_result["comms_s"] - off_result["comms_s"] == pytest.approx(expected_comms_term, rel=1e-9)


def test_fit_loads_real_old_shape_params_file_without_raising():
    """Verify against the actual on-disk file named in the task: a params
    file with NO 'comms' key at all (phase5_clean_fit_params.json) must load
    and predict zero comms contribution without raising."""
    import json

    old_params_path = os.path.join(
        REPO_ROOT, "scripts", "scaling_study", "findings", "phase5_clean_fit_params.json")
    with open(old_params_path) as f:
        old_params = json.load(f)

    assert "comms" not in old_params  # sanity: this fixture predates Phase-D comms

    on_config = _base_config(exchange="on", distance="intra_fed", n_edges=64,
                              max_fed_in=5, msg_width=4, freq=2)
    off_config = _base_config(exchange="none")

    on_result = cost_model.predict(on_config, old_params)  # must not raise
    off_result = cost_model.predict(off_config, old_params)

    assert on_result["comms_s"] == pytest.approx(off_result["comms_s"])
