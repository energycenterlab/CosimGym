#!/usr/bin/env python3
"""Cost-model fitter + recommender for the CosimGym scaling study (D4).

Implements the theoretical framework in `docs/future_and_TODOs/scaling_study_plan.md`
Section 1:

    T_sim  = n_ticks * T_tick
    T_tick = max_m( compute_m + sync_m + comms_m )     # max over machines

    compute (per federate):
        seq -> M * c(model, work)
        par -> ceil(M / W) * c(model, work) + O_par
    sync (per federate, shared per broker):
        s(N, core_type) = s0[core_type] + s1[core_type] * N
    comms (per federate, only if placement is remote):
        rtt_s

Three subcommands (CONTRACT.md D4):
    fit       <bench_csv> [--out results/scaling/fit_params.json]
    predict   --params <json> --F --N --M --mode --W --core-type --model --work [--machines]
    recommend --scenario <spec.json> --machines <machines.json> --params <json>

The `fit` subcommand reads the D2 bench CSV (locked schema, see CONTRACT.md
"Locked CSV schema") and writes the locked fitted-params JSON (see CONTRACT.md
"Locked fitted-params JSON"). It is deliberately robust to a THIN csv (few
rows, little variation across knobs): whatever is identifiable from the data
is fit via `numpy.linalg.lstsq`; everything else falls back to a documented
default and is called out in the `notes` field. It never raises on thin input.

No sklearn/scipy dependency -- plain least squares via `numpy.linalg.lstsq`.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = REPO_ROOT / "results" / "scaling" / "fit_params.json"

# Locked CSV schema (CONTRACT.md D2) -- kept here only for reference/validation,
# not enforced strictly (csv.DictReader already gives us the header).
CSV_FIELDS = [
    "F", "N", "M", "mode", "W", "core_type", "model", "work", "placement",
    "n_machines", "n_ticks", "repeat",
    "scenario_name", "sim_id",
    "setup_s", "broker_setup_s", "federate_spawn_s", "sim_wall_s",
    "perf_n_ticks", "tick_mean_s", "tick_median_s", "tick_p95_s",
    "failure_mode", "peak_rss_mb", "cpu_util_pct", "throughput_inst_steps_s",
    # Phase-D extension (2026-07-28): appended, never inserted -- absent
    # entirely in every Part-A CSV. See DEFAULT_EXCHANGE / load_bench_csv.
    "exchange", "distance", "fanout", "msg_width", "freq", "causality",
    "n_edges", "n_subs", "max_fed_in", "max_fed_out",
]

# Per-broker federate ceiling defaults (Section 2 of the plan: the zmq_ss
# ~33-federate ceiling hypothesis). Overridable via --ceiling-zmq-ss / recommend
# scenario spec. "Large" stands in for "effectively unbounded" for zmq/tcp.
DEFAULT_CEILING = {
    "zmq": 100_000,
    "tcp": 100_000,
    "zmq_ss": 33,
    "tcp_ss": 33,
}


# ---------------------------------------------------------------------------
# helpers: typed CSV load
# ---------------------------------------------------------------------------

def _to_float(v):
    if v is None or v == "":
        return None
    return float(v)


def _to_int(v):
    if v is None or v == "":
        return None
    return int(float(v))


# Phase-D exchange columns (CONTRACT.md "Phase-D extension") -- appended after
# throughput_inst_steps_s. Every Part-A CSV lacks them entirely; DEFAULT_EXCHANGE
# gives the "no wiring" values used both when the column is absent and when a
# row leaves it blank.
DEFAULT_EXCHANGE = {
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
}


def load_bench_csv(path):
    """Read the D2 bench CSV into a list of dicts with proper python types.

    Numeric knob/metric columns are cast to float/int (None when blank);
    string columns kept as-is. Robust to a thin file (few rows / few distinct
    values per column) -- never raises on missing optional columns.

    Phase-D exchange columns (`exchange,distance,fanout,msg_width,freq,
    causality,n_edges,n_subs,max_fed_in,max_fed_out`) are optional: absent
    entirely (every Part-A CSV) or blank per-row both default per
    DEFAULT_EXCHANGE (exchange=none, n_edges=n_subs=max_fed_in=max_fed_out=0,
    msg_width=freq=1).
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for k in ("F", "N", "M", "n_machines", "n_ticks", "repeat", "perf_n_ticks"):
                if k in row:
                    row[k] = _to_int(row[k])
            for k in ("W",):
                if k in row:
                    row[k] = _to_int(row[k]) if row[k] not in (None, "") else None
            for k in ("work", "setup_s", "broker_setup_s", "federate_spawn_s",
                      "sim_wall_s", "tick_mean_s", "tick_median_s", "tick_p95_s",
                      "peak_rss_mb", "cpu_util_pct", "throughput_inst_steps_s"):
                if k in row:
                    row[k] = _to_float(row[k])
            row["failure_mode"] = raw.get("failure_mode") or None
            # Phase-D exchange columns: typed + defaulted (missing column or
            # blank value both fall back to DEFAULT_EXCHANGE).
            for k in ("exchange", "distance", "fanout", "causality"):
                v = row.get(k)
                row[k] = v if v not in (None, "") else DEFAULT_EXCHANGE[k]
            for k in ("msg_width", "freq", "n_edges", "n_subs", "max_fed_in", "max_fed_out"):
                v = row.get(k)
                iv = _to_int(v) if v not in (None, "") else None
                row[k] = iv if iv is not None else DEFAULT_EXCHANGE[k]
            rows.append(row)
    return rows


def _successful(rows):
    return [r for r in rows if not r.get("failure_mode")]


def _lstsq(X, y):
    """Thin wrapper: numpy.linalg.lstsq via SVD -- handles rank-deficient X
    (thin/degenerate data) by returning the minimum-norm solution instead of
    raising. Returns the coefficient vector."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _r_squared(y, pred):
    """Coefficient of determination. 1.0 for a perfect fit (or a degenerate
    all-equal-y sample with zero residual); 0.0 for the degenerate all-equal-y
    sample with nonzero residual (mean-of-y R^2 is undefined -> treat as no
    explanatory power rather than raising)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return 0.0
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _r_squared_weighted(y, pred, weights):
    """Weighted coefficient of determination -- same degenerate-input handling
    as `_r_squared`, but both the residual and total sum-of-squares (and the
    mean of y they're centered on) are weighted. Used for the comms fit,
    which is solved by weighted least squares (see `fit()` section 6) so the
    reported R^2 should reflect the same weighting, not plain OLS R^2."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(y) == 0:
        return 0.0
    wsum = float(np.sum(weights))
    if wsum == 0.0:
        return 0.0
    ybar = float(np.sum(weights * y) / wsum)
    ss_res = float(np.sum(weights * (y - pred) ** 2))
    ss_tot = float(np.sum(weights * (y - ybar) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def fit(bench_csv, notes_extra=None):
    """Fit c(model,work), s(N,core_type), O_par, rss_per_instance, rtt_s from
    a D2 bench CSV. Returns the locked fitted-params dict. Never raises on
    thin input -- falls back to documented defaults per-parameter and records
    the method actually used in `notes`.
    """
    notes = []
    all_rows = load_bench_csv(bench_csv)
    rows = _successful(all_rows)
    if not rows:
        notes.append("no successful rows in bench_csv -- all params are defaults.")
        return _empty_params("; ".join(notes))

    seq_rows = [r for r in rows if r["mode"] == "seq" and r["M"]]
    par_rows = [r for r in rows if r["mode"] == "par" and r["M"] and r["W"]]

    # --- 1. sync s(N, core_type): s0 + s1*N -------------------------------
    # Cleanest available subset: seq rows only (no O_par confound), grouped
    # by core_type. Regress tick_mean_s on [1, N, M] to hold the (unknown,
    # lumped-across-models) compute contribution roughly constant per M while
    # extracting the N-dependence. If N never varies for a core_type (thin
    # data), s1 can't be identified -> falls back to 0 and s0 is estimated by
    # extrapolating tick_mean_s to M=0 at the fixed N (documented below).
    s_params = {}
    core_types = sorted({r["core_type"] for r in seq_rows if r.get("core_type")})
    for ct in core_types:
        ct_rows = [r for r in seq_rows if r["core_type"] == ct]
        Ns = sorted({r["N"] for r in ct_rows})
        Ms = sorted({r["M"] for r in ct_rows})
        if len(Ns) >= 2:
            X = [[1.0, r["N"], r["M"]] for r in ct_rows]
            y = [r["tick_mean_s"] for r in ct_rows]
            s0, s1, _k = _lstsq(X, y)
            method = f"regressed tick_mean_s ~ 1 + N + M over {len(ct_rows)} seq rows (N varies: {Ns})"
        elif len(Ms) >= 2:
            X = [[1.0, r["M"]] for r in ct_rows]
            y = [r["tick_mean_s"] for r in ct_rows]
            s0, _k = _lstsq(X, y)
            s1 = 0.0
            method = (f"N constant ({Ns[0] if Ns else '?'}) for core_type={ct}; "
                      f"extrapolated tick_mean_s to M=0 over {len(ct_rows)} seq rows "
                      f"for s0, s1 defaulted to 0 (N-dependence unidentifiable)")
        else:
            s0 = min(r["tick_mean_s"] for r in ct_rows)
            s1 = 0.0
            method = (f"only one (N,M) combination for core_type={ct}; "
                      f"s0 = min observed tick_mean_s (upper bound, includes compute), s1=0")
        s0 = max(0.0, float(s0))
        s1 = float(s1)
        s_params[ct] = {"s0": s0, "s1": s1}
        notes.append(f"s[{ct}]: {method}")
    # default entries for core_types with no seq data at all (predict() may
    # still be asked about them) -- zero sync, documented as unmeasured.
    for ct in DEFAULT_CEILING:
        if ct not in s_params:
            s_params[ct] = {"s0": 0.0, "s1": 0.0}
            notes.append(f"s[{ct}]: no data -- defaulted to {{s0:0, s1:0}}")

    def sync_of(row):
        p = s_params.get(row["core_type"], {"s0": 0.0, "s1": 0.0})
        return p["s0"] + p["s1"] * row["N"]

    # --- 2. compute c(model, work) = a + b*work ---------------------------
    # compute_part = tick_mean_s - sync(N,core_type); c_sample = compute_part / M
    # (seq rows only -- no O_par confound). Then regress c_sample on work.
    c_params = {}
    models = sorted({r["model"] for r in seq_rows if r.get("model")})
    for model in models:
        m_rows = [r for r in seq_rows if r["model"] == model]
        samples = []
        for r in m_rows:
            compute_part = r["tick_mean_s"] - sync_of(r)
            compute_part = max(0.0, compute_part)  # guard against extrapolation noise
            c_sample = compute_part / r["M"]
            samples.append((r.get("work"), c_sample))
        works = sorted({w for w, _ in samples if w is not None})
        if len(works) >= 2:
            X = [[1.0, w if w is not None else 0.0] for w, _ in samples]
            y = [c for _, c in samples]
            a, b = _lstsq(X, y)
            method = f"lstsq c ~ 1 + work over {len(samples)} seq rows (work varies: {works})"
        else:
            a = float(np.mean([c for _, c in samples])) if samples else 0.0
            b = 0.0
            method = (f"work constant/absent ({works}) for model={model}; "
                      f"a = mean(c_sample) over {len(samples)} seq rows, b=0")
        c_params[model] = {"a": max(0.0, float(a)), "b": float(b)}
        notes.append(f"c[{model}]: {method}")

    def c_of(model, work):
        p = c_params.get(model)
        if p is None:
            return 0.0
        w = work if work is not None else 0.0
        return p["a"] + p["b"] * w

    # --- 3. O_par: tick_mean_s ~ ceil(M/W)*c + O_par + sync ----------------
    if par_rows:
        o_samples = []
        for r in par_rows:
            ceil_m_w = math.ceil(r["M"] / r["W"])
            predicted = ceil_m_w * c_of(r["model"], r.get("work")) + sync_of(r)
            o_samples.append(r["tick_mean_s"] - predicted)
        O_par = float(np.mean(o_samples))
        O_par = max(0.0, O_par)
        notes.append(f"O_par: mean residual (tick_mean_s - ceil(M/W)*c - sync) over {len(par_rows)} par rows")
    else:
        O_par = 0.0
        notes.append("O_par: no par rows -- defaulted to 0.0")

    # --- 4. rss_per_instance_mb[model]: peak_rss_mb ~ rss0 + rss1*M -------
    rss_params = {}
    all_models = sorted({r["model"] for r in rows if r.get("model")})
    for model in all_models:
        m_rows = [r for r in rows if r["model"] == model and r.get("peak_rss_mb") is not None]
        Ms = sorted({r["M"] for r in m_rows})
        if len(Ms) >= 2:
            X = [[1.0, r["M"]] for r in m_rows]
            y = [r["peak_rss_mb"] for r in m_rows]
            _rss0, rss1 = _lstsq(X, y)
            rss1 = max(0.0, float(rss1))
            method = f"regressed peak_rss_mb ~ 1 + M over {len(m_rows)} rows"
        elif m_rows:
            r0 = m_rows[0]
            rss1 = (r0["peak_rss_mb"] / r0["M"]) if r0["M"] else 0.0
            method = f"single M value ({Ms}) -- rss_per_instance = peak_rss_mb / M"
        else:
            rss1 = 0.0
            method = "no peak_rss_mb data -- defaulted to 0.0"
        rss_params[model] = float(rss1)
        notes.append(f"rss_per_instance_mb[{model}]: {method}")

    # --- 5. rtt_s: local-vs-distributed tick delta -------------------------
    dist_rows = [r for r in rows if r.get("placement") and r["placement"] != "local"]
    local_rows = [r for r in rows if r.get("placement") == "local"]
    if dist_rows and local_rows:
        key = lambda r: (r["F"], r["N"], r["M"], r["mode"], r["W"], r["core_type"], r["model"], r.get("work"))
        local_by_key = {}
        for r in local_rows:
            local_by_key.setdefault(key(r), []).append(r["tick_mean_s"])
        deltas = []
        for r in dist_rows:
            matches = local_by_key.get(key(r))
            if matches:
                local_tick = float(np.mean(matches))
                nm = r.get("n_machines") or 2
                remote_hops = max(1, nm - 1)
                deltas.append((r["tick_mean_s"] - local_tick) / remote_hops)
        if deltas:
            rtt_s = max(0.0, float(np.mean(deltas)))
            notes.append(f"rtt_s: mean (distributed_tick - matched_local_tick)/remote_hops over {len(deltas)} pairs")
        else:
            rtt_s = 0.0
            notes.append("rtt_s: distributed rows present but no matching local config found -- defaulted to 0.0")
    else:
        rtt_s = 0.0
        notes.append("rtt_s: local-only fit -- defaulted to 0.0")

    # --- 6. comms: per_edge_s[distance], in_per_link_s[distance], ---------
    #        per_byte_s (2026-07-28 wide-matrix revision -- CONTRACT.md
    #        "comms cost term"). Iteration 1 fit only a shared per-edge
    #        routing cost on scenario-wide totals (per_edge_s*n_edges).
    #        Iteration 2 replaced it with a purely per-federate shape
    #        (in/out_per_link_s*max_fed_in/out + fixed_per_tick_s) because a
    #        narrow 27-cell matrix with N pinned at 4 made n_edges and
    #        max_fed_in collinear -- it looked like edge PLACEMENT dominated.
    #        A wide matrix (N up to 32, M up to 64, edge counts 8..4096, 177
    #        runs) showed the per-federate-only shape scores R^2=0.22 there
    #        vs 0.97 for the totals shape: at N=16/M=4 cross_fed, Nto1 and
    #        all2all share max_fed_in=64 but differ 64 vs 1024 edges and
    #        +234us vs +5383us delta -- proof n_edges is NOT redundant with
    #        max_fed_in. Both mechanisms are real, so this shape keeps both:
    #        a dominant shared per-edge routing cost, plus a smaller
    #        per-federate inbound-polling cost. fixed_per_tick_s and
    #        out_per_link_s are dropped entirely -- the wide fit found the
    #        intercept statistically indistinguishable from zero once the
    #        edge term was present, and max_fed_out added nothing beyond
    #        max_fed_in.
    # Paired delta against CONTROL rows (exchange != "on") sharing every other
    # knob: compute+sync are already fitted above from Part-A data, so we do
    # NOT want comms to re-absorb them. delta = wired_tick - control_tick for
    # matched (F,N,M,mode,W,core_type,model,work,placement,n_ticks).
    DISTANCES = ("intra_fed", "cross_fed", "cross_machine")

    def _comms_key(r):
        return (r["F"], r["N"], r["M"], r["mode"], r["W"], r["core_type"],
                r["model"], r.get("work"), r.get("placement"), r["n_ticks"])

    control_rows = [r for r in rows if r.get("exchange") != "on"]
    wired_rows = [r for r in rows if r.get("exchange") == "on"]

    control_by_key = {}
    for r in control_rows:
        control_by_key.setdefault(_comms_key(r), []).append(r["tick_mean_s"])
    control_mean_by_key = {k: float(np.mean(v)) for k, v in control_by_key.items()}

    deltas = []
    design_rows = []
    n_skipped_no_control = 0
    for r in wired_rows:
        k = _comms_key(r)
        control_mean = control_mean_by_key.get(k)
        if control_mean is None:
            n_skipped_no_control += 1
            continue
        delta = r["tick_mean_s"] - control_mean
        distance = r.get("distance") or ""
        n_edges = r.get("n_edges") or 0
        max_fed_in = r.get("max_fed_in") or 0
        msg_width = r.get("msg_width") or 1
        freq = r.get("freq") or 1
        design_rows.append([
            n_edges if distance == "intra_fed" else 0.0,
            n_edges if distance == "cross_fed" else 0.0,
            n_edges if distance == "cross_machine" else 0.0,
            # in_per_link_s is POOLED across distance -- deliberately one column,
            # not one per distance. Physically, `distance` describes an edge's
            # ROUTING path (which broker chain carries it), whereas max_fed_in
            # prices the subscriber's own per-tick poll+deserialise loop, which is
            # local CPU work and cannot know where the value came from. Splitting it
            # per distance is also unidentifiable in practice: within a distance
            # stratum n_edges and max_fed_in correlate ~0.4-0.5 (most cells sit near
            # a fixed n_edges = k*max_fed_in ratio), and the per-distance split
            # produced a 9.5x-too-high intra_fed coefficient and a NEGATIVE
            # cross_fed one (clamped to 0) -- while pooling recovers 3.95e-6 /
            # 1.67e-6, matching the hand-computed values.
            max_fed_in,
            8.0 * msg_width * n_edges / freq,
        ])
        deltas.append(delta)

    comms_col_names = [
        "per_edge_s.intra_fed", "per_edge_s.cross_fed", "per_edge_s.cross_machine",
        "in_per_link_s",
        "per_byte_s",
    ]

    per_edge_s = {d: 0.0 for d in DISTANCES}
    in_per_link_s = 0.0
    per_byte_s = 0.0

    if n_skipped_no_control:
        notes.append(f"comms: skipped {n_skipped_no_control} wired row(s) -- "
                      f"no matching control row found for their (F,N,M,mode,W,"
                      f"core_type,model,work,placement,n_ticks) key")

    if not design_rows:
        notes.append("comms: no usable wired rows (with matched control) -- "
                      "all comms terms defaulted to 0.0")
    else:
        X_full = np.asarray(design_rows, dtype=float)
        y = np.asarray(deltas, dtype=float)

        # Weighted least squares, weight = 1 / max(|delta|, 20e-6): deltas in
        # the wide matrix span ~50us to ~22000us (nearly 3 orders of
        # magnitude), so plain OLS minimizes ABSOLUTE squared error and ends
        # up fitting only the largest cells while effectively ignoring
        # everything below ~1ms. Weighting by the inverse delta instead
        # minimizes RELATIVE error, so a district-scale scenario built from
        # thousands of small edges is represented as faithfully as the few
        # huge all2all cells at the top of the measured range -- this is the
        # difference between a model that works at district scale and one
        # that only works at the top of the range. The 20us floor guards
        # against divide-by-~0 for near-zero/noisy deltas.
        weights = 1.0 / np.maximum(np.abs(y), 20e-6)
        sqrt_w = np.sqrt(weights)
        X_weighted = X_full * sqrt_w[:, None]
        y_weighted = y * sqrt_w

        # Drop all-zero columns before solving -- an unidentifiable column
        # (e.g. no cross_machine rows in this csv) otherwise poisons lstsq.
        nonzero_mask = np.any(X_full != 0.0, axis=0)
        dropped = [comms_col_names[i] for i in range(len(comms_col_names)) if not nonzero_mask[i]]
        if dropped:
            notes.append(f"comms: columns {dropped} all-zero in usable data -- "
                          f"no data -- defaulted to 0.0")
        if nonzero_mask.any():
            X_reduced = X_weighted[:, nonzero_mask]
            coef_reduced = _lstsq(X_reduced, y_weighted)
            coef = np.zeros(len(comms_col_names))
            coef[nonzero_mask] = coef_reduced
        else:
            coef = np.zeros(len(comms_col_names))

        clamped = []
        for i, name in enumerate(comms_col_names):
            if coef[i] < 0.0:
                clamped.append(name)
                coef[i] = 0.0
        if clamped:
            notes.append(f"comms: clamped negative coefficient(s) {clamped} to 0.0 "
                          f"(delta likely buried in noise)")

        per_edge_s = {"intra_fed": float(coef[0]), "cross_fed": float(coef[1]),
                      "cross_machine": float(coef[2])}
        in_per_link_s = float(coef[3])
        per_byte_s = float(coef[4])

        pred = X_full @ coef
        r2 = _r_squared_weighted(y, pred, weights)
        rel_errs = np.abs(pred - y) / np.maximum(np.abs(y), 20e-6)
        median_rel_err = float(np.median(rel_errs))
        distances_with_data = sorted({r.get("distance") for r in wired_rows
                                       if r.get("distance") and
                                       _comms_key(r) in control_mean_by_key})
        notes.append(f"comms: weighted-least-squares fit (per_edge_s[distance]*n_edges "
                      f"+ in_per_link_s*max_fed_in [pooled across distance] + "
                      f"per_byte_s*8*msg_width*n_edges/freq), weight=1/max(|delta|,20us) "
                      f"to minimize relative error, used {len(design_rows)} wired "
                      f"row(s), distances with data = {distances_with_data}, "
                      f"weighted R^2 = {r2:.4f}, median relative error = "
                      f"{median_rel_err * 100:.2f}%")

    comms_params = {
        "per_edge_s": per_edge_s,
        "in_per_link_s": in_per_link_s,
        "per_byte_s": per_byte_s,
    }

    if notes_extra:
        notes.append(notes_extra)

    return {
        "c": c_params,
        "s": s_params,
        "O_par": float(O_par),
        "rss_per_instance_mb": rss_params,
        "rtt_s": float(rtt_s),
        "comms": comms_params,
        "notes": " | ".join(notes),
    }


def _empty_params(note):
    return {
        "c": {},
        "s": {ct: {"s0": 0.0, "s1": 0.0} for ct in DEFAULT_CEILING},
        "O_par": 0.0,
        "rss_per_instance_mb": {},
        "rtt_s": 0.0,
        "comms": {
            "per_edge_s": {"intra_fed": 0.0, "cross_fed": 0.0, "cross_machine": 0.0},
            "in_per_link_s": 0.0,
            "per_byte_s": 0.0,
        },
        "notes": note,
    }


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def predict(config, params, machines=None):
    """Apply the T_tick/T_sim framework to a single config.

    `config` keys: F, N, M, mode, W, core_type, model, work, placement,
    n_ticks, plus the optional Phase-D exchange knobs `exchange` ("none"|"on",
    default "none"), `distance` ("intra_fed"|"cross_fed"|"cross_machine"),
    `msg_width` (default 1), `freq` (default 1), `n_edges` (default 0),
    `max_fed_in` (default 0). `n_subs`/`max_fed_out` are still accepted on
    the config dict (harmless -- unused by predict()) for callers still
    carrying the older per-federate-only knob set. `machines` is currently
    unused for a single homogeneous config (kept as a parameter for
    interface symmetry with recommend(); a config with placement != "local"
    adds one rtt_s hop
    regardless of which machine).

    Returns dict with compute_s, sync_s, comms_s, T_tick_s, T_sim_s.
    `comms_s` = the existing distributed rtt_s hop PLUS the Phase-D
    data-exchange term (2026-07-28 wide-matrix revision -- CONTRACT.md "comms
    cost term"): per_edge_s[distance]*n_edges + in_per_link_s[distance]*
    max_fed_in + per_byte_s*8*msg_width*n_edges/freq. Both a shared per-edge
    routing cost and a per-federate inbound-polling cost are real (a wide
    N/M/edge-count matrix showed the edge-count term is NOT redundant with
    max_fed_in -- see `fit()` section 6 for the measurement), so both are
    applied. The `comms` params block is optional -- a pre-Phase-D params
    dict (no "comms" key) makes this term zero. A params dict still carrying
    an older shape (e.g. fixed_per_tick_s/out_per_link_s from the superseded
    per-federate-only iteration) also does not raise: the missing current
    keys default to 0.0/empty dict, so only its same-named `per_byte_s` --
    now applied against n_edges instead of max_fed_in -- carries over.
    """
    model = config["model"]
    work = config.get("work")
    M = config["M"]
    N = config["N"]
    mode = config["mode"]
    W = config.get("W")
    core_type = config["core_type"]
    placement = config.get("placement", "local")
    n_ticks = config.get("n_ticks", 1)

    c_entry = params.get("c", {}).get(model, {"a": 0.0, "b": 0.0})
    c = c_entry["a"] + c_entry["b"] * (work if work is not None else 0.0)

    if mode == "seq":
        compute = M * c
    elif mode == "par":
        if not W or W < 1:
            raise ValueError("mode=par requires W >= 1")
        O_par = params.get("O_par", 0.0)
        compute = math.ceil(M / W) * c + O_par
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    s_entry = params.get("s", {}).get(core_type, {"s0": 0.0, "s1": 0.0})
    sync = s_entry["s0"] + s_entry["s1"] * N

    rtt_s = params.get("rtt_s", 0.0)
    comms = rtt_s if placement and placement != "local" else 0.0

    # Phase-D data-exchange term (additive to the existing rtt_s hop above;
    # optional "comms" params block -- absent (pre-Phase-D params) -> zero).
    # Both mechanisms are real (see fit() section 6): a shared per-edge
    # routing cost (n_edges) and a per-federate inbound-polling cost
    # (max_fed_in) -- a wide N/M/edge-count matrix showed neither term alone
    # explains the data (n_edges is not redundant with max_fed_in).
    exchange = config.get("exchange", "none")
    if exchange == "on":
        distance = config.get("distance")
        n_edges = config.get("n_edges", 0) or 0
        max_fed_in = config.get("max_fed_in", 0) or 0
        msg_width = config.get("msg_width", 1) or 1
        freq = config.get("freq", 1) or 1
        comms_entry = params.get("comms", {})
        per_edge_s = comms_entry.get("per_edge_s", {}).get(distance, 0.0)
        # in_per_link_s is a scalar (pooled across distance). Older params files
        # stored it as a per-distance dict -- accept both so they still load.
        _in_raw = comms_entry.get("in_per_link_s", 0.0)
        in_per_link_s = _in_raw.get(distance, 0.0) if isinstance(_in_raw, dict) else _in_raw
        per_byte_s = comms_entry.get("per_byte_s", 0.0)
        comms += (per_edge_s * n_edges
                  + in_per_link_s * max_fed_in
                  + per_byte_s * 8 * msg_width * n_edges / freq)

    # T_tick = max over machines(compute+sync+comms); for a single homogeneous
    # config every federate/machine sees the same contribution, so the max is
    # just this one value.
    T_tick = compute + sync + comms
    T_sim = n_ticks * T_tick

    return {
        "compute_s": compute,
        "sync_s": sync,
        "comms_s": comms,
        "T_tick_s": T_tick,
        "T_sim_s": T_sim,
    }


# ---------------------------------------------------------------------------
# recommend()
# ---------------------------------------------------------------------------

def _ceiling_for(core_type, ceiling_zmq_ss):
    if core_type in ("zmq_ss", "tcp_ss"):
        return ceiling_zmq_ss
    return DEFAULT_CEILING.get(core_type, 100_000)


def recommend(scenario_spec, machines, params, ceiling_zmq_ss=33):
    """Grid/greedy search over decision vars (F, N, M, mode, W, core_type,
    n_machines_used) minimizing predicted T_sim, subject to:
      - N <= per-broker federate ceiling(core_type)
      - sum(RSS) <= RAM per machine (instances assumed evenly split: one
        federation per machine used, N federates * M instances resident)

    `scenario_spec`: {"total_instances": int, "model": str, "work": number|null,
                       "n_ticks": int, "tick_budget_s": number|null}
    `machines`: {"<alias>": {"cores": int, "ram_mb": number}, ...}

    This is a recommender, not an optimizer paper: the search is a bounded
    grid over two macro-strategies -- (a) everything on the single most
    powerful machine (placement=local, core_type=zmq/tcp, no ceiling/RTT
    pressure), and (b) sharding one federation per machine across the top
    `k` machines by core count (placement=distributed_nat, core_type=zmq_ss/
    tcp_ss, subject to the per-broker ceiling) -- for k = 1..len(machines).
    Multi-federation-per-machine and mixed direct/NAT placement are not
    searched; documented as a scoping simplification, not a contract gap
    (CONTRACT.md only requires "grid/greedy over sensible ranges").
    """
    total_instances = int(scenario_spec["total_instances"])
    model = scenario_spec["model"]
    work = scenario_spec.get("work")
    n_ticks = int(scenario_spec.get("n_ticks", 1))
    tick_budget_s = scenario_spec.get("tick_budget_s")

    rss_per_instance = params.get("rss_per_instance_mb", {}).get(model, 0.0)

    machine_list = sorted(
        ((alias, m) for alias, m in machines.items()),
        key=lambda kv: kv[1].get("cores", 0),
        reverse=True,
    )
    if not machine_list:
        raise ValueError("machines dict is empty")

    best = None
    all_feasible = []

    def consider(F, N, mode, W, core_type, placement, machines_used):
        nonlocal best
        if N < 1:
            return
        ceiling = _ceiling_for(core_type, ceiling_zmq_ss)
        if N > ceiling:
            return
        M = math.ceil(total_instances / (F * N))
        if M < 1:
            return
        if mode == "par" and (not W or W < 1 or W > M):
            return
        # RAM check: assume one federation's worth (N*M instances) resides on
        # each machine used (even split across `machines_used`).
        resident_instances = N * M
        rss_mb = resident_instances * rss_per_instance
        for alias, m in machines_used:
            ram_mb = m.get("ram_mb", float("inf"))
            if rss_mb > ram_mb:
                return  # infeasible on this machine

        config = {
            "F": F, "N": N, "M": M, "mode": mode, "W": W,
            "core_type": core_type, "model": model, "work": work,
            "placement": placement, "n_ticks": n_ticks,
        }
        pred = predict(config, params, machines)
        total_provided = F * N * M
        entry = {
            "config": config,
            "predicted": pred,
            "total_instances_provided": total_provided,
            "machines_used": [a for a, _ in machines_used],
            "rss_mb_per_machine": rss_mb,
        }
        all_feasible.append(entry)
        if best is None or pred["T_sim_s"] < best["predicted"]["T_sim_s"]:
            best = entry

    # macro-strategy (a): single most powerful machine, local, plain zmq/tcp
    top_alias, top_machine = machine_list[0]
    max_N_local = max(1, min(200, top_machine.get("cores", 32) * 4))
    for N in range(1, max_N_local + 1):
        for core_type in ("zmq", "tcp"):
            for mode in ("seq", "par"):
                if mode == "seq":
                    consider(1, N, "seq", None, core_type, "local", [(top_alias, top_machine)])
                else:
                    for W in sorted({1, 2, 4, min(top_machine.get("cores", 32), 32)}):
                        consider(1, N, "par", W, core_type, "local", [(top_alias, top_machine)])

    # macro-strategy (b): shard one federation per machine across top-k
    # machines (k = 2..len(machines)), distributed placement, *_ss core types
    # (forced by NAT per CONTRACT.md/plan Section 2).
    for k in range(2, len(machine_list) + 1):
        used = machine_list[:k]
        F = k
        max_N_dist = max(1, min(200, ceiling_zmq_ss * 2))
        for N in range(1, max_N_dist + 1):
            for core_type in ("zmq_ss", "tcp_ss"):
                for mode in ("seq", "par"):
                    per_machine_cores = min(m.get("cores", 32) for _, m in used)
                    if mode == "seq":
                        consider(F, N, "seq", None, core_type, "distributed_nat", used)
                    else:
                        for W in sorted({1, 2, 4, per_machine_cores}):
                            consider(F, N, "par", W, core_type, "distributed_nat", used)

    if best is None:
        raise RuntimeError("recommend(): no feasible config found under given constraints "
                            "(check machines RAM/cores vs. total_instances/rss_per_instance)")

    meets_budget = tick_budget_s is None or best["predicted"]["T_sim_s"] <= tick_budget_s
    best = dict(best)
    best["tick_budget_s"] = tick_budget_s
    best["meets_tick_budget"] = meets_budget
    best["n_candidates_evaluated"] = len(all_feasible)
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def _cmd_fit(args):
    params = fit(args.bench_csv)
    out_path = _write_json(params, args.out)
    print(f"wrote {out_path}")
    print(json.dumps(params, indent=2, sort_keys=True))


def _cmd_predict(args):
    with open(args.params) as f:
        params = json.load(f)
    machines = None
    if args.machines:
        with open(args.machines) as f:
            machines = json.load(f)
    config = {
        "F": args.F, "N": args.N, "M": args.M, "mode": args.mode, "W": args.W,
        "core_type": args.core_type, "model": args.model, "work": args.work,
        "placement": args.placement, "n_ticks": args.n_ticks,
        "exchange": args.exchange, "distance": args.distance, "fanout": args.fanout,
        "msg_width": args.msg_width, "freq": args.freq, "causality": args.causality,
        "n_edges": args.n_edges, "n_subs": args.n_subs,
        "max_fed_in": args.max_fed_in,
    }
    result = predict(config, params, machines)
    print(f"config: {config}")
    print(f"compute_s = {result['compute_s']:.6f}")
    print(f"sync_s    = {result['sync_s']:.6f}")
    print(f"comms_s   = {result['comms_s']:.6f}")
    print(f"T_tick_s  = {result['T_tick_s']:.6f}")
    print(f"T_sim_s   = {result['T_sim_s']:.6f}  (n_ticks={args.n_ticks})")


def _cmd_recommend(args):
    with open(args.scenario) as f:
        scenario_spec = json.load(f)
    with open(args.machines) as f:
        machines = json.load(f)
    with open(args.params) as f:
        params = json.load(f)
    result = recommend(scenario_spec, machines, params, ceiling_zmq_ss=args.ceiling_zmq_ss)
    print(f"evaluated {result['n_candidates_evaluated']} feasible candidates")
    print("recommended config:")
    print(json.dumps(result["config"], indent=2, sort_keys=True))
    print(f"machines used: {result['machines_used']}")
    print(f"total_instances_provided: {result['total_instances_provided']} "
          f"(requested: {scenario_spec['total_instances']})")
    print(f"predicted RSS per machine: {result['rss_mb_per_machine']:.1f} MB")
    p = result["predicted"]
    print(f"predicted compute_s={p['compute_s']:.6f} sync_s={p['sync_s']:.6f} "
          f"comms_s={p['comms_s']:.6f}")
    print(f"predicted T_tick_s={p['T_tick_s']:.6f} T_sim_s={p['T_sim_s']:.6f}")
    if result["tick_budget_s"] is not None:
        status = "MEETS" if result["meets_tick_budget"] else "EXCEEDS"
        print(f"tick_budget_s={result['tick_budget_s']} -> {status} budget")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="Fit cost-model params from a D2 bench CSV.")
    p_fit.add_argument("bench_csv")
    p_fit.add_argument("--out", default=str(DEFAULT_PARAMS_PATH))
    p_fit.set_defaults(func=_cmd_fit)

    p_pred = sub.add_parser("predict", help="Predict T_tick/T_sim for one config.")
    p_pred.add_argument("--params", required=True)
    p_pred.add_argument("--machines", default=None)
    p_pred.add_argument("--F", type=int, default=1)
    p_pred.add_argument("--N", type=int, default=1)
    p_pred.add_argument("--M", type=int, default=1)
    p_pred.add_argument("--mode", choices=["seq", "par"], default="seq")
    p_pred.add_argument("--W", type=int, default=None)
    p_pred.add_argument("--core-type", dest="core_type", default="zmq")
    p_pred.add_argument("--model", default="heavy_compute_dummy")
    p_pred.add_argument("--work", type=float, default=None)
    p_pred.add_argument("--placement", default="local")
    p_pred.add_argument("--n-ticks", dest="n_ticks", type=int, default=100)
    p_pred.add_argument("--exchange", choices=["none", "on"], default="none",
                         help="Phase-D data-exchange wiring: none (default, Part-A) | on")
    p_pred.add_argument("--distance", choices=["intra_fed", "cross_fed", "cross_machine"],
                         default=None, help="where subscribers live relative to publishers")
    p_pred.add_argument("--fanout", choices=["1to1", "1toN", "Nto1", "all2all"], default=None,
                         help="edge pattern between publisher/subscriber federates (informational)")
    p_pred.add_argument("--msg-width", dest="msg_width", type=int, default=1,
                         help="published payload vector length in doubles (default 1)")
    p_pred.add_argument("--freq", type=int, default=1,
                         help="publisher emits every freq-th tick (default 1)")
    p_pred.add_argument("--causality", choices=["same_step", "next_step"], default=None,
                         help="subscription causality (informational)")
    p_pred.add_argument("--n-edges", dest="n_edges", type=int, default=0,
                         help="total HELICS input->target links (default 0; used by "
                              "predict() for per_edge_s[distance]*n_edges and the "
                              "per_byte_s bytes-routed term)")
    p_pred.add_argument("--n-subs", dest="n_subs", type=int, default=0,
                         help="total HELICS input handles registered (default 0; "
                              "informational -- predict() regresses on n_edges/"
                              "max_fed_in, not this total)")
    p_pred.add_argument("--max-fed-in", dest="max_fed_in", type=int, default=0,
                         help="M x targets on the busiest subscriber federate (default 0; "
                              "used by predict() for in_per_link_s[distance]*max_fed_in)")
    p_pred.set_defaults(func=_cmd_predict)

    p_rec = sub.add_parser("recommend", help="Recommend a config for a target scenario + machine set.")
    p_rec.add_argument("--scenario", required=True, help="path to scenario-spec JSON")
    p_rec.add_argument("--machines", required=True, help="path to machines JSON")
    p_rec.add_argument("--params", required=True, help="path to fitted-params JSON")
    p_rec.add_argument("--ceiling-zmq-ss", dest="ceiling_zmq_ss", type=int, default=33,
                        help="per-broker federate ceiling for zmq_ss/tcp_ss (default 33, plan Section 2)")
    p_rec.set_defaults(func=_cmd_recommend)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
