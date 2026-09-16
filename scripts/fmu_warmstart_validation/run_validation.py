"""
run_validation.py — does a rewound EnergyPlus FMU land where it should?

Written for `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md` §7. It
measures a rewind against the only reference that counts: the same building
stepped straight through without ever being restarted.

Two arms per case, driven by the *same* deterministic schedule so that any
difference is the restart and nothing else:

  truth    stepped straight through, never restarted — the reference
  replay   run to `rewind_from`, rewound to `target`, then stepped on

`replay` re-instantiates the slave at the beginning of its run period and
re-simulates every tick up to the target, feeding it the inputs it originally
saw, so it should come back *identical* to truth. Anything else is a bug.

This harness was first written to measure a second strategy, `runperiod_shift`,
which moved the RunPeriod's begin date instead of replaying to it. That strategy
was removed: the classic EnergyPlusToFMU wrapper ignores the begin date in the
IDF it re-preprocesses, so the "shifted" slave kept simulating January while the
model believed it had jumped forward. The harness stays because it is what
caught it — and what any future restart shortcut has to pass before it is
believed. Keep the truth arm.

Usage (from the project root, cosim_gym active — needs `energyplus` on PATH):

    python scripts/fmu_warmstart_validation/run_validation.py            # all cases
    python scripts/fmu_warmstart_validation/run_validation.py --quick    # winter, short

Writes results.json next to this file and prints a markdown table.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

RESOURCES = ROOT / 'src/models/model_catalog/physical_models/resources'
BUI0_FMU = RESOURCES / 'BUI0.fmu'      # replaced by --fmu

REAL_PERIOD = 600.0                      # BUI0: 6 timesteps/hour, fixed
TICKS_PER_DAY = int(86400 / REAL_PERIOD)
YEAR = 365 * 86400

IN_VARS = ['PeopleNumber', 'LightsWatt', 'EEquipWatt',
           'OthEquRadWatt', 'OthEquFCWatt', 'ZoneSetPoint']
OUT_VARS = ['TBuilding', 'HeatingLoadTarget']


# ---------------------------------------------------------------------------
# The schedule the building is driven with — a function of the model's own tick,
# so both arms see identical inputs at identical simulated moments.
# ---------------------------------------------------------------------------

def schedule(tick: int) -> dict:
    """Occupied 08:00-18:00 on weekdays, mirroring bui0_input_feeder."""
    tick_of_day = (tick - 1) % TICKS_PER_DAY
    day_index = (tick - 1) // TICKS_PER_DAY
    hour = tick_of_day * REAL_PERIOD / 3600.0
    weekend = (day_index % 7) in (5, 6)          # run period starts on a Monday
    occupied = (not weekend) and 8.0 <= hour < 18.0
    return {
        'PeopleNumber':  4.0 if occupied else 0.0,
        'LightsWatt':    400.0 if occupied else 40.0,
        'EEquipWatt':    500.0 if occupied else 50.0,
        'OthEquRadWatt': 100.0 if occupied else 0.0,
        'OthEquFCWatt':  100.0 if occupied else 0.0,
        'ZoneSetPoint':  21.0 if occupied else 18.0,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def make_model(record_history=False, name='bui0.0'):
    from models.base_FMU_model import BaseFMUModel
    from models.model_catalog.ModelCatalog import ModelMetadata, ParameterSpec, ParameterType

    def spec(n):
        return ParameterSpec(name=n, type=ParameterType('float'), default_value=0.0)

    metadata = ModelMetadata(
        name='bui0_building_fmu',
        class_name='BaseFMUModel',
        module_path='models.base_FMU_model',
        version='1.0.0',
        description='BUI0',
        inputs={n: spec(n) for n in IN_VARS},
        outputs={n: spec(n) for n in OUT_VARS},
        max_sim_time=YEAR,
        sim_start_date='01-01',
        user_defined={
            'fmu_source': {'type': 'local', 'path': str(BUI0_FMU)},   # noqa: B008
            'fmu_reset': {'replay_inputs': 'history'},
        },
    )
    config = SimpleNamespace(
        user_defined={},
        inputs=IN_VARS,
        outputs=OUT_VARS,
        parameters={},
        init_state={n: 0.0 for n in IN_VARS + OUT_VARS} | {'ZoneSetPoint': 20.0},
        start_time='2024-01-01T00:00:00',
        end_time='2024-12-31T00:00:00',
        time_stop=None,
        real_period=REAL_PERIOD,
        # 'rolling' is what turns input recording on; the spin-up replays from it.
        reset_mode='rolling' if record_history else None,
        rolling_window=None,
        episode_length=None,
        reset_period=None,          # unbounded history: this harness rewinds by hand
        n_episodes=None,
    )
    return BaseFMUModel(name, metadata, config, logging.getLogger('validation'))


def run_straight(end_tick, from_tick=1):
    """Step a fresh slave from *from_tick* to *end_tick* with no restart."""
    m = make_model()
    try:
        return step_range(m, from_tick, end_tick)
    finally:
        m.finalize()


def run_with_rewind(rewind_from, target, end_tick):
    """Step to *rewind_from*, rewind to *target*, then step on to *end_tick*.

    The replay is instrumented: a spin-up that quietly fell back to held-constant
    inputs would otherwise look exactly like a spin-up that did not help.
    """
    from models.base_FMU_model import BaseFMUModel

    m = make_model(record_history=True)
    trace = {'replay_ticks': 0, 'history_used': None, 'first_replayed_tick': None}
    original_advance = BaseFMUModel._advance

    def traced_advance(self, n_steps):
        trace['replay_ticks'] = n_steps
        trace['history_used'] = len(self._input_history) >= n_steps
        return original_advance(self, n_steps)

    BaseFMUModel._advance = traced_advance
    try:
        step_range(m, 1, rewind_from)
        m.reposition(target, at_ts=rewind_from + 1, reason='validation rewind')
        # After the rewind federation tick `rewind_from + 1` counts as `target`.
        offset = (rewind_from + 1) - target
        series = step_range(m, target, end_tick, ts_offset=offset)
        return series, trace
    finally:
        BaseFMUModel._advance = original_advance
        m.finalize()


def step_range(model, first_tick, last_tick, ts_offset=0):
    """Drive the model over [first_tick, last_tick]; return {tick: outputs}."""
    out = {}
    for tick in range(first_tick, last_tick + 1):
        model._step(tick + ts_offset, schedule(tick))
        out[tick] = {k: model.state.outputs[k] for k in OUT_VARS}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compare(truth, arm, first_tick, last_tick):
    ticks = [t for t in range(first_tick, last_tick + 1) if t in truth and t in arm]
    dT = [arm[t]['TBuilding'] - truth[t]['TBuilding'] for t in ticks]

    def rmse(values):
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else float('nan')

    def after(days):
        cut = first_tick + days * TICKS_PER_DAY
        window = [d for t, d in zip(ticks, dT) if t >= cut]
        return rmse(window), (max((abs(d) for d in window), default=float('nan')))

    # First tick from which the error stays below 0.2 K for the rest of the run.
    settled = None
    for i in range(len(ticks)):
        if all(abs(d) < 0.2 for d in dT[i:]):
            settled = (ticks[i] - first_tick) * REAL_PERIOD / 3600.0
            break

    def heating_kwh(series):
        return sum(max(0.0, series[t]['HeatingLoadTarget']) for t in ticks) * REAL_PERIOD / 3.6e6

    truth_kwh, arm_kwh = heating_kwh(truth), heating_kwh(arm)
    rmse_1d, max_1d = after(1)
    return {
        'ticks_compared': len(ticks),
        'rmse_K': rmse(dT),
        'max_abs_K': max((abs(d) for d in dT), default=float('nan')),
        'rmse_after_1d_K': rmse_1d,
        'max_abs_after_1d_K': max_1d,
        'settle_hours_below_0.2K': settled,
        'heating_kwh_truth': truth_kwh,
        'heating_kwh_arm': arm_kwh,
        'heating_err_pct': (arm_kwh - truth_kwh) / truth_kwh * 100.0 if truth_kwh else float('nan'),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def sample_series(truth, arm, first_tick, days, every=6):
    """Hourly TBuilding from both arms over the first *days* days after the rewind."""
    last = first_tick + days * TICKS_PER_DAY - 1
    return [
        {'tick': t,
         'hours_after_rewind': (t - first_tick) * REAL_PERIOD / 3600.0,
         'truth_T': round(truth[t]['TBuilding'], 3),
         'arm_T': round(arm[t]['TBuilding'], 3),
         'truth_Q': round(truth[t]['HeatingLoadTarget'], 1),
         'arm_Q': round(arm[t]['HeatingLoadTarget'], 1)}
        for t in range(first_tick, min(last, max(truth)) + 1, every)
        if t in truth and t in arm
    ]


def run_case(name, target_day, rewind_from_day, compare_days, dump_series=0):
    """One season. Ticks are 1-based; day d starts at tick (d-1)*TICKS_PER_DAY + 1."""
    target = (target_day - 1) * TICKS_PER_DAY + 1
    rewind_from = (rewind_from_day - 1) * TICKS_PER_DAY
    end_tick = target + compare_days * TICKS_PER_DAY - 1

    print(f"\n### case {name}: rewind from day {rewind_from_day} back to day {target_day}, "
          f"compare {compare_days} days", flush=True)

    print("  truth …", flush=True)
    truth = run_straight(end_tick)

    results = {}
    print("  replay …", flush=True)
    arm, trace = run_with_rewind(rewind_from, target, end_tick)
    print(f"    replayed {trace['replay_ticks']} ticks, "
          f"recorded inputs used: {trace['history_used']}", flush=True)
    results['replay'] = compare(truth, arm, target, end_tick) | trace
    if dump_series:
        results['replay']['series'] = sample_series(truth, arm, target, dump_series)

    return {
        'target_day': target_day,
        'rewind_from_day': rewind_from_day,
        'compare_days': compare_days,
        'arms': results,
    }


def markdown(cases):
    lines = ["| case | arm | replayed | inputs | RMSE ΔT | max ΔT | heating err |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for case_name, case in cases.items():
        for arm_name, m in case['arms'].items():
            lines.append(
                f"| {case_name} | {arm_name} | "
                f"{m.get('replay_ticks', '—')} ticks | "
                f"{'recorded' if m.get('history_used') else 'held'} | "
                f"{m['rmse_K']:.4f} K | {m['max_abs_K']:.4f} K | "
                f"{m['heating_err_pct']:+.4f} % |")
    return "\n".join(lines)


def main():
    global BUI0_FMU

    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='winter only, 10 compared days, spin-ups 0 and 1')
    ap.add_argument('--fmu', type=Path, default=BUI0_FMU,
                    help='FMU to validate (default: the repo BUI0.fmu). Used to compare '
                         'against a variant, e.g. one with daily shading updates.')
    ap.add_argument('--out', type=Path, default=None,
                    help='where to write the results JSON (default: results.json here)')
    ap.add_argument('--cases', default='winter,summer,long_rewind',
                    help='comma-separated subset of winter,summer,long_rewind')
    ap.add_argument('--dump-series', type=int, default=0, metavar='DAYS',
                    help='also store hourly TBuilding for the first DAYS days after each rewind')
    args = ap.parse_args()
    BUI0_FMU = args.fmu.resolve()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('validation').setLevel(logging.ERROR)

    if not BUI0_FMU.exists():
        sys.exit(f"FMU not found at {BUI0_FMU}")
    print(f"FMU under test: {BUI0_FMU}")

    wanted = [c.strip() for c in args.cases.split(',') if c.strip()]

    # target_day, rewind_from_day, compared days, spin-ups
    #   winter      deep heating season - the set-point anchors the zone
    #   summer      free floating, nothing anchors it, no heating to mask an error
    #   long_rewind does the distance rewound matter, or only the spin-up?
    catalogue = {
        'winter':      dict(target_day=31,  rewind_from_day=40,  compare_days=10 if args.quick else 30),
        'summer':      dict(target_day=196, rewind_from_day=205, compare_days=30),
        'long_rewind': dict(target_day=31,  rewind_from_day=150, compare_days=15),
    }
    if args.quick:
        wanted = ['winter']

    cases = {}
    for name in wanted:
        if name not in catalogue:
            sys.exit(f"unknown case '{name}'; choose from {', '.join(catalogue)}")
        cases[name] = run_case(name, dump_series=args.dump_series, **catalogue[name])

    out = args.out or Path(__file__).with_name('results.json')
    out.write_text(json.dumps(cases, indent=2))
    table = markdown(cases)
    print("\n" + table)
    print(f"\nwritten: {out}")


if __name__ == '__main__':
    main()
