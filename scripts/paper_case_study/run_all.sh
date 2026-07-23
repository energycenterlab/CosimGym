#!/usr/bin/env bash
# ============================================================================
# Paper case study — FULL experiment batch (S1 .. S6), one command.
#
#   bash scripts/paper_case_study/run_all.sh              # everything
#   FAST=1     bash .../run_all.sh    # skip the multi-hour RL sweep (S2/S3)
#   SKIP_S4C=1 bash .../run_all.sh    # skip distributed (needs SSH)
#   SEEDS="42 43 44" bash .../run_all.sh
#
# SAFETY MODEL (why this is safe to leave unattended):
#  * Every scenario runs SEQUENTIALLY, one process at a time. Concurrency causes
#    HELICS broker/port + Redis contention and produces spurious failures.
#  * Before EVERY run, `cg_between_runs` kills stray helics_broker /
#    federate_launcher processes, waits for the broker port range to free, and
#    fsyncs the previous run's results to disk.
#  * Docker services are verified healthy before the batch starts.
#  * Every run goes through run_profiled.py, which runs the scenario in its own
#    process group, hard-kills the group on timeout, and records wall-clock +
#    peak RSS into results/paper_case_study/exec_metrics.csv (instruction #9:
#    one comparative execution-metrics table for the paper).
#  * A failing scenario is recorded as FAIL and the batch CONTINUES, so one bad
#    stage never costs you the whole night.
#
# Prereqs: conda env cosim_gym; run from anywhere (script cd's to repo root).
# ============================================================================
set -uo pipefail          # NOT -e: a failed stage must not abort the batch
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export PYTHONPATH="scripts/paper_case_study:${PYTHONPATH:-}"
RUN="conda run -n cosim_gym python"
SEEDS="${SEEDS:-42 43 44}"
PCS="scripts/paper_case_study"
# shellcheck source=/dev/null
source "$PCS/batch_guard.sh"

banner() { echo; echo "============================================================"; echo "== $*"; echo "============================================================"; }

# profile_run <reps> <scenario...>   — guarded, sequential, metrics-recorded
profile_run() {
  local reps="$1"; shift
  for s in "$@"; do
    cg_between_runs
    echo "-- [$(date +%H:%M:%S)] $s (reps=$reps)"
    $RUN "$PCS/run_profiled.py" "$s" --reps "$reps" --timeout 86400
    cg_assert_results "$s"
  done
}

banner "PREFLIGHT"
cg_require_services
cg_kill_strays
$RUN -c "import helics,gymnasium,stable_baselines3,matplotlib,psutil,paho.mqtt.client,minio,fmpy" \
  && echo "   deps OK" || echo "   WARNING: a python dep is missing"
echo "   git commit: $(git rev-parse --short HEAD)  |  cores: $(nproc)"
# exec_metrics.csv is append-only; rotate it so the table matches THIS batch
# instead of accumulating duplicate rows across repeated batches.
mkdir -p results/paper_case_study
if [ -s results/paper_case_study/exec_metrics.csv ]; then
  mv results/paper_case_study/exec_metrics.csv \
     "results/paper_case_study/exec_metrics.prev-$(date +%Y%m%d_%H%M%S).csv"
  echo "   rotated previous exec_metrics.csv"
fi

# ---------------------------------------------------------------- S1 ---------
banner "S1 — PID baseline"
profile_run 1 cs_s1_baseline
cg_between_runs
$RUN "$PCS/fig_s1_traces.py"

# ------------------------------------------------------------- S2 + S3 -------
if [ "${FAST:-0}" = "1" ]; then
  echo; echo "== SKIPPING S2/S3 RL sweep (FAST=1) =="
else
  banner "S2 + S3 — RL sweep (seeds: $SEEDS) — THIS IS THE LONG ONE (hours)"
  $RUN "$PCS/make_seed_variants.py" --seeds $SEEDS
  VARIANTS=""
  for b in cs_s2_sac cs_s2_dqn cs_s2_reset_full cs_s2_reset_rolling cs_s2_reset_none cs_s3_fmu; do
    for s in $SEEDS; do
      [ -f "src/scenarios/${b}_s${s}.yaml" ] && VARIANTS="$VARIANTS ${b}_s${s}"
    done
  done
  echo "   variants:$VARIANTS"
  # shellcheck disable=SC2086
  profile_run 1 $VARIANTS
  cg_between_runs
  $RUN "$PCS/tab_s2_metrics.py"          --seeds $SEEDS
  $RUN "$PCS/fig_s2_learning_curves.py"  --seeds $SEEDS
  $RUN "$PCS/tab_s3_metrics.py"          --seeds $SEEDS
fi

# --------------------------------------------------------------- S4a --------
banner "S4a — vertical scaling (parallel vs sequential model execution)"
profile_run 3 cs_s4_vert_seq_N1  cs_s4_vert_par_N1 \
              cs_s4_vert_seq_N5  cs_s4_vert_par_N5 \
              cs_s4_vert_seq_N10 cs_s4_vert_par_N10 \
              cs_s4_vert_seq_N20 cs_s4_vert_par_N20 \
              cs_s4_vert_seq_N40 cs_s4_vert_par_N40
cg_between_runs
$RUN "$PCS/fig_s4_throughput.py"

# --------------------------------------------------------------- S4b --------
banner "S4b — multi-federation topology (hierarchy-broker evidence)"
profile_run 1 cs_s4_topo
D=$(ls -1dt logs/cs_s4_topo/*/ 2>/dev/null | head -1)
if [ -n "$D" ]; then
  grep -rhiE "sub_brokers|broker_address|hierarchy_broker=" "$D" 2>/dev/null \
    | grep -v None | sort -u > results/paper_case_study/s4b_hierarchy_broker_evidence.txt
  echo "   evidence -> results/paper_case_study/s4b_hierarchy_broker_evidence.txt"
fi

# --------------------------------------------------------------- S4c --------
if [ "${SKIP_S4C:-0}" = "1" ]; then
  echo; echo "== SKIPPING S4c (SKIP_S4C=1) =="
else
  banner "S4c — distributed across machines (SSH)"
  echo "   NOTE: ships as LOOPBACK (all hosts 127.0.0.1) = mechanism check, NOT speedup."
  echo "   For the real figure, edit deployment.machines.*.host + manager_address first."
  profile_run 3 cs_s4_dist_1m cs_s4_dist_2m cs_s4_dist_3m
  cg_between_runs
  if grep -q '127.0.0.1' src/scenarios/cs_s4_dist_1m.yaml; then
    $RUN "$PCS/fig_s4_machines.py" --loopback
  else
    $RUN "$PCS/fig_s4_machines.py"
  fi
fi

# ---------------------------------------------------------------- S5 --------
banner "S5 — digital-twin interface (external MQTT feeder)"
cg_between_runs
$RUN "$PCS/s5_external_feeder.py" --duration 400 --period 0.5 > /tmp/cg_s5_feeder.log 2>&1 &
FEEDER=$!
sleep 2
$RUN -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('cs_s5_dt')"
kill "$FEEDER" 2>/dev/null; wait "$FEEDER" 2>/dev/null
cg_assert_results cs_s5_dt
$RUN - <<'PY'
import json, glob, os
try:
    d = sorted(glob.glob('results/cs_s5_dt/*/'))[-1]
    ti = json.load(open(d+'federation_1/pid_federate_test_storage.json'))['inputs']['pid_federate.0']['T_indoor']
    real = [x for x in ti if x > -1e40]
    open('results/paper_case_study/s5_dt_acceptance_evidence.txt','w').write(
        f"S5 digital-twin acceptance ({os.path.basename(d.rstrip('/'))})\n"
        "building_federate = type:interface (no physics model); PID input T_indoor is fed\n"
        "externally over MQTT (cosim/cs_s5_dt/sensor/T_indoor -> building_federate.0/T_indoor).\n"
        f"ticks={len(ti)}  external range=[{min(real):.3f},{max(real):.3f}] degC\n"
        f"first external values: {[round(x,3) for x in real[:8]]}\n"
        "PASS: subscriber consumes externally fed values, not a physics model.\n")
    print(f"   S5 acceptance: external T_indoor range [{min(real):.2f},{max(real):.2f}] degC")
except Exception as e:
    print(f"   S5 acceptance check FAILED: {e}")
PY

# ---------------------------------------------------------------- S6 --------
banner "S6 — engineering-effort LOC audit"
$RUN "$PCS/s6_loc_audit.py"

# --------------------------------------------------------------- WRAP -------
banner "DONE"
cg_kill_strays
echo "Deliverables: results/paper_case_study/"
ls -1 results/paper_case_study/ 2>/dev/null | sed 's/^/   /'
echo
echo "Execution-metrics table (timings + peak RSS, all runs):"
column -s, -t results/paper_case_study/exec_metrics.csv 2>/dev/null || cat results/paper_case_study/exec_metrics.csv
echo
echo "Remaining HUMAN step: fig_s5_dashboard.png — run src/dashboard/run_dashboard.sh,"
echo "open the Live page during a cs_s5_dt run (with s5_external_feeder.py), screenshot it."
