#!/usr/bin/env bash
# FULL RL sweep for the paper case study (S2A + S2B + S3). LONG — hours.
# Run this yourself when ready; the fast tier (run_all.sh) is independent.
#
# Prereqs: docker services up, conda env cosim_gym, run from repo ROOT.
#   docker compose -f src/docker-compose.yaml up -d
#
# What it does:
#   1. generates per-seed scenario variants (seeds 42/43/44, unique checkpoints)
#   2. runs every variant sequentially (NEVER in parallel — concurrent runs cause
#      HELICS broker/port contention; that was observed and is why this is serial)
#   3. regenerates the S2/S3 tables and figures
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="scripts/paper_case_study:${PYTHONPATH:-}"
RUN="conda run -n cosim_gym python"
SEEDS="${SEEDS:-42 43 44}"

echo "== 1. generate seed variants (seeds: $SEEDS) =="
$RUN scripts/paper_case_study/make_seed_variants.py --seeds $SEEDS

echo "== 2. run the sweep (SEQUENTIAL, 1 rep each — RL cost is in the training) =="
BASES="cs_s2_sac cs_s2_dqn cs_s2_reset_full cs_s2_reset_rolling cs_s2_reset_none cs_s3_fmu"
VARIANTS=""
for b in $BASES; do
  for s in $SEEDS; do
    [ -f "src/scenarios/${b}_s${s}.yaml" ] && VARIANTS="$VARIANTS ${b}_s${s}"
  done
done
echo "   variants:$VARIANTS"
# run_profiled also records wall-clock + peak RSS into exec_metrics.csv
$RUN scripts/paper_case_study/run_profiled.py $VARIANTS --reps 1 --timeout 86400

echo "== 3. regenerate S2/S3 deliverables =="
$RUN scripts/paper_case_study/tab_s2_metrics.py --seeds $SEEDS
$RUN scripts/paper_case_study/fig_s2_learning_curves.py --seeds $SEEDS
$RUN scripts/paper_case_study/tab_s3_metrics.py --seeds $SEEDS

echo "DONE. See results/paper_case_study/ (tab_s2_metrics, fig_s2_learning_curves,"
echo "      tab_s2_sample_eff, tab_s3_metrics) and exec_metrics.csv."
