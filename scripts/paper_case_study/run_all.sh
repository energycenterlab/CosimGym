#!/usr/bin/env bash
# Reproduce all fast-tier paper case-study deliverables from a clean results/ dir.
# The long S2 RL sweep is a SEPARATE step (run_s2.sh) gated on author go-ahead.
#
# Prereqs: docker services up, conda env cosim_gym, run from repo ROOT.
#   docker compose -f src/docker-compose.yaml up -d
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH="scripts/paper_case_study:${PYTHONPATH:-}"
RUN="conda run -n cosim_gym python"
LAUNCH() { $RUN -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('$1')"; }

echo "== S1 baseline =="
LAUNCH cs_s1_baseline
$RUN scripts/paper_case_study/fig_s1_traces.py           # -> fig_s1_traces, tab_s1_metrics

echo "== S4a vertical scaling (10 scenarios x3 reps) =="
$RUN scripts/paper_case_study/run_profiled.py \
  cs_s4_vert_seq_N1 cs_s4_vert_par_N1 cs_s4_vert_seq_N5 cs_s4_vert_par_N5 \
  cs_s4_vert_seq_N10 cs_s4_vert_par_N10 cs_s4_vert_seq_N20 cs_s4_vert_par_N20 \
  cs_s4_vert_seq_N40 cs_s4_vert_par_N40 --reps 3 --timeout 900
$RUN scripts/paper_case_study/fig_s4_throughput.py       # -> fig_s4_throughput

echo "== S4b multi-federation (log evidence) =="
LAUNCH cs_s4_topo
D=$(ls -1dt logs/cs_s4_topo/*/ | head -1)
grep -rhiE "sub_brokers|broker_address|hierarchy_broker=" "$D" | grep -v None | sort -u \
  > results/paper_case_study/s4b_hierarchy_broker_evidence.txt

echo "== S5 digital-twin (feeder + acceptance) =="
$RUN scripts/paper_case_study/s5_external_feeder.py --duration 220 --period 0.5 &
FEEDER=$!
LAUNCH cs_s5_dt
kill $FEEDER 2>/dev/null || true
# (fig_s5_dashboard.png is a HUMAN screenshot of the live dashboard, see MANIFEST)

echo "== S6 LOC audit =="
$RUN scripts/paper_case_study/s6_loc_audit.py            # -> tab_s6_loc

echo "DONE. Deliverables under results/paper_case_study/"
