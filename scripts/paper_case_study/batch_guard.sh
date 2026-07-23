#!/usr/bin/env bash
# Batch-run safety helpers, sourced by run_all.sh / run_s2.sh.
#
# WHY: back-to-back scenario runs interfere unless the previous run is fully torn
# down. Observed failure modes in this project:
#   - leftover `helics_broker` / federate processes holding TCP ports
#     -> "Unable to bind zmq pull socket" / "address already in use"
#   - a run started before Docker services (redis/mosquitto/minio) were healthy
#   - concurrent runs contending on Redis + broker ports (never run in parallel)
# These helpers make each run start from a known-clean state.

CG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- kill any stray simulation processes left by a previous run ---------------
# Deliberately NARROW: only helics brokers and this repo's federate launchers.
# Never touches Redis/Mosquitto/MinIO containers or unrelated user processes.
cg_kill_strays() {
  local n=0
  for pat in "helics_broker" "federate_launcher.py"; do
    while read -r pid; do
      [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null && n=$((n+1))
    done < <(pgrep -f "$pat" 2>/dev/null)
  done
  if [ "$n" -gt 0 ]; then
    sleep 3
    for pat in "helics_broker" "federate_launcher.py"; do
      while read -r pid; do
        [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null
      done < <(pgrep -f "$pat" 2>/dev/null)
    done
    echo "   [guard] terminated $n stray sim process(es)"
  fi
  return 0
}

# --- wait until no SIMULATION-owned listener remains --------------------------
# Only counts sockets held by helics_broker / federate_launcher. Checking the raw
# 20000-29999 range is wrong: unrelated long-lived services legitimately listen
# there, which would make this warn forever and stall every run.
cg_wait_ports() {
  local tries=${1:-20}
  for _ in $(seq "$tries"); do
    if ! ss -ltnp 2>/dev/null | grep -qE 'helics_broker|federate_launcher'; then
      return 0
    fi
    sleep 1
  done
  echo "   [guard] WARNING: a simulation socket is still bound after ${tries}s; continuing"
  return 0
}

# --- verify docker services are up and healthy -------------------------------
cg_require_services() {
  local compose="$CG_ROOT/src/docker-compose.yaml"
  if ! docker compose -f "$compose" ps --format '{{.Service}} {{.Status}}' 2>/dev/null | grep -q "redis.*Up"; then
    echo "   [guard] Redis not up — starting docker services..."
    docker compose -f "$compose" up -d >/dev/null 2>&1 || true
  fi
  for _ in $(seq 30); do
    if docker compose -f "$compose" ps --format '{{.Service}} {{.Status}}' 2>/dev/null \
        | grep -q "redis.*healthy"; then
      return 0
    fi
    sleep 2
  done
  echo "   [guard] WARNING: redis did not report healthy; continuing anyway"
  return 0
}

# --- full inter-run barrier ---------------------------------------------------
# Call BEFORE every scenario run. Idempotent and never fails the batch.
cg_between_runs() {
  cg_kill_strays
  cg_wait_ports 20
  sync                     # flush result/log writes of the previous run to disk
  sleep 2
}

# --- confirm a run actually produced results ---------------------------------
# usage: cg_assert_results <scenario_name>   (skip for sink:none scenarios)
cg_assert_results() {
  local name="$1"
  if compgen -G "$CG_ROOT/results/$name/*" > /dev/null; then
    echo "   [guard] results present for $name"
    return 0
  fi
  echo "   [guard] WARNING: no results dir for $name (expected for sink:none scenarios)"
  return 0
}
