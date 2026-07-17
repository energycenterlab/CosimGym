# HANDOFF: Distributed SSH Federate Spawning — Scaling Fix

Branch: `distributed-ssh-spawning-plan`. **NEVER commit — ask user first.**

**This supersedes the old T1–T7 handoff below the `---`.** That work (base distributed
spawning: `deployment:` YAML, `RemoteExecutor`, verified via `distributed_demo.yaml`) is
DONE and merged into this branch's history. This session is a follow-on: distributed runs
worked for small federate counts but broke at scale. That's what's being fixed now.

## Goal

Make distributed runs (federates spread across manager + remote machines via SSH) scale
to hundreds of federates per machine, not just a handful. Also fix three related
operational bugs the user hit while stress-testing: Ctrl+C not killing everything, and
result folders piling up empty on failed runs.

## Current Progress (IN PROGRESS — one test run currently hanging, not yet root-caused)

### Root cause found and fixed: ssh session/connection limits, not a HELICS limit

`benchmark_scale_distributed.yaml` (201 federates, 36 remote per machine on 2 remote
machines) failed with `✗ Federate failed with code 255`, logs empty. 255 = ssh's own
error code. Reproduced standalone (no CosimGym) against `eclab-cloud1`
(130.192.238.9): 36 concurrent `-tt` ssh sessions over one ControlMaster socket → 7/36
fail with `mux_client_request_session: session request failed: Session open refused by
peer` then `kex_exchange_identification: read: Connection reset by peer`. Cause:
`RemoteExecutor.spawn()` (old code) opened **one ssh session per federate**. sshd's
`MaxSessions` (default 10, unset on both remotes) refuses sessions past 10 on one
connection; ssh then silently opens a **new TCP connection** per refusal, and sshd's
`MaxStartups` (default `10:30:100`) **probabilistically drops** those past 10 concurrent
unauthenticated connections — hence the nondeterministic failure count run to run.

**I had earlier (wrongly) attributed this to a "zmq_ss federate ceiling" (~33 federates)
and told the user to avoid scaling past it. That diagnosis is RETRACTED.** Proof: ran
201 federates fully local (same `zmq_ss` core, `benchmark_scale_local.yaml`) → 0
failures, 15s total. The ceiling was ssh sessions per machine, not HELICS/zmq_ss at all.

### Fix implemented: one ssh session per machine, not per federate

- **New `src/core/remote_spawner.py`**: runs ON the remote machine. Reads its machine's
  federate list from Redis (key `cosim:spawn:<sim_id>:<alias>`, NOT passed as argv — so
  the ssh command line is fixed-size regardless of federate count), spawns each as a
  local child (not `setsid`'d — stays in this process's group so a SIGHUP reaches all of
  them), supervises them, tears all down together (SIGTERM to all, then one bounded
  wait, then SIGKILL stragglers) on first failure or on receiving
  SIGTERM/SIGINT/SIGHUP itself.
- **`RemoteExecutor.spawn()` → `RemoteExecutor.spawn_many(spawn_key, redis_url,
  remote_log_file)`** (`src/core/remote_executor.py`): opens ONE `-tt` ssh session
  running `remote_spawner.py`. Old `spawn()` deleted, not deprecated — nothing else
  called it.
- **`ScenarioManager` (`src/core/ScenarioManager.py`)**:
  - `_create_remote_federate` now only QUEUES a federate spec into
    `self._pending_remote_federates[host_alias]` — does not spawn. (A machine's
    federates can span multiple federations, so nothing can spawn until every
    federation's setup loop has run.)
  - New `_spawn_remote_batches()`: called once after ALL federations are set up
    (`_setup_classic_scenario`, right after the federation loop). For each machine
    alias with pending federates: publishes the list to Redis, calls
    `executor.spawn_many(...)`. The returned `Popen` (the ssh child) gets two synthetic
    attrs, `_cosim_weight` (= federate count on that machine) and `_cosim_label`
    (e.g. `"Remote machine 'machine_a' (36 federates)"`), so downstream counting/logging
    treats one ssh child as N federates instead of 1.
  - `_collect_completed` (used by `_monitor_processes`) reads `_cosim_weight`/
    `_cosim_label` (defaulting to 1/`label` for local federates, which have neither
    attr) so `federates_started`/`_completed`/`_failed` counters stay in units of
    federates, and the log line names the batch, not an anonymous "Federate".
  - `_spawn_keys` list + cleanup in `_emergency_cleanup`: deletes the per-machine Redis
    spawn-list keys on teardown (mirrors the existing `self.redis_key` deletion).

Both this design's safety properties were verified BEFORE writing code (not assumed):
- one ssh session CAN spawn 100+ remote children — tested standalone, confirmed.
- killing the local ssh client (SIGHUP via `-tt` pty) DOES kill all of that machine's
  children — tested standalone, confirmed (0 leftover after kill).
- ONE failing child DOES abort that machine's batch fast (~3s), not after the slowest
  child's timeout — tested standalone, confirmed (`wait -n` loop pattern).

### Fix implemented: double-Ctrl+C orphan bug

`_signal_handler` (`ScenarioManager.py`) used to call `_emergency_cleanup()` then
`exit(0)` with no re-entrancy guard. A second SIGINT while cleanup was still running
(killing federates one by one) re-entered the handler, `_emergency_cleanup` early-
returned on its `_cleanup_done` flag, and `exit(0)`'s `SystemExit` unwound through the
**first** cleanup's kill loop, abandoning everything not yet reached. Measured before
the fix: 83–124 orphaned local federates out of 201 on a double Ctrl+C ~0.3–1s apart;
0 orphans on a single, patient Ctrl+C. **Fix**: `_signal_handler` now does
`signal.signal(SIGINT, SIG_IGN)` and same for `SIGTERM` as its first action, before
anything else — further interrupts during this process's remaining lifetime (which is
just cleanup) are ignored outright, so there is no re-entrant window.

### Fix implemented: empty result-folder pileup

`_setup_results_folder()` creates `results/<scenario>/<sim_id>/` and writes
`metadata.json` during SETUP, before any federate runs. Every failed run (very common
while stress-testing sshd's limits) left one of these behind containing only that stub
— 19 accumulated during this debugging session. **Fix**: new
`_prune_empty_results_folder()`, called from `_emergency_cleanup` (all exit paths). If
the run's results dir contains nothing but `metadata.json` anywhere under it, delete
the whole dir. Any real result file present anywhere → no-op. Never raises.

### Tests

`tests/test_remote_executor.py`'s `TestSpawnUsesPtyAndDetachedGroup` rewritten for
`spawn_many` (asserts remote command runs `remote_spawner.py` with the Redis spawn key,
and explicitly asserts `federate_launcher.py` does NOT appear in the ssh argv — proving
the command line no longer grows with federate count).

`pytest tests/test_remote_executor.py tests/test_scenario_manager_remote.py
tests/test_deployment_config.py -q` → **38 passed, 1 skipped** (the 1 skip is the
pre-existing gated live-loopback test, `COSIMGYM_TEST_SSH_LOCALHOST=1` not set).

Full suite NOT yet re-run after these changes — do that before considering this task
done (`pytest tests/ -q`).

### Live 201-federate test: PROGRESS BUT NOT YET GREEN

Ran `benchmark_scale_distributed` (201 federates: ~129 local on manager, 36 each on
`eclab-cloud1`/`eclab-cloud5`) end to end:

- **The ssh-255 failure is GONE.** 0 occurrences of `failed with code` in this run's
  log — the batching fix works as far as spawn reliability goes.
- **But the run then hung** and was still running at the 500s timeout I set, so I sent
  it SIGTERM from outside. Cleanup (both new fixes) DID work correctly on that SIGTERM:
  clean shutdown logged, 0 local orphans, 0 orphans on either remote afterward.
- **Not yet root-caused why it hung.** What I'd found right before being interrupted:
  - Manager's own `federation_manager.log` for that run was completely empty — odd,
    worth checking log level / whether the manager log handler is even attached at that
    point, or whether output is buffered and lost on SIGTERM.
  - Exactly one stdio log had content:
    `federates/federate_weather_federate.stdio.log` contained only:
    `[console] [warning] weather_federate (...)[t=24]:: disconnect Timer expired forcing disconnect`
    — this is a LOCAL federate (weather_federate isn't `host:`-tagged in this scenario,
    confirm by grepping the YAML), and "Timer expired forcing disconnect" is a HELICS
    message meaning that federate's *own* disconnect timer fired because it never
    finished — i.e. it was one of the ones still stuck when I killed the run, not
    necessarily the federate that caused the stall.
  - Broker log showed normal startup: broker up, listening, `--federates=201` from
    `federation_1.broker`, and at least one federate (`root`) connected successfully
    (log cuts off there in what I'd read so far).
  - I had NOT yet checked: whether all 201 federates actually connected to the broker
    (partial join = classic HELICS hang, federation waits forever for the missing
    ones), whether `remote_spawner.py` itself started correctly on both remotes and
    actually launched 36 children each (its own stdout goes to
    `federates/_spawner_<alias>.log` on the remote, collected back by rsync — I hadn't
    checked whether that file even made it back, since the run was killed rather than
    finishing cleanly, and `_collect_remote_results` may not have had a chance to run
    on a SIGTERM-based abort — worth checking whether cleanup collects remote logs on
    the interrupt path or only on the success path).

## What Worked

- **Test infrastructure hypotheses standalone, outside CosimGym, before touching code.**
  Every claim in the "Fix implemented" sections above (sshd limits, batching viability,
  SIGHUP propagation, fast-fail propagation) was proven with bare `ssh`/`bash` loops
  against the real remote machines FIRST. This is what caught that my first-instinct fix
  ("just throttle spawns" or "raise MaxSessions on the remotes") would have been wrong —
  measured that ssh clients cost ~41MB RSS each and live for the whole run, so 1000
  target federates = O(federates) manager processes forever, not just a startup hiccup.
  User explicitly asked to think about scaling before implementing; this line of
  investigation is why the recommendation changed from "throttle" to "batch per
  machine."
- Reading `remote_executor.py` and `ScenarioManager.py`'s existing `_create_local_federate`/
  `_create_remote_federate`/`_monitor_processes`/`_collect_completed` fully before writing
  new code — the batching design (queue-then-flush, `_cosim_weight`/`_cosim_label` on the
  Popen handle) was shaped to slot into the existing counter/monitor code with minimal
  disruption rather than a parallel bookkeeping path.
- Redis as the spawn-list channel: remotes already reach the manager's Redis (verified
  reachable on both `eclab-cloud1`/`eclab-cloud5` before deciding on this), and every
  federate already fetches its config from Redis, so this reuses an existing channel
  instead of inventing a new one.

## What Didn't Work / Retracted

- **"zmq_ss has a ~33-federate ceiling" — WRONG, retracted this session.** Was actually
  the ssh session limit. Don't reintroduce core-type warnings or federate-count caps
  based on that old (bad) data.
- Raising `MaxSessions`/`MaxStartups` on the remote sshd — considered, rejected. Needs
  sudo on shared machines the user doesn't own outright, and doesn't fix the underlying
  O(federates) manager-side ssh-client cost; just moves the wall further out.
- Two of my early process-count measurements during earlier Ctrl+C debugging (see old
  section below) were wrong before being corrected — worth remembering the counting
  pitfalls if writing more diagnostic scripts: `pgrep -c -f federate_launcher.py`
  matches the ssh CLIENT process too (its argv contains the remote command string), and
  `pgrep -c ... || echo 0` double-counts on a match. The correct pattern used throughout
  this session:
  `ps -eo args= | grep -F federate_launcher.py | grep -vE "^(ssh|bash|sh) " | grep -cE "^(/[^ ]*)?python[0-9.]* "`.

## Next Steps

1. **Root-cause the 201-federate hang** (the actual current blocker). Suggested angle:
   rerun with a SHORTER timeout and, before killing it, check
   `ss -tn state established '( dport = :23404 or sport = :23404 )'` on the manager to
   count actual broker connections against the expected 201, and check both remotes for
   `ps -eo args= | grep remote_spawner.py` / count of federate children each spawned —
   this tells you whether it's a partial-join hang (some federates never got spawned or
   never reached the broker) vs. something else (e.g. sync/offset stall once all 201
   *did* join). Also check why `federation_manager.log` was empty for that run — may be
   a log level or buffering issue unrelated to the hang itself but worth fixing
   regardless.
2. Once the hang is understood and fixed (or confirmed to be a pre-existing scenario
   config issue unrelated to this session's spawn-batching change — test against the
   OLD per-federate-spawn code on a small scenario to isolate whether the hang is new),
   re-run the 201-federate distributed scenario to a clean completion and confirm result
   files land (not just metadata.json).
3. Run the full test suite: `pytest tests/ -q` (only the 3 targeted files were run this
   session).
4. `graphify update .` (per CLAUDE.md rule — not yet run this session for the new
   `remote_spawner.py` file or the `ScenarioManager.py`/`remote_executor.py` edits).
5. Re-verify the ORIGINAL small-scale distributed demo still passes after the spawn
   dispatch change (`python src/verify_distributed_demo.py` — see old section below) —
   the queue-then-batch refactor touches the same code path that demo exercises.
6. Ask user before committing. Do not commit mid-debug.

---

# ORIGINAL HANDOFF (T1–T7, base distributed-spawning feature — DONE, retained for reference)

## Goal

Let `ScenarioManager` spawn federates on remote machines over SSH, declared in scenario
YAML (`deployment:` block + per-federate `host:` key). Brokers stay local. No `host:` =
byte-identical local behavior. Full spec + rationale: `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`
(read this first, it is the source of truth — decisions there are user-approved, do not
relitigate). Alternatives considered: `distributed_ssh_spawning_alternatives.md` same folder.

## Current Progress

**ALL TASKS DONE — T1–T7 complete, live localhost-as-remote E2E PASSED.**

**T6 live E2E result (2026-07-15):** ran `python src/verify_distributed_demo.py` on this machine
with passwordless ssh to 127.0.0.1 + the cosim docker stack up. `distributed_demo` (pv_federate
spawned remotely over ssh) completed: 1 broker, 5 federates. **2304/2304 recorded timeseries
records matched the all-local twin `pv_batt_test_base` within 1e-9.** Remote federate results were
rsynced back into the manager's local `results/` tree (T5 collection verified). Mid-simulation
SIGINT (remote federate confirmed alive first, then interrupted) → remote federate GONE, zero
orphans local & remote (T5 `_cleanup_remote_execution` pkill + master close verified).

**Two bugs found during E2E and fixed:**
1. `remote_executor.py` `_master_cmd()`: used `ssh ... -nNf` (background-fork master). With
   `subprocess.run(capture_output=True)` the backgrounded ssh child inherits the stdout/stderr
   pipes → `run` blocks on them until the 15s timeout even after the connection succeeds
   (`open_master` always timed out). **Fix:** run a trivial `true` over the connection instead of
   `-nNf`; `ControlMaster=auto` + `ControlPersist=60` already keep the master socket alive, and the
   client exits immediately (still captures stderr on auth failure). Test `test_master_cmd_flags`
   updated (`-nNf` → asserts `true` last token, `-nNf` absent).
2. `ScenarioManager.start_scenario()`: on SIGINT during setup the signal handler calls `exit(0)`
   (`SystemExit`, caught by neither `except KeyboardInterrupt` nor `except Exception`), so the
   `finally`'s `_emergency_cleanup(success=success)` hit `UnboundLocalError: cannot access local
   variable 'success'`. **Fix:** initialize `success = False` before the `try`.

**Env note for reproducing the E2E:** scripts must run with the `cosim_gym` env bin on `PATH`
(i.e. `conda activate cosim_gym`, not just calling the env's python by absolute path) — the manager
spawns `helics_broker` (a CLI in the env bin) as a subprocess; without it on PATH, setup fails with
"helics_broker executable not found". Also note the demo's `conda run -n cosim_gym` resolves fine
over ssh on this machine (no `python:` escape hatch needed).

Full suite: `pytest tests/ -q` → **156 passed, 2 skipped** (includes T5's 5, ports' 10, updated
remote-executor master test). `graphify update .` run.

---

**T1, T2, T3, T4: DONE. T5 — Collection + cleanup: DONE. T7 — Docs: DONE.**

Full suite after T5: `pytest tests/ -q` → **145 passed, 2 skipped** (140 + 5 new T5 tests).

**T5 details:**

- `src/core/ScenarioManager.py`:
  - `start_scenario()`: added `self._collect_remote_results()` right after `_monitor_processes()`
    returns on the success path (before `simulation_end` metrics) — federates have all exited, so
    `sink: json` is written / `sink: parquet` finalized, safe to pull back.
  - `_emergency_cleanup()`: added `self._cleanup_remote_execution()` after the broker-kill loop,
    before `cleanup_end` metric. Runs on every exit path (success, exception, signal, atexit) via
    the existing once-only `_cleanup_done` guard.
  - New `_collect_remote_results()`: no-op if `remote_executors` empty. For each remote machine,
    `executor.collect()` the remote `results/<scenario>/<sim_id[-15:]>/` and
    `logs/<scenario_log_dir_rel>/` back into the identical local dirs (rsync, no `--delete` →
    collision-free merge since each federate writes distinct files). Any failure → ERROR log with
    a ready-to-paste manual `rsync` command (built from `machine_conf.user`/`host`), never raises —
    valid remote results must not turn a good run into a reported failure.
  - New `_cleanup_remote_execution()`: no-op if empty. Per machine: `executor.run(['pkill','-f',
    self.simulation_id])` (full sim id = unique per run → only this run's federates; rc=1/no-match
    ignored) then `executor.close()`, each wrapped in try/except → teardown never raises nor masks
    the original error. Clears `self.remote_executors`.
- `tests/test_scenario_manager_remote.py`: added `TestCollectionAndCleanup` (5 tests): collect
  no-op when local, collect issues 2 rsyncs (results+logs) with correct paths, collect swallows
  rsync failure, cleanup pkills `simulation_id` + closes + clears, cleanup never raises on ssh
  failure. Uses the same `object.__new__` fake-manager pattern; sets `simulation_id`/`scenario_name`.
- `graphify update .` run.

**T7 details:**

- New `docs/user_guide/distributed_deployment.md`: what-goes-where (brokers local, federates
  remote), one-time remote setup (ssh keys, conda env / `python:` escape hatch, firewall ports),
  full YAML reference, run lifecycle (preflight→deploy→spawn→collect→cleanup), localhost-as-remote
  demo instructions, troubleshooting table, LAN-trust security note, v1 limitations.
- `CLAUDE.md` Config Reference: added terse `deployment` block entry + `host` federate key.
- `docs/future_and_TODOs/distributed_ssh_spawning_plan.md`: status header updated to
  "T1–T5 + T7 implemented; T6 written, live E2E pending".

**T6 details (partial — code written, live run blocked):**

- New `src/scenarios/distributed_demo.yaml`: exact physics copy of `pv_batt_test_base.yaml` with a
  `deployment:` block (`manager_address: 127.0.0.1`, machine alias `local_box` → `127.0.0.1`,
  `workdir: /tmp/cosimgym_remote_test`, `conda_env: cosim_gym`, `python:` shown commented as the
  escape hatch) and `pv_federate` tagged `host: local_box`. Parses + validates cleanly through
  `read_scenario_config` (deployment present, manager_address set, alias valid).
- New `src/verify_distributed_demo.py`: runs the remote demo + its all-local twin
  (`pv_batt_test_base`), then compares every recorded federate timeseries (reuses
  `dashboard_data.load_all_records`) within `1e-9`. `--no-run` compares latest existing runs only.
  Import/entry verified (`--no-run` runs, correctly reports the demo hasn't executed yet).

## Original T1–T4 progress (retained for reference)

**T1, T2, T3: DONE. T4 — Spawn dispatch + address normalization: DONE.**

See git history / prior conversation for exhaustive T1–T4 file-by-file detail (config
dataclasses for `deployment:`/`host:`, `RemoteExecutor` class, spawn dispatch, address
normalization). Not reproduced here to keep this handoff focused on the CURRENT
(scaling-fix) work — the code is stable and unit-tested; re-read it directly if needed
rather than trusting a summary of a summary. `graphify query` first per the repo hook.

Constraints that apply to every remaining task (from plan doc, do not drop):
- Never commit without asking.
- `deployment:` absent → provably identical behavior to main branch.
- All ssh/rsync knowledge stays inside `RemoteExecutor`; `ScenarioManager` only decides where to spawn.
- No hardcoded hosts/paths/users/ports anywhere — everything from `DeploymentConfig`.
- Run `graphify query` before reading source files (project hook enforces this), `graphify update .`
  after code changes.
- Run `pytest tests/` + demo scenario before declaring any task done.
- Update this handoff file after each task.
