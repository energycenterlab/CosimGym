# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session, 2026-11-19, machine `eclab-cloud1`)

Post-crash recovery, part 2: complete the `cosim_gym` env, check port conflicts with the other
project's (`dtp`) containers, verify the Claude setup, build the graphify **docs** graph and
fix the doc inconsistencies it surfaced; investigate the helics/torch segfault; move MQTT off
the conflicting port. **Docker containers were NOT started** (author's instruction).
**Nothing is committed.**

## Current Progress — done, uncommitted

### Environment (`/opt/conda/envs/cosim_gym`; conda = `/opt/conda/bin/conda`, not on PATH)
- `environment.yml`: added `ray[rllib]`, `pandapower`, `pandapipes`, `ijson`, `pytest`; env updated,
  all third-party imports resolve, `pip check` clean. Side downgrades forced by the new deps:
  gymnasium 1.4→1.2.2 (ray), pandas 3.1→2.3.3 and scipy 1.18→1.16.3 (pandapower/pandapipes).
  Pre-update freeze was in the session scratchpad (may be gone).
- torch is **not** pinned (2.15.0+cu132). Machine has **no GPU**: CUDA torch = several GB of
  useless nvidia libs — a CPU wheel would do (not changed; author to decide).

### helics/torch segfault — understood, KNOWN_ISSUES #7 rewritten
- `import helics` *before anything that loads `libstdc++.so`* → `import torch` / `import ray`
  segfaults. Cause: the helics wheel's `libhelics.so` statically contains an old (CentOS 7 GCC)
  libstdc++ and exports its symbols into the global scope; the env's real libstdc++ loaded later
  binds ~120 of its own symbols to helics' copies (`LD_DEBUG=bindings`) → two half-merged C++
  runtimes.
- Loading numpy/pandas/torch first is safe for helics too: verified with a real tcp broker +
  2 federates pub/sub in every order (backward-compatible newer libstdc++).
- Pinning older torch does NOT fix it (torch 2.5.1 → `GLIBCXX_3.4.29 not found`; helics 3.5.3
  same crash). Real runs are safe today **by accident**: `RL_Federate.py:13` imports numpy
  before helics (`:16`). An import sorter would break it.

### MQTT host port 11883 → 11884 (11883 is taken by `dtp-mosquitto-1`)
- Single source of truth = `utils.ports.mqtt_port()` (`COSIM_MQTT_PORT` / `src/.env`, default
  11884). Changed: `src/utils/ports.py`, `src/docker-compose.yaml` fallback, `src/.env.example`,
  `MqttAdapter` (`port=None` → `mqtt_port()`), `catalog.yaml` mqtt_adapter `port` default → `null`,
  `live_dashboard.py` and `bk4_demo_external_sensor.py` (both now `sys.path` src + `mqtt_port()`),
  removed hardcoded `port: 11883` from m2 (×2), m3, m4, m5, cs_s5_dt YAMLs, `tests/test_ports.py`,
  user docs. Before this, interface federates ignored `COSIM_MQTT_PORT`.
- `cs_s1_models_*` contain `...federate.11883/...` = federate **indices**, not ports — untouched.
- **catalog.yaml changed → reload it into Redis** when containers are next started.
- Other ports checked free: Redis 6379, MinIO 9000/9101, dashboard 8052 (and 8501 via
  `make run-dashboard`), HELICS 20000–30000. No `src/.env` exists.

### Claude setup — verified OK
CRG MCP connected + graph current, pyright works, settings JSON valid, git hooks installed,
claude-sync cron OK, `graphify-out/.graphify_python` fixed to the uv tool python.
Notes: `Bash(python:*)` in the allow list ≈ allow anything; claude-mem's MCP instructions push
its own to-do tool (CLAUDE.md overrides).

### graphify docs graph
Built (62 docs, 431 nodes, 624 edges, 25 communities) — replaced the old code graph.
Cost ≈725k subagent tokens (mostly the added code-consistency audit). `docs/KNOWN_ISSUES.md` is
cached as partial; **docs changed a lot this session → graph is stale; `--update` only on
author request.**

### Docs fixed (~30 files, each claim checked against code)
Scenario-config reference (`core_type` default `tcp` + one-protocol-per-scenario rule, broker
`host` 127.0.0.1, auto ports from range, federate `log_level` = INFO not inherited, `deployment`
/ `host` keys, `multi_computer` legacy, validation `extra='ignore'` caveat), reset semantics in
`rl_integration.md`, broken links (`distributed_ssh_spawning_plan.md`, `handoffs/...SUMMARY.md`,
`influxdb_client.py`, "Ports section in CLAUDE.md"), dashboard 8501 vs 8052 explained,
docker-compose v1 fallback removed, Redis/ufw guidance reconciled, models_reference structure,
scenario catalogues, stale TODO items pruned (Bugs C/E, orphan cleanup, pandapower 7.4,
sim_start_date, followups §5, uv drift, Phase D), graphify-on-code instructions, RUNBOOK
`/media/space` paths, CLAUDE.md "typed Pydantic models" wording.
KNOWN_ISSUES: #6 line ref fixed, #7 rewritten, new #8–#12 (default_subscription_causality
ignored; RL Bugs A/B/D — A is masked in the suite by `_shorten`'s forced checkpoint; RL keys
parsed-not-implemented; B12 teardown stall; B13 worker oversubscription). `Last reviewed` → 2026-11-19.
`conda activate` in user docs deliberately left (correct for interactive shells).

### Tests
`pytest tests/ --ignore=tests/regression_suite.py`: **310 passed, 2 failed (the known
KNOWN_ISSUES #6), 2 skipped**, 6m43s. Unit tests need no Docker. Regression suite NOT run.

## What Didn't Work / Watch Out
- Reading `src/.env` is blocked by a permission rule — don't retry (the file doesn't exist anyway).
- HELICS wheel has no `test` core; `--interface` is not a valid broker arg. Use `tcp` + `--port`.
- `LD_DEBUG_OUTPUT` files: parse with `awk '{print $4,$NF}'`, not regex on backticks.
- graphify coverage check: files without trailing newline give false "partial" (wc -l off by one).

## Decisions taken (author, end of session)
1. **helics import order → made explicit (DONE, uncommitted):** `BaseFederate.py`,
   `RL_Federate.py`, `InterfaceFederate.py` import `numpy` right before `helics` inside
   `# isort: off/on`; verified each module then `import torch, ray` works. KNOWN_ISSUES #7
   downgraded to LOW/mitigated. Torch left unpinned.
2. **Commits:** the author commits (session work up to `7872bb8` already committed).
3. **Containers + regression suite:** the author starts the containers, then asks Claude to run
   the suite. Before that: passwordless ssh to 127.0.0.1 is NOT working on this machine
   (`Host key verification failed`) → the `DIST*` scenarios will fail until the author fixes it
   (Claude must not touch `~/.ssh`). `catalog-loader` reloads `catalog.yaml` into Redis on
   every `docker compose up`.

## Done after the decisions (uncommitted)
- **New-machine portability (code):** `RemoteExecutor._python_cmd` — loopback host without
  `python:` uses the manager's `sys.executable` (+ test); `BaseFMUModel._resolve_fmu_path` resolves
  relative `fmu_source.path` against the repo root; `catalog.yaml` BUI0/Feedthrough FMU paths made
  repo-relative (were `/media/space/...` → FMU scenarios would fail); removed old-machine `python:`
  from `distributed_multifederation_test` / `combo_dist_parallel`. 48 related tests pass.
  Still machine-specific: `~/miniconda3` `python:` paths in the RUN_CLOUD scenarios
  (`distributed_demo_multi`, `benchmark_scale_distributed*`, `cs_s4c_shard_*`) and
  `/media/space` in `scripts/scaling_study/plot_*.py`.
- KNOWN_ISSUES **#13** (new machine: SSH incl. 127.0.0.1). Guided **"Adding a machine
  (checklist)"** in `docs/user_guide/distributed_deployment.md` (case A new manager, case B new remote).
- New **`docs/developer_guide.md`** (all commands in one page), linked from CLAUDE.md + mkdocs nav;
  nav now also lists Distributed Deployment + Multi-Machine Walkthrough. `mkdocs build` clean
  (only pre-existing links from scaling_study_plan into `scripts/`).
- **Doc renames (`git mv`, links fixed):** DONE plan → `changes_reports/`; alternatives →
  `changes_reports/distributed_ssh_spawning_changes.md`; `pandapower_grid_followups.md`,
  `remote_brokers_plan.md`, `scenario_yaml_TODO.md`; new `distributed_deployment_followups.md`
  (open items moved out of the DONE plan).
- **Scaling study:** author says it was exploratory; will be redone as future work for the paper.
  All `scripts/scaling_study/findings/*.md`, `EXPERIMENTS.md` got a "not validated, do not cite"
  banner; `scaling_study_plan.md` marked as future work. Numbers NOT reconciled (on purpose).
- Walkthrough real manager IP → `<MANAGER_IP>`; co-user account names removed; index.md typos.

## Next Steps
1. Author: fix localhost ssh, `docker compose -f src/docker-compose.yaml up -d`, then ask Claude
   to run `tests/regression_suite.py` (~30+ scenarios, long: needs the go-ahead, which is given
   by that request).
2. Item 4 below is resolved (renames done, scaling marked exploratory).
3. (was: commit) — author handles commits.
4. **Docs left for the author:** file renames/moves violating the doc rules
   (`distributed_ssh_spawning_alternatives.md`, `pandapower_extension.md`,
   `phaseH_remote_brokers_design.md`, `scenario_yaml_fixes_needed.md`,
   `distributed_ssh_spawning_plan_DONE.md` in future_and_TODOs, `scripts/scaling_study/{EXPERIMENTS.md,findings/README.md,findings/all_phases_synthesis.md,findings/bottlenecks.md}`,
   `src/scenarios/to_review_rolling_window/README.md`); **scaling-study numeric contradictions**
   (456 vs 386 runs, R² 0.97 vs 0.815, 300 vs 143 MB/federate, phase01 vs phase5 fit values,
   `paper_ready_sentences.md` medians labelled as means + 3.66/4.90 vs 3.14/4.36 µs + "≈100 crossover");
   stale line-number anchors in plans (not fixed, low value); hardcoded real IP in
   `multi_machine_test_walkthrough.md`; other users' account names in `bottlenecks.md`.
5. Minor code nit found: federate `core_type` default `"tcp"` makes every zmq scenario log one
   "Overriding core_type" warning per federate (`ScenarioManager.py:~1795`).
6. Carried over: reset refactor **F5** (parallel reset forwarding) + **F6** (closure) in
   `docs/changes_reports/model_reset_redesign_plan.md`; open question: one
   `fmu_output/restart_<n>/` per restart → reuse one dir per instance? 9 latent bugs (native
   memory `latent-bugs-codebase-read`) still not in KNOWN_ISSUES — decide with the author.
7. On the Mac: `uv tool install pyright code-review-graph graphifyy`; recreate/update the conda
   env from the new `environment.yml`.
