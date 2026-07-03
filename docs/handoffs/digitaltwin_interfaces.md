# Handoff — `digitaltwin_interfaces` (Plan 1) — COMPLETE

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`)

> **Process (in effect since M1):** the agent never runs `git commit`.
> It stages (`git add`) and hands off; **you** run the commit.

## Plan 1 status: all milestones committed ✅

M0 `3db608d`/`868fb6c` → M1 `b56fe1f` → M2 `d9f0eba` (message says "M3",
content is M2 — see Blockers) → M3 `0cd8d15` → M4 `47648ce` → M5 `797f618`.
Every box in the Progress Tracker for Plan 1 is ticked.

## Staged, awaiting your commit: post-M5 documentation pass

Not a new milestone — a follow-up pass making sure every Plan-1 feature
(`streaming`, `type: interface`, `interface_config`, override registry, BK4
demo pair, live dashboard) is discoverable from the project's existing doc
set, not just the dedicated design note added in M5. Staged, not committed:

- **`README.md`** — new bullet in Main Features.
- **`docs/Installation_Setup.md`** — Mosquitto/paho-mqtt noted alongside Redis
  in both "Start Infrastructure" steps (manual setup appears twice in this
  file); live-dashboard cross-link in the "Run Dashboard" steps.
- **`docs/overview/architecture.md`** — step 4 (Federate Launching) and step 5
  (Execution Loop) now mention `InterfaceFederate` and `streaming.stream`.
- **`docs/overview/core_concepts.md`** — new "Digital-Twin Interfaces & Live
  Streaming" section.
- **`docs/overview/terminology.md`** — new **Interface Federate** and
  **Adapter** terms.
- **`docs/user_guide/scenario_configuration/overview.md`** — `type` union and
  hierarchy diagram now include `interface`/`interface_config`; new row in the
  Sections table.
- **`docs/user_guide/scenario_configuration/federate.md`** — discriminator
  updated everywhere; `streaming`/`override_enabled` added to the common
  fields table; `rt_lag`/`rt_lead` added to the `timing_configs` table; two
  new sections: `streaming` (all types) and `type: "interface"` (full
  `interface_config` example + field table).
- **`docs/user_guide/custom_models.md`** — new "§4 Interface Adapters"
  subsection (the `interface_adapter` catalog category, same dynamic-import
  mechanism as physics models).
- **`docs/user_guide/running_scenarios.md`** — short "Digital-twin / streaming
  scenarios" note pointing at the design doc + the BK4 demo pair.
- **`docs/user_guide/troubleshooting.md`** — two new entries: the `cd src`
  cwd gotcha found during M5 verification (`FileNotFoundError:
  'src/core/mappings.yaml'` + manager hang), and "MQTT/digital-twin features
  silently produce no data" (Mosquitto not running / wrong port).

**Verified:** `mkdocs build --strict` — no broken links/anchors (fixed two
anchor-slug mismatches: `#` links must match the auto-generated slug, which
strips repeated hyphens differently than a naive guess — e.g.
`type-interface-interface-federate-digital-twin-bridge`, not
`type-interface--interface-...`). `pytest tests/test_rl_config.py` still 78
passed / 1 skipped (docs-only change, no config/code touched).

**Your action:** review the staged diff, commit (e.g. `docs(digital-twin):
surface streaming/interface federate features across README, overview,
scenario-config, custom-models, running-scenarios, troubleshooting docs`),
then say continue — I'll confirm the doc commit landed. This does not need a
Progress Tracker box (it's not a plan milestone), just confirmation before
the branch is considered fully wrapped up.

## Testing this branch end-to-end (quick guide)

```bash
cd /media/space/rando/CODE/CosimGym
conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# 1. Regressions — must be unaffected (everything in Plan 1 is opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# 2. Config/parse gate:
python -m pytest tests/test_rl_config.py -v      # expect 78 passed, 1 skipped

# 3. BK4 demo — the headline feature (config-only sim-to-real swap):
#    run from the REPO ROOT, never `cd src` first (see Blockers #8 below)
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m5_bk4_demo_a_full_sim')"
#    then, concurrently:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m5_bk4_demo_b_digital_twin')" &
sleep 3 && python src/scenarios/bk4_demo_external_sensor.py --duration 18
#    compare force in results/m5_bk4_demo_b_digital_twin/*/federation_1/spring_federate_test_storage.json
#    against the sensor script's printed tick values — they should match once messages arrive.

# 4. Live dashboard (open in a browser while step 3's run (b) or any
#    `streaming.stream: true` scenario is executing):
./src/dashboard/run_dashboard.sh    # http://localhost:8052 — open the "Live" page

# 5. Docs build check:
mkdocs build --strict -d /tmp/mkdocs_out

# Clean up scratch results/logs afterwards (gitignored, local hygiene only):
rm -rf results/m5_bk4_demo_a_full_sim results/m5_bk4_demo_b_digital_twin \
       logs/m5_bk4_demo_a_full_sim logs/m5_bk4_demo_b_digital_twin
```

## Next: Plan 2 (`nonblocking_storage`) — separate, independent branch

Plan 1 is fully done. Plan 2 is unrelated work (non-blocking Parquet storage)
on its **own branch off `main`**, per the plan file's own instructions —
**not** built on `digitaltwin_interfaces`. Kickoff commands for a fresh
session:

```bash
cd /media/space/rando/CODE/CosimGym
git checkout main
git pull                                    # if digitaltwin_interfaces has been merged/reviewed upstream
git checkout -b nonblocking_storage
mkdir -p docs/handoffs                      # docs/handoffs/nonblocking_storage.md will live here
docker compose -f src/docker-compose.yaml up -d
conda activate cosim_gym
python src/test_script.py                   # confirm green baseline before any change
```

Then resume at box **S0** in the plan file's Progress Tracker
(`sink` field on `MemoryConfig` + plumb through, default `json` = unchanged)
and follow the same per-milestone loop: implement → check → regression →
**stage, do not commit** → update `docs/handoffs/nonblocking_storage.md` →
wait for the user's commit + "continue" before ticking the box.

### One-line kickoff prompt for Plan 2's first fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> in full — Plan 1 (`digitaltwin_interfaces`) is complete, all milestones
> committed. Start Plan 2 (`nonblocking_storage`): create a new branch off
> `main` (`git checkout main && git checkout -b nonblocking_storage`,
> independent of the Plan-1 branch), confirm a green baseline
> (`python src/test_script.py`), then implement the lowest unchecked box
> (S0) in the Progress Tracker. Follow the per-milestone loop exactly: run
> the milestone's Check + regression test, then stage the change (`git add`)
> — do NOT commit, the user commits — write `docs/handoffs/nonblocking_storage.md`,
> and stop to wait for the user's commit + 'continue' signal before ticking
> the box and starting the next milestone."

## Blockers / deviations from the plan (Plan 1, for reference)

1. **Process change (from M1 onward):** agent stages, user commits.
2. **Commit-message/milestone mismatch:** `d9f0eba` says "M3", contains M2's
   diff. Trust the diff / handoff / Progress Tracker, not commit messages.
3. Branch-first ordering at M0, mosquitto port remap to 11883,
   `model_configs` guard on `InterfaceFederateConfig`, drop-oldest outbound
   queue — see `git log -p -- docs/handoffs/digitaltwin_interfaces.md` for
   the full prior write-ups.
4. **`BridgeSpec.helics_key` is overloaded by `scope`**: HELICS pub name for
   `scope: input`; an override-registry target string for `scope:
   output`/`param` (parsed by `parse_target`, no HELICS registration at all).
5. **Two M4 override-registry bugs**, both fixed and regression-tested:
   `parse_target` originally reconstructed entity as the bare instance number
   instead of `"federate.instance"`; output override wasn't written back into
   `self.outputs`, so storage didn't reflect it even though HELICS delivery
   was correct.
6. **Live dashboard uses `time.sleep` + `st.rerun()` polling**, not
   `streamlit-autorefresh` or websockets — avoids a new dependency at the
   cost of a full-page redraw each tick.
7. **BK4 demo's first ~9 ticks show HELICS's `-1e+49` sentinel** before the
   external sensor script's first MQTT message arrives (`mode: replace`
   publishes nothing until then) — expected, not a bug; start the sensor
   script first for a live demo/presentation to avoid this stretch.
8. **Never `cd src` before running `ScenarioManager`** — federate subprocesses
   resolve config paths relative to the **repo root**. `cd src` produces a
   `FileNotFoundError: 'src/core/mappings.yaml'` in each federate's
   `.stdio.log` while the manager hangs "Monitoring N federates" (federates
   died, broker didn't). Now documented in `docs/user_guide/troubleshooting.md`.
9. **Redis logging noise (from M4, still not fixed):** `RedisClient.get_json`
   logs a WARNING on every absent-key lookup — noisy for every
   `override_enabled` federate's per-tick checks with no active override.
   Revisit with a `log_missing: bool` param if it becomes a problem.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; docs pass staged, not committed
git log --oneline -7                       # 797f618, 47648ce, 0cd8d15, d9f0eba, b56fe1f, 868fb6c, 3db608d
git diff --staged --stat                   # the docs-pass diff
```

## One-line kickoff prompt to resume THIS branch (if the docs pass isn't committed yet)

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. Plan 1 (M0-M5) is fully committed. A follow-up
> documentation pass (README, overview docs, scenario-config reference,
> custom-models, running-scenarios, troubleshooting) is staged but not
> committed — review and commit it, then say continue. After that, Plan 1 on
> this branch is done; Plan 2 (`nonblocking_storage`) is a separate branch,
> see the 'Next: Plan 2' section above for kickoff commands."
