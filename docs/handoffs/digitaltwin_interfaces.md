# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

> **Process (in effect since M1):** the agent never runs `git commit`.
> It stages (`git add`) and hands off; **you** run the commit. A milestone's box in
> the Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last **committed** milestone

**M4 — OUTPUT then PARAMETER override** ✅ ticked. Commit: `47648ce`.
(Earlier: M3 `0cd8d15`, M2 `d9f0eba` (message says "M3", content is M2 — see
Blockers), M1 `b56fe1f`, M0 `3db608d`/`868fb6c`.)

## Staged, awaiting your commit: M5 — Live dashboard, BK4 demo, docs

**This is the last milestone of Plan 1.** Implemented and verified, staged not
committed.

- **`src/dashboard/live_dashboard.py`** (new) — a separate Streamlit page
  (does **not** touch `dashboard_app.py`, which stays the post-run historical
  explorer). Subscribes to `cosim/#` on Mosquitto via a background `paho-mqtt`
  client (same `CallbackAPIVersion.VERSION2` pattern as `mqtt_adapter.py`),
  cached per-session with `st.cache_resource` so the connection/buffers
  survive page reruns. Shows a "latest value per topic" table plus a
  multi-topic Plotly line chart, refreshed via a `time.sleep(N); st.rerun()`
  polling loop (no extra `streamlit-autorefresh` dependency). Works with
  **both** externalization mechanisms unmodified — a `streaming.stream: true`
  federate's mirror topics and any interface federate's `streams`/`bridges`
  topics all show up identically, since the dashboard only reads the shared
  JSON payload shape (`sim_id, key, value, sim_time, wall_time`) and doesn't
  care which mechanism produced it.
- **`src/dashboard/run_live_dashboard.sh`** (new) — launcher on port 8053
  (post-run dashboard stays on 8052), mirrors `run_dashboard.sh`'s conda-env
  guard.
- **BK4 demo pair** (the actual "config-only sim-to-real" demonstration):
  - `src/scenarios/m5_bk4_demo_a_full_sim.yaml` — `spring_federate` (base,
    `spring_mass_damper`) + `input_federate` (base, `inputs4spring` model:
    constant force=10, randomized disturbance). Fast, non-realtime, 20 ticks.
  - `src/scenarios/m5_bk4_demo_b_digital_twin.yaml` — **identical
    `spring_federate` block** (byte-for-byte the same subscription targets:
    `input_federate.0/force`, `input_federate.0/disturbance`). The **only**
    change is `input_federate`: `type: base` → `type: interface`, with
    `model_configs` replaced by `interface_config` (`mqtt_adapter`, two
    `scope: input`/`mode: replace` bridges registered at the **same** global
    publication keys the model federate used). Realtime-paced
    (`rt_lag`/`rt_lead: 1.0`) so there's a wall-clock window for an external
    process to feed it.
  - `src/scenarios/bk4_demo_external_sensor.py` (new) — stand-in "real
    hardware": a plain script (no CosimGym imports) publishing a sinusoidal
    force + randomized disturbance over MQTT to the two bridge topics, once a
    second. Demonstrates the bridge side is genuinely protocol-only — nothing
    about the sensor script is CosimGym-specific.
- **Docs:**
  - `docs/user_guide/digital_twin_interfaces.md` (new) — full design note:
    both mechanisms, the `scope: input/output/param` distinction, the BK4
    pattern, and the demo pair. Added to `mkdocs.yml` nav (User Guide, after
    "Reinforcement Learning Integration").
  - `docs/user_guide/dashboard.md` — new "Live View" section pointing at
    `run_live_dashboard.sh` and cross-linking the design note.
  - `CLAUDE.md` — new "Digital-Twin Interfaces & Live Streaming" subsection
    (mirrors how the RL config schema is documented there): `streaming`,
    `interface_config` (`streams`/`bridges`, `scope`, `mode`), the override
    registry, the BK4 pattern, and the live dashboard, all in one place.

**Verified:**
- `m5_bk4_demo_a_full_sim` run standalone (`PYTHONPATH=src python -c
  "from core.ScenarioManager import main; main('m5_bk4_demo_a_full_sim')"` from
  the repo root — see Blockers #9 on why *not* `cd src` first) — completed in
  2.6s, 2 federates.
- `m5_bk4_demo_b_digital_twin` run alongside `bk4_demo_external_sensor.py`
  (started ~3s in): completed in 21.5s (realtime-paced as expected). Recorded
  `spring_federate.0/force` timeseries: HELICS's `-1e+49` "unconnected" sentinel
  for the first ~9 ticks (before the sensor's first MQTT message landed —
  broker/subprocess/MQTT-connect startup overhead, same class of "wall-clock
  margin" issue as M3's timing gotcha, not a bug), then values `0.0, 4.43, 8.47,
  11.75, 13.98, 14.96, 14.61, 12.95, 10.13, 6.41, 2.12` — an **exact match** to
  the sensor script's own printed tick-by-tick output. `spring_federate`'s own
  YAML block is untouched between (a) and (b) — confirms the one-line BK4 swap.
- `live_dashboard.py` smoke-tested: `streamlit run ... --server.headless=true`,
  curled `http://localhost:8053` → `200`, no exceptions in the server log.
- Regression: `python src/test_script.py` (green, 4 federates,
  `dh_district_jan_base`) and `OMP_NUM_THREADS=1 python src/test_script_rl.py`
  (green, 3 brokers/3 federates, `bui0_heatingpower_DQN`).
- `python -m pytest tests/test_rl_config.py -q` → **78 passed, 1 skipped** (up
  from 76 — the two new BK4 demo YAMLs are picked up by the scenario
  parse-gate glob automatically).
- Cleaned up scratch `results/`/`logs/` for all four scenarios run during
  verification (`m5_bk4_demo_a_full_sim`, `m5_bk4_demo_b_digital_twin`,
  `dh_district_jan_base`, `bui0_heatingpower_DQN`).

**Your action:** review the staged diff, commit (e.g. `feat(digital-twin): M5
live dashboard, BK4 config-swap demo, docs`), then say continue — I'll tick M5.
**That completes Plan 1.**

## Next step (after you commit M5)

Plan 1 (M0-M5) is fully done. Plan 2 (`nonblocking_storage`, S0-S4) is a
**separate, independent effort** — per the plan, start it fresh off `main`
(`git checkout main && git checkout -b nonblocking_storage`), **not** built on
this branch. Its own handoff doc will be `docs/handoffs/nonblocking_storage.md`.
Not yet started. If you want to merge `digitaltwin_interfaces` into `main`
first, that's a separate decision or open question to make with the user —
not something to do unprompted.

## Files touched across M0-M5

**New:** `src/adapters/__init__.py`, `src/adapters/base_adapter.py`,
`src/adapters/mqtt_adapter.py`, `src/core/InterfaceFederate.py`,
`src/core/override_registry.py`, `src/mosquitto/mosquitto.conf`,
`src/dashboard/live_dashboard.py` (staged), `src/dashboard/run_live_dashboard.sh`
(staged), `docs/user_guide/digital_twin_interfaces.md` (staged), seven smoke/demo
scenarios (`m0`...`m4_interface_override_smoke_test.yaml` committed;
`m5_bk4_demo_a_full_sim.yaml`, `m5_bk4_demo_b_digital_twin.yaml`,
`bk4_demo_external_sensor.py` staged).
**Modified (committed M0-M4):** `environment.yml`, `src/core/federate_launcher.py`,
`src/core/mappings.yaml`, `src/docker-compose.yaml` (mosquitto @ host 11883),
`catalog_loader.py`, `catalog.yaml` (mqtt_adapter entry), `src/core/BaseFederate.py`,
`src/models/base_model.py`, `src/utils/config_dataclasses.py`, `tests/test_rl_config.py`.
**Modified (staged, M5):** `CLAUDE.md`, `docs/user_guide/dashboard.md`, `mkdocs.yml`.

## State of the tree

On `digitaltwin_interfaces`, 6 commits ahead of `main` (`3db608d`, `868fb6c`,
`b56fe1f`, `d9f0eba`, `0cd8d15`, `47648ce`). M5's changes are `git add`ed but
**uncommitted** — waiting on you. `git status --short` currently shows only
the M5 files staged plus an unrelated pre-existing modification to
`.claude/scheduled_tasks.lock` (not part of this plan — left untouched,
not staged).

## Blockers / deviations from the plan

1. **Process change (from M1 onward):** agent stages, user commits.
2. **Commit-message/milestone mismatch:** `d9f0eba` says "M3", contains M2's
   diff. Trust the diff / handoff / Progress Tracker, not commit messages.
3. Earlier deviations (branch-first ordering at M0, mosquitto port remap to
   11883, `model_configs` guard on `InterfaceFederateConfig`, drop-oldest
   outbound queue, the two M4 override-registry bugs) — see
   `git log -p -- docs/handoffs/digitaltwin_interfaces.md` for the full prior
   write-ups; summarized here so this doc stays a current pointer, not a
   history log.
4. **`BridgeSpec.helics_key` is overloaded by `scope`**: for `scope: input`
   it's a HELICS publication name; for `scope: output`/`param` it's an
   override-registry target string parsed by `parse_target` — no HELICS
   registration at all for those. Documented at each point of use; flagging in
   case it needs unifying/renaming later.
5. **Live dashboard uses a polling rerun (`time.sleep` + `st.rerun()`)**, not
   `streamlit-autorefresh` or websockets — avoids a new dependency, at the
   cost of the whole page redrawing every refresh tick rather than partial
   updates. Fine for a first live-view path; revisit if it needs to be
   smoother/less flickery.
6. **BK4 demo's first ~9 ticks show HELICS's `-1e+49` "unconnected" sentinel**
   for `spring_federate.0/force` before `bk4_demo_external_sensor.py`'s first
   MQTT message arrives (`mode: replace` bridges publish nothing until an
   external value shows up — same behavior M3 already established). This is
   expected/documented, not a bug — but if the demo is used for a live
   presentation, start the sensor script *before* the scenario, or add a few
   seconds of startup lead, to avoid an awkward sentinel-value stretch at the
   start.
7. **Redis logging noise (from M4, still not fixed):** `RedisClient.get_json`
   logs a WARNING on every absent-key lookup — noisy for every
   `override_enabled` federate's per-tick checks with no active override.
   Left untouched (shared utility); revisit with a `log_missing: bool` param
   if it becomes a problem in practice.
8. **Scenario run gotcha (new, hit during M5 verification):** running
   `ScenarioManager` from a `cd src` shell breaks federate subprocesses — they
   read config paths like `src/core/mappings.yaml` relative to the **repo
   root**, not `src/`. Always run from the repo root, either as
   `python src/test_script.py`-style (script itself under `src/`, cwd stays
   root) or `PYTHONPATH=src python -c "from core.ScenarioManager import main; ..."`
   from the root. A `cd src` first produces a `FileNotFoundError:
   'src/core/mappings.yaml'` in each federate's `.stdio.log`, while the
   manager process itself hangs "Monitoring N federates" indefinitely because
   the federates die immediately but the broker (started from the same,
   correct cwd) stays up — kill the stray broker/manager PIDs if this happens.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; M5 files staged, not committed
git log --oneline -7                       # 47648ce, 0cd8d15, d9f0eba, b56fe1f, 868fb6c, 3db608d on top of 38948b3 (main)
git diff --staged --stat                   # M5's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface/override are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M5 check — BK4 demo pair (run from repo root, NOT from src/ — see Blockers #8):
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m5_bk4_demo_a_full_sim')"
# then, in one terminal:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m5_bk4_demo_b_digital_twin')" &
# and within a couple seconds, in another:
python src/scenarios/bk4_demo_external_sensor.py --duration 18
# compare results/m5_bk4_demo_b_digital_twin/*/federation_1/spring_federate_test_storage.json
# inputs.spring_federate.0.force against the sensor script's printed values.

# Live dashboard:
./src/dashboard/run_live_dashboard.sh   # http://localhost:8053, while a stream/interface scenario runs

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 78 passed, 1 skipped

# Clean up scratch results/logs after verifying (all gitignored, local hygiene only):
rm -rf results/m5_bk4_demo_a_full_sim results/m5_bk4_demo_b_digital_twin \
       logs/m5_bk4_demo_a_full_sim logs/m5_bk4_demo_b_digital_twin
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. M0-M4 are committed; M5 (live dashboard, BK4
> config-swap demo pair, docs) is implemented and verified but staged, not
> committed — review and commit it yourself first (see 'Staged, awaiting your
> commit' above), then tell the agent to tick the M5 box. **That completes
> Plan 1** — the agent should stop there and wait for direction on Plan 2
> (`nonblocking_storage`, a separate branch/effort) rather than starting it
> unprompted. Remember: the agent stages changes and never runs `git commit`
> — you do."
