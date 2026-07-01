# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

> **Process change (mid-plan):** the agent no longer runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. The plan file's per-milestone
> loop and this doc were both updated to reflect that. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last **committed** milestone

**M0 — Shared scaffolding** ✅ ticked in the plan file. Commit: `3db608d`
(preceded by cleanup commit `626e03b`).

## Staged, awaiting your commit: M1 — `stream` flag outbound mirror

Implemented and verified, but **not committed** — `git status` on this branch
will show these staged (not the working tree, already `git add`ed):

- `src/adapters/mqtt_adapter.py` — real outbound path: `publish()` now enqueues
  into a bounded `queue.Queue` (drop-oldest when full), drained by a dedicated
  background thread (separate from paho's own `loop_start()` network thread)
  that calls `client.publish(topic, json.dumps(payload), qos=...)`.
- `src/core/BaseFederate.py`:
  - `__init__`: `self._stream_adapter = None` (lazy).
  - new `_stream_outbound()` method: no-op unless `config.streaming.stream` is
    `True`; respects `every_n_ticks`; lazily creates+connects an `MqttAdapter`
    (host/port from `MQTT_HOST`/`MQTT_PORT` env vars, default `localhost`/`11883`
    to match this repo's docker-compose mosquitto remap); publishes one message
    per input/output variable to topic
    `{stream_topic_prefix or f"cosim/{sim_id}/{federate_name}"}/{inputs|outputs}/{entity_id}/{var_name}`
    with payload `{sim_id, key, value, sim_time, wall_time}`.
  - called from `run()` right after `update_storage()`.
  - `finalize()` now closes `self._stream_adapter` if one was created.
- `src/scenarios/m1_stream_smoke_test.yaml` — new smoke-test scenario (spring +
  input federate, `streaming.stream: true` on `spring_federate`) kept as a
  permanent fixture; the config parse-gate test in `tests/test_rl_config.py`
  picks it up automatically (62 passed, 1 skipped — Adelaide known-broken YAML,
  pre-existing).

**Verified (see "How to verify" below for exact commands):**
- `mosquitto_sub` on `cosim/#` showed live JSON messages for every step while
  `m1_stream_smoke_test` ran (position/velocity/force/disturbance/acceleration,
  one message per var per tick).
- `python src/test_script.py` and `test_script_rl.py` both still match `main`
  exactly (streaming defaults to `False`, so the hook is a no-op there).
- `tests/test_rl_config.py`: 62 passed, 1 skipped.

**Your action:** review the staged diff, commit it (Conventional Commits, e.g.
`feat(digital-twin): M1 stream flag outbound MQTT mirror`), then tell me to
continue — I'll tick the M1 box in the plan file and move to M2.

## Next step (after you commit M1)

**M2 — Interface federate outbound (co-sim → external)**. Per the plan:
override `_register_connections`/`_register_pubs`/`_register_subs` on
`InterfaceFederate` to build HELICS subscriptions from `interface_config.streams`
(instead of the inherited empty-entities behavior from M0), relay each to the
adapter resolved from `interface_config.adapter` (catalog dynamic-import, same
mechanism `BaseFederate._register_entities` uses for physics models — see
`RedisCatalog.get_model_metadata`), and enable realtime pacing
(`flags.realtime` + `helics_property_time_rt_lag`/`rt_lead`, per
`BaseFederate.py:246-248` today only sets flags generically — an optional
timing field for rt_lag/rt_lead tuning needs adding to `_register_federate`).

First concrete action: read `src/core/InterfaceFederate.py` (currently just the
M0 shell) and start there.

## Files touched across M0+M1

**New (M0):** `src/adapters/__init__.py`, `src/adapters/base_adapter.py`,
`src/core/InterfaceFederate.py`, `src/mosquitto/mosquitto.conf`,
`src/scenarios/m0_interface_smoke_test.yaml`.
**New (M1, staged):** `src/scenarios/m1_stream_smoke_test.yaml`.
**Modified (M0, committed):** `environment.yml`, `src/core/BaseFederate.py`
(model_configs guard), `src/core/federate_launcher.py`, `src/core/mappings.yaml`,
`src/docker-compose.yaml` (mosquitto @ host 11883), `catalog_loader.py`,
`catalog.yaml` (mqtt_adapter entry), `src/utils/config_dataclasses.py`
(`StreamingConfig`/`InterfaceConfig`/`InterfaceFederateConfig`),
`tests/test_rl_config.py`.
**Modified (M1, staged):** `src/adapters/mqtt_adapter.py` (real `publish()`),
`src/core/BaseFederate.py` (`_stream_outbound` hook + finalize cleanup).

## State of the tree

On `digitaltwin_interfaces`, 2 commits ahead of `main` (`3db608d`, `868fb6c`).
M1's changes are `git add`ed but **uncommitted** — waiting on you.

## Blockers / deviations from the plan

1. **Process change:** agent stages, user commits (see banner at top) — added
   mid-plan by explicit user instruction; the plan file itself was updated to
   match (per-milestone loop, Handoff protocol, kickoff prompt).
2. **Branch-first ordering (M0):** cleanup commit (`HANDOFF.md` etc.) was done
   on `digitaltwin_interfaces`, not `main`, because the auto-mode classifier
   blocked deleting tracked files directly on `main`. `main` was never touched.
3. **Port conflicts on this shared server:** host `1883` is occupied by a
   pre-existing system-wide mosquitto service — ours is remapped to host
   `11883` (container side stays `1883`), same pattern as the existing MinIO
   9101 remap. `MqttAdapter`/catalog defaults and the `_stream_outbound` env
   vars (`MQTT_HOST`/`MQTT_PORT`) all default to `11883` to match.
4. **`ScenarioManager._enrich_dynamic_catalog_metadata`** required adding
   `model_configs: Optional[ModelConfig] = None` to `InterfaceFederateConfig`
   (M0, already committed) — not a plan deviation, just an undocumented detail.
5. Chose **paho's own `loop_start()` thread for network I/O** plus a **separate
   drain thread** for our bounded drop-oldest outbound queue (M1) rather than
   relying on paho's built-in `max_queued_messages_set` — paho's internal queue
   is reject-newest when full, not drop-oldest, which is wrong for a live
   telemetry mirror (you want the latest value, not the oldest queued one).

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; M1 files staged, not committed
git log --oneline -5                       # 868fb6c, 3db608d, 626e03b on top of 38948b3 (main)
git diff --staged --stat                   # M1's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M1 check — live MQTT mirror while streaming.stream:true:
mosquitto_sub -h localhost -p 11883 -t 'cosim/#' -C 8 -W 25 &
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m1_stream_smoke_test')"
# then: rm -rf results/m1_stream_smoke_test logs/m1_stream_smoke_test (gitignored, local hygiene only)

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 62 passed, 1 skipped
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. M0 is committed; M1 ('stream' flag outbound MQTT
> mirror) is implemented and verified but staged, not committed — review and
> commit it yourself first (see 'Staged, awaiting your commit' above), then
> tell the agent to tick the M1 box and continue to M2 (Interface federate
> outbound) exactly as scoped in the plan and in this handoff's 'Next step'.
> Remember: the agent stages changes and never runs `git commit` — you do."
