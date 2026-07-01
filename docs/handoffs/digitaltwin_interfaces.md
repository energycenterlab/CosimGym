# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

> **Process (mid-plan change, in effect since M1):** the agent never runs `git commit`.
> It stages (`git add`) and hands off; **you** run the commit. A milestone's box in
> the Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last **committed** milestone

**M1 — `stream` flag outbound mirror** ✅ ticked. Commit: `b56fe1f`.
(M0 — `3db608d`, `868fb6c`.)

## Staged, awaiting your commit: M2 — Interface federate outbound (co-sim → external)

Implemented and verified, **staged not committed**:

- `src/core/InterfaceFederate.py` — full rewrite for M2:
  - `_register_entities()`: resolves the adapter from `interface_config.adapter`
    via the catalog (`self.catalog.get_model_metadata(name)` → dynamic import,
    same mechanism `BaseFederate._register_entities` uses for physics models),
    instantiates it with `{**catalog_defaults, **interface_config.adapter.params}`,
    calls `.connect()`. Returns `[]` (still no physics entities).
  - `initialize()`: now keys `self.inputs`/`self.outputs`/`self._deferred_inputs`
    by `self.name` (the federate has no physics entities, so it's its own
    implicit "entity" for bookkeeping).
  - `_register_connections()`: builds one HELICS global input per
    `interface_config.streams[i]` (topic name `f"{self.name}/stream_{i}"`,
    target = `stream.helics_key`) — the streams-only half of what the plan
    calls "build HELICS pubs/subs from an `interface_config` block"; `bridges`
    (external → co-sim pubs) is M3/M4.
  - `_receive_inputs()`: calls `super()` then, per stream (respecting
    `every_n_ticks`), reads the just-updated value and calls
    `self._adapter.publish(stream.topic, {sim_id, key: stream.helics_key,
    value, sim_time, wall_time})`.
  - `finalize()`: closes `self._adapter` before calling `super().finalize()`.
- `src/utils/config_dataclasses.py`:
  - `FedTimingConfig` gained `rt_lag`/`rt_lead: Optional[float] = None`.
  - `StreamSpec` gained `type: str = "double"` and `units: str = ""` (needed to
    register the HELICS input; both optional/defaulted, additive not breaking).
- `src/core/BaseFederate.py` — `_register_federate()`: if `timing_configs.rt_lag`/
  `rt_lead` are not `None`, sets `helics_property_time_rt_lag`/`rt_lead` on the
  federate info (opt-in, `None` default = unchanged HELICS behavior everywhere
  else).
- `src/scenarios/m2_interface_outbound_smoke_test.yaml` — new fixture: spring +
  input federates unchanged, plus `dt_bridge_relay` (`type: interface`,
  `flags.realtime: true`, `rt_lag/rt_lead: 1.0`) subscribing to
  `spring_federate.0/position` and relaying it to
  `cosim/m2_smoke/relay/position`.
- `tests/test_rl_config.py` — `TestInterfaceOutboundConfig` (3 new tests:
  `StreamSpec` defaults, `rt_lag`/`rt_lead` default-None and explicit).

**Verified (see "How to verify" below for exact commands):**
- `mosquitto_sub -t 'cosim/m2_smoke/#'` showed spring's `position` relayed out
  once per tick, `wall_time` spaced ~1s apart across 5 ticks (real_period=1,
  realtime pacing active) — `simulation_duration` was 7.6s for a 5-tick scenario
  vs ~2.5s unpaced in M0/M1 fixtures, confirming realtime is actually engaged.
- Model federates (`spring_federate`/`input_federate`) wrote their usual result
  JSONs unchanged; `dt_bridge_relay` wrote none (still a storage no-op).
- `python src/test_script.py` and `test_script_rl.py` unchanged vs main.
- `tests/test_rl_config.py`: 66 passed, 1 skipped.

**Your action:** review the staged diff, commit (e.g.
`feat(digital-twin): M2 interface federate outbound relay + realtime pacing`),
then say continue — I'll tick M2 and move to M3.

## Next step (after you commit M2)

**M3 — Interface federate inbound = INPUT injection (external → co-sim)**:
- `MqttAdapter.subscribe()`/`.latest()`: implement inbound — subscribe to
  `interface_config.bridges[*].topic`, keep a lock-guarded latest-value dict
  fed by `on_message`, exposed via `latest(topic)`.
- `InterfaceFederate`: register a HELICS **publication** per `bridges[i]`
  (`scope: input` only for M3 — output/param scopes are M4) targeting
  `bridges[i].helics_key`; each step, read `adapter.latest(bridges[i].topic)`,
  clip to `bridges[i].bounds` if set, and `pub.publish(value)` before the
  model federates read it. `mode: replace` (repoint target) vs `mode:
  passthrough` (subscribe the real source too, only override when an MQTT
  message has actually arrived) — passthrough needs a "have we ever received
  anything on this topic" check, `latest()` returning `None` until first
  message is the natural signal for that.
- **Check:** `mosquitto_pub` a sensor value mid-run on a bridge's topic; the
  target federate's input follows it (clipped to bounds); before any message
  arrives, `mode: passthrough` falls back to the real source unchanged.

First concrete action: implement `MqttAdapter.subscribe()`/`latest()` (currently
`raise NotImplementedError`), then wire `InterfaceFederate._register_connections()`
to also build `bridges` pubs alongside the M2 `streams` subs.

## Files touched across M0+M1+M2

**New:** `src/adapters/__init__.py`, `src/adapters/base_adapter.py`,
`src/core/InterfaceFederate.py`, `src/mosquitto/mosquitto.conf`,
`src/scenarios/m0_interface_smoke_test.yaml`,
`src/scenarios/m1_stream_smoke_test.yaml` (all committed),
`src/scenarios/m2_interface_outbound_smoke_test.yaml` (staged).
**Modified (committed M0+M1):** `environment.yml`, `src/core/federate_launcher.py`,
`src/core/mappings.yaml`, `src/docker-compose.yaml` (mosquitto @ host 11883),
`catalog_loader.py`, `catalog.yaml` (mqtt_adapter entry).
**Modified (staged, M2):** `src/core/BaseFederate.py` (rt_lag/rt_lead wiring —
on top of the already-committed model_configs guard + `_stream_outbound` hook),
`src/utils/config_dataclasses.py` (rt_lag/rt_lead, StreamSpec type/units),
`tests/test_rl_config.py`.

## State of the tree

On `digitaltwin_interfaces`, 3 commits ahead of `main` (`3db608d`, `868fb6c`,
`b56fe1f`). M2's changes are `git add`ed but **uncommitted** — waiting on you.

## Blockers / deviations from the plan

1. **Process change (from M1 onward):** agent stages, user commits.
2. **Branch-first ordering (M0):** cleanup commit done on `digitaltwin_interfaces`,
   not `main` — classifier blocked deleting tracked files directly on `main`.
3. **Port conflicts on this shared server:** mosquitto remapped to host `11883`
   (container side stays `1883`) — `1883` is owned by a pre-existing system-wide
   mosquitto service. All adapter/env-var defaults (`MQTT_HOST`/`MQTT_PORT`,
   catalog `mqtt_adapter` params) point at `11883`.
4. **`ScenarioManager._enrich_dynamic_catalog_metadata`** required
   `model_configs: Optional[ModelConfig] = None` on `InterfaceFederateConfig` (M0).
5. **M1's outbound queue** uses our own bounded drop-oldest `queue.Queue` +
   dedicated drain thread rather than paho's built-in `max_queued_messages_set`
   (which is reject-newest, not drop-oldest — wrong for live telemetry).
6. **M2's HELICS input topic naming**: interface federate's own subscription
   topics are synthetic (`f"{self.name}/stream_{i}"`), not derived from the
   variable name — this is fine because both the write side
   (`BaseFederate._receive_inputs`, inherited via `super()`) and the read side
   (`InterfaceFederate._receive_inputs`) derive the same dict key
   (`subid.name.split('/')[-1]`) from the same `subid`, so they stay consistent;
   the *actual* variable identity for downstream consumers (MQTT payload,
   logging) comes from `stream.helics_key`/`stream.topic`, not from that
   synthetic key.
7. Realtime pacing is opt-in per-federate (HELICS allows mixed realtime/non-realtime
   federates in one federation) — only `dt_bridge_relay` is paced in the M2
   fixture; `spring_federate`/`input_federate` are not, and that's expected/fine
   since the bridge has no dependents relying on its timing.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; M2 files staged, not committed
git log --oneline -5                       # b56fe1f, 868fb6c, 3db608d on top of 38948b3 (main)
git diff --staged --stat                   # M2's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M2 check — bridge relays spring's position out to MQTT, paced to wall-clock:
mosquitto_sub -h localhost -p 11883 -t 'cosim/m2_smoke/#' -C 6 -W 20 &
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m2_interface_outbound_smoke_test')"
# expect ~5-7s simulation_duration (paced) vs ~2.5s if unpaced; wall_time in the
# captured messages should be ~1s apart.
# then: rm -rf results/m2_interface_outbound_smoke_test logs/m2_interface_outbound_smoke_test (gitignored, local hygiene only)

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 66 passed, 1 skipped
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. M0/M1 are committed; M2 (interface federate outbound
> relay + realtime pacing) is implemented and verified but staged, not committed
> — review and commit it yourself first (see 'Staged, awaiting your commit'
> above), then tell the agent to tick the M2 box and continue to M3 (interface
> federate inbound / INPUT injection) exactly as scoped in the plan and in this
> handoff's 'Next step'. Remember: the agent stages changes and never runs
> `git commit` — you do."
