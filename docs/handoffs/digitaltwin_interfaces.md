# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

> **Process (mid-plan change, in effect since M1):** the agent never runs `git commit`.
> It stages (`git add`) and hands off; **you** run the commit. A milestone's box in
> the Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last **committed** milestone

**M2 — Interface federate outbound (co-sim → external)** ✅ ticked. Commit: `d9f0eba`
(message says "M3 is done" but its diff is actually M2's scope — outbound relay +
realtime pacing, no bridges/inbound code at all. Flagging so you're not confused
by `git log` — the Progress Tracker checkboxes are the source of truth for what
landed where, not individual commit messages.)
(Earlier: M1 — `b56fe1f`. M0 — `3db608d`, `868fb6c`.)

## Staged, awaiting your commit: M3 — Interface federate inbound = INPUT injection

Implemented and verified, **staged not committed**:

- `src/adapters/mqtt_adapter.py`: `subscribe()` now calls `client.subscribe()` per
  topic; `latest()` reads a lock-guarded `_inbound` dict; a new `_on_message`
  callback (registered in `__init__` alongside `on_connect`) writes incoming
  messages into that dict, JSON-decoded (raw bytes under `{'value': ...}` if not
  valid JSON).
- `src/utils/config_dataclasses.py` — `BridgeSpec` gained `type`/`units` (like
  `StreamSpec` in M2, needed to register the HELICS publication) and
  `source_key: Optional[str]` + a validator requiring it when `mode ==
  "passthrough"`. Clarified the docstring: `helics_key` is the bridge's own
  **publication name** (what a model federate's subscription should target),
  not a subscribe target — this reading wasn't 100% explicit in the plan's
  YAML example, see Blockers #6.
- `src/core/InterfaceFederate.py`:
  - `_register_connections()`: now also builds one HELICS global publication
    per `interface_config.bridges[i]` where `scope == "input"` (`output`/`param`
    scopes are logged and skipped — M4). `mode: passthrough` bridges additionally
    register a HELICS input subscribed to `source_key`. Collects all bridge
    topics and calls `self._adapter.subscribe(topics)` once, after the adapter
    exists.
  - New `_publish_outputs()` override: for each bridge, checks
    `adapter.latest(bridge.topic)` first; if present, uses `['value']`; else
    (only for `mode: passthrough`) falls back to
    `self._read_subscription_value(pub['source_subid'])`; else skips (nothing
    to publish yet, `mode: replace` with no external value). Clips to
    `bridge.bounds` if set, `pub['pubid'].publish(value)`.
- `src/scenarios/m3_interface_inbound_smoke_test.yaml` — new fixture:
  `spring_federate`'s `force` subscription now targets `dt_bridge_inbound/force`
  (not `input_federate` directly); `dt_bridge_inbound` bridges that key in
  `mode: passthrough` from real source `input_federate.0/force` (constant 10.0),
  bounds `[0, 20]`.

**Verified (see "How to verify" below for exact commands):** ran the scenario
in the background, waited ~10s into a paced 25-tick run, then
`mosquitto_pub`'d `{"value": 25}` on the bridge's topic. Spring's recorded
`force` timeseries: `10.0` (real passthrough, unclipped) for the ticks before
the publish, `20.0` (external 25 clipped to bounds `[0,20]`) for every tick
after — clean, unambiguous transition. `test_script.py`/`test_script_rl.py`
unchanged vs main; `tests/test_rl_config.py`: 70 passed, 1 skipped.

**Your action:** review the staged diff, commit (e.g.
`feat(digital-twin): M3 interface federate inbound INPUT injection`), then say
continue — I'll tick M3 and move to M4.

## Next step (after you commit M3)

**M4 — OUTPUT then PARAMETER override**:
- **Output override:** per the plan, a guarded hook in
  `BaseFederate._publish_outputs` (around `BaseFederate.py:716` pre-M0, check
  current line — several edits have shifted it) — if an external override
  exists for a given (entity, output-var) key, substitute it (bounds-clipped)
  before `pub['pubid'].publish(data)`. This is the `scope: output` case that
  `InterfaceFederate._register_connections` currently skips with a log message.
  Needs a way for a **model federate** (not the interface federate) to look up
  "is there an active override for my output X" — likely via a small shared
  registry (Redis key or a module-level dict keyed by
  `(simulation_id, federation, federate, entity, var)`) that
  `InterfaceFederate._publish_outputs` writes to instead of/alongside
  publishing its own HELICS key, since for `scope: output` the override target
  is *someone else's* output, not a HELICS pub/sub at all.
- **Param override:** `set_parameter(name, value)` on `BaseModel` (bounds-checked
  vs catalog `parameters.min/max`), routed through the same channel, called
  each step (or on change) from the target model federate before `model._step()`.
- **Check:** force a federate's output, then a parameter, from MQTT mid-run;
  illegal values rejected/clipped; disabling restores computed behavior.

First concrete action: decide and implement the override-registry mechanism
(Redis is the natural fit — reuse the existing `RedisClient` already used for
config distribution, e.g. `cosim:override:<sim_id>:<federation>:<federate>:<entity>:<var>`)
before touching `BaseFederate._publish_outputs`/`BaseModel.set_parameter`.

## Files touched across M0-M3

**New:** `src/adapters/__init__.py`, `src/adapters/base_adapter.py`,
`src/core/InterfaceFederate.py`, `src/mosquitto/mosquitto.conf`, four smoke-test
scenarios (`m0_interface_smoke_test.yaml`, `m1_stream_smoke_test.yaml`,
`m2_interface_outbound_smoke_test.yaml` — all committed,
`m3_interface_inbound_smoke_test.yaml` — staged).
**Modified (committed M0-M2):** `environment.yml`, `src/core/federate_launcher.py`,
`src/core/mappings.yaml`, `src/docker-compose.yaml` (mosquitto @ host 11883),
`catalog_loader.py`, `catalog.yaml` (mqtt_adapter entry).
**Modified (staged, M3):** `src/adapters/mqtt_adapter.py` (inbound
subscribe/latest/on_message), `src/utils/config_dataclasses.py` (BridgeSpec
type/units/source_key), `tests/test_rl_config.py`.

## State of the tree

On `digitaltwin_interfaces`, 4 commits ahead of `main` (`3db608d`, `868fb6c`,
`b56fe1f`, `d9f0eba`). M3's changes are `git add`ed but **uncommitted** —
waiting on you.

## Blockers / deviations from the plan

1. **Process change (from M1 onward):** agent stages, user commits.
2. **Commit-message/milestone mismatch:** commit `d9f0eba` ("M3 is done") is
   actually M2's diff — no functional issue, just don't trust that commit's
   message for content; trust the diff / this handoff / the Progress Tracker.
3. **Branch-first ordering (M0):** cleanup commit done on `digitaltwin_interfaces`,
   not `main`.
4. **Port conflicts:** mosquitto @ host `11883` (system mosquitto owns `1883`
   on this shared server).
5. **`ScenarioManager._enrich_dynamic_catalog_metadata`** required
   `model_configs: Optional[ModelConfig] = None` on `InterfaceFederateConfig` (M0).
6. **M1's outbound queue** is our own bounded drop-oldest queue + drain thread,
   not paho's `max_queued_messages_set` (which is reject-newest).
7. **M2's synthetic HELICS topic naming** (`f"{self.name}/stream_{i}"`) is
   internally consistent (see prior handoff) but cosmetic-only; real identity
   comes from `stream.helics_key`/`topic`.
8. **`BridgeSpec.helics_key` semantics (M3):** interpreted as the bridge's own
   HELICS **publication name**, not a subscribe target — the plan's YAML
   comment ("key a model federate subscribes to") supports this reading but
   doesn't spell it out as unambiguously as `StreamSpec.helics_key`'s opposite
   role (a subscribe target). If a future milestone needs it to mean something
   else, this is the place to revisit.
9. **Added `BridgeSpec.source_key`** — the plan's YAML example for `mode:
   passthrough` doesn't show how the "real source" is named; added this field
   (required only when `mode == "passthrough"`) as the natural minimal
   interpretation of "subscribe real source + override".
10. **Known first-tick artifact:** the very first recorded value on a
    passthrough bridge can read `0.0` instead of the real source's actual value
    (self-heals by tick 2) — an initialization-phase ordering race between the
    bridge's `_publish_init_state()` and the real source federate's own init
    publish, both happening before regular time-stepping begins. Not fixed in
    M3 (out of this milestone's scope); worth a look if a later milestone needs
    tick-1 fidelity.
11. **Test timing gotcha (process note, not a code issue):** when scripting a
    "publish mid-run" check against a realtime-paced scenario, budget generous
    margin — process/import/broker startup overhead easily eats several
    seconds before the sim's own tick loop begins, so a `sleep N` measured from
    process launch needs `N` well short of the *scenario's total duration*,
    with the actual publish command issued only after confirming (or generously
    assuming) the sim is still mid-flight. The first M3 attempt published after
    the sim had already finished and silently no-opped.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; M3 files staged, not committed
git log --oneline -6                       # d9f0eba, b56fe1f, 868fb6c, 3db608d on top of 38948b3 (main)
git diff --staged --stat                   # M3's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M3 check — passthrough then override, clipped:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m3_interface_inbound_smoke_test')" > /tmp/m3.log 2>&1 &
RUNPID=$!
sleep 12
mosquitto_pub -h localhost -p 11883 -t 'cosim/m3_smoke/sensor/force' -m '{"value": 25}'
wait $RUNPID
python3 -c "
import json, glob
p = sorted(glob.glob('results/m3_interface_inbound_smoke_test/*/federation_1/spring_federate_test_storage.json'))[-1]
data = json.load(open(p))
print(data['inputs']['spring_federate.0']['force'])
"
# expect: 0.0, then 10.0 repeated (passthrough), then 20.0 repeated from around
# the tick the publish landed (25 clipped to bounds [0,20])
# then: rm -rf results/m3_interface_inbound_smoke_test logs/m3_interface_inbound_smoke_test (gitignored, local hygiene only)

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 70 passed, 1 skipped
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. M0-M2 are committed; M3 (interface federate inbound
> INPUT injection) is implemented and verified but staged, not committed —
> review and commit it yourself first (see 'Staged, awaiting your commit'
> above), then tell the agent to tick the M3 box and continue to M4 (output
> then parameter override) exactly as scoped in the plan and in this handoff's
> 'Next step'. Note: commit `d9f0eba`'s message says 'M3' but its content is
> actually M2 — trust the diff, not the message. Remember: the agent stages
> changes and never runs `git commit` — you do."
