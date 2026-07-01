# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

> **Process (mid-plan change, in effect since M1):** the agent never runs `git commit`.
> It stages (`git add`) and hands off; **you** run the commit. A milestone's box in
> the Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last **committed** milestone

**M3 — Interface federate inbound = INPUT injection** ✅ ticked. Commit: `0cd8d15`.
(Earlier: M2 — `d9f0eba` (commit message says "M3", content is M2 — see prior
handoff revision / `git log` note). M1 — `b56fe1f`. M0 — `3db608d`, `868fb6c`.)

## Staged, awaiting your commit: M4 — OUTPUT then PARAMETER override

Implemented and verified, **staged not committed**. This is the last core
milestone of Plan 1 (M5 is dashboard/demo/docs polish).

- **New `src/core/override_registry.py`** — the mechanism the plan flagged as
  needed but didn't fully specify: output/param overrides have **no HELICS
  representation** (the target already computes that value/parameter itself),
  so they can't reuse M3's HELICS-pub approach. `OverrideRegistry` is a thin
  Redis-backed key/value channel (reuses the existing `utils.redis_client.RedisClient`,
  same as config distribution): `set_override`/`get_override`/`clear_override`,
  keyed `cosim:override:<scope>:<sim_id>:<federation>:<federate>:<entity>:<var>`.
  `parse_target(helics_key, default_federation)` parses a bridge's target string
  into those four components — **important subtlety**: `entity` is
  reconstructed as `"<federate>.<instance>"` (e.g. `"spring_federate.0"`), not
  just the bare instance number, because that's what `BaseFederate` actually
  uses as an entity id everywhere (`entity['id']`, `pub['entity_name']`). Got
  this wrong on the first pass (parsed entity as `"0"`) — overrides silently
  never matched until fixed; see Blockers #9.
- **`src/utils/config_dataclasses.py`**:
  - `override_enabled: bool = False` added to `_FederateConfigBase` — the
    opt-in flag a **consuming** federate sets to allow interface-federate
    overrides. `False` (default) = zero registry lookups, i.e. zero behavior
    change, at every call site.
  - `BridgeSpec`'s passthrough validator relaxed: `source_key` is only required
    when `mode == "passthrough" AND scope == "input"` — output/param scopes
    have no "real source" to pass through (absence of an override already
    means "use the computed value"), so requiring it there made no sense.
- **`src/core/InterfaceFederate.py`**: `_register_connections()` now routes
  `scope: output`/`param` bridges into a new `self._override_bridges` list
  instead of skipping them (no HELICS registration for these — see above).
  `_publish_outputs()` calls a new `_publish_override_bridges()`: for each
  override bridge, if the adapter has a value, bounds-clip it and
  `registry.set_override(...)`; if not (never arrived, or after `finalize()`
  clears it), `registry.clear_override(...)` — this is what makes "disabling
  restores computed behavior" work. `finalize()` also explicitly clears every
  override bridge's registry entry on shutdown.
- **`src/core/BaseFederate.py`** (the consumer side, works for **any** federate
  with `override_enabled: true`, not just spring/base — RL federates get this
  too since it's on `_FederateConfigBase`):
  - `_publish_outputs()`: if `override_enabled`, checks
    `_get_output_override(entity_id, var_name)` before publishing; if present,
    substitutes it **and also writes it back into `self.outputs[entity_id][var_name]`**
    — see Blockers #10, this was a second bug I had to fix: without it, the
    override only affected what got HELICS-published, not what got recorded in
    storage, so verification looked like the override silently did nothing.
  - New `_apply_param_overrides()`, called each tick right before the
    model-step loop in `run()`: for every entity's every declared parameter
    name, checks the registry and calls `model.set_parameter(name, value)`
    if an override is active.
- **`src/models/base_model.py`** — new `BaseModel.set_parameter(name, value)`:
  bounds-clips against the catalog's `parameters[name].min_value/max_value`,
  writes into `self.state.parameters[name]` (the live dict physics models read
  from every step).
- **`src/scenarios/m4_interface_override_smoke_test.yaml`** — new fixture:
  `spring_federate` (`override_enabled: true`) + `input_federate` unchanged,
  plus `dt_bridge_override` with two bridges (no HELICS registration at all):
  `scope: output` on `spring_federate.0/velocity` (bounds `[-0.5, 0.5]`),
  `scope: param` on `spring_federate.0/damping` (bounds `[0.0, 10.0]`). All
  three federates realtime-paced — **necessary** here because, unlike M2/M3,
  there is no HELICS pub/sub dependency forcing the consumer to wait on the
  bridge (overrides are out-of-band via Redis), so without pacing the whole
  federation would just race to completion in under a second.

**Verified** (see "How to verify" below for exact commands): ran the 25-tick
paced scenario, published both an out-of-range output value (`2.0` → clipped
to `0.5`) and out-of-range param value (`50.0` → clipped to `10.0`) around
tick 10-11. Recorded timeseries:
- `damping`: `2.0` (computed) for ticks 1-10, `10.0` (clipped override) ticks
  11-25.
- `velocity`: natural growth (`0.19` → `1.15`) for ticks 1-10, pinned at `0.5`
  (clipped override) ticks 11-24, **and** tick 25 reverts to `0.328` (the real
  computed value) — this happened organically because the bridge's own
  `finalize()` cleared its overrides slightly before spring's very last
  publish, which is exactly the "disabling restores computed behavior" case,
  demonstrated without any extra scripting.
- `test_script.py`/`test_script_rl.py` unchanged vs main; `tests/test_rl_config.py`:
  76 passed, 1 skipped.

**Your action:** review the staged diff, commit (e.g.
`feat(digital-twin): M4 output/parameter override via Redis registry`), then
say continue — I'll tick M4 and move to M5 (dashboard/demo/docs — the last
milestone of Plan 1).

## Next step (after you commit M4)

**M5 — Live dashboard, BK4 demo, docs**:
- Streamlit live view subscribing to `cosim/#` (first *live* dashboard path —
  today's dashboard, `src/dashboard/streamlit_dashboard.py`, only reads
  post-run result files).
- Example scenario pair: identical YAML run (a) fully simulated vs (b) one
  federate swapped `type: model → type: interface` bridging to an external
  process — the actual BK4 "one-line config swap" demo. The M2/M3/M4 smoke
  scenarios already prove each mechanism works in isolation; this milestone's
  job is a clean, presentable **pair** of scenarios showing the swap itself.
- Docs: a design note in `docs/` plus an `interface_config`/`stream` reference
  section in `CLAUDE.md` (the project's own top-level `CLAUDE.md` doesn't yet
  mention `stream`, `interface_config`, `override_enabled`, or the Mosquitto
  service at all — this is the natural place to add that, mirroring how the
  RL config schema is documented there today).

First concrete action: look at `src/dashboard/dashboard_data.py` and
`streamlit_dashboard.py` to see how the existing (post-run) dashboard loads
data, then design the minimal live-subscribe addition (probably a new page or
tab, not a rewrite).

## Files touched across M0-M4

**New:** `src/adapters/__init__.py`, `src/adapters/base_adapter.py`,
`src/adapters/mqtt_adapter.py`, `src/core/InterfaceFederate.py`,
`src/core/override_registry.py` (staged), `src/mosquitto/mosquitto.conf`, five
smoke-test scenarios (`m0`...`m3` committed, `m4_interface_override_smoke_test.yaml`
staged).
**Modified (committed M0-M3):** `environment.yml`, `src/core/federate_launcher.py`,
`src/core/mappings.yaml`, `src/docker-compose.yaml` (mosquitto @ host 11883),
`catalog_loader.py`, `catalog.yaml` (mqtt_adapter entry).
**Modified (staged, M4):** `src/core/BaseFederate.py` (output override +
`_apply_param_overrides`), `src/core/InterfaceFederate.py` (override bridges),
`src/models/base_model.py` (`set_parameter`), `src/utils/config_dataclasses.py`
(`override_enabled`, relaxed passthrough validator), `tests/test_rl_config.py`.

## State of the tree

On `digitaltwin_interfaces`, 5 commits ahead of `main` (`3db608d`, `868fb6c`,
`b56fe1f`, `d9f0eba`, `0cd8d15`). M4's changes are `git add`ed but
**uncommitted** — waiting on you.

## Blockers / deviations from the plan

1. **Process change (from M1 onward):** agent stages, user commits.
2. **Commit-message/milestone mismatch:** `d9f0eba` says "M3", contains M2's
   diff. Trust the diff / handoff / Progress Tracker, not commit messages.
3. **Branch-first ordering (M0), port remap (mosquitto @ 11883), `model_configs`
   guard, drop-oldest outbound queue** — see earlier handoff revisions in git
   history (`git log -p -- docs/handoffs/digitaltwin_interfaces.md`) if needed;
   summarized here so this doc stays current, not a full history log.
4. **`BridgeSpec.helics_key` for scope `input`** = the bridge's own HELICS
   publication name. **For scope `output`/`param`** it's overloaded again —
   here it's an **override-registry target** (parsed by `parse_target`), not a
   HELICS name at all, since there's no HELICS registration for these scopes.
   Same field name, three different meanings across `StreamSpec`/`BridgeSpec`
   scopes — flagging clearly in case this needs unifying later, but each
   individual meaning is documented at its point of use.
5. **`OverrideRegistry` mechanism itself is a plan-filling addition**, not
   explicitly specified — the plan said "prefer existing Redis plumbing over
   new HELICS wiring" for params but didn't design the channel. Built as the
   smallest thing that satisfies that: a plain Redis JSON key per (scope, sim,
   federation, federate, entity, var), no queueing/history, last-write-wins,
   TTL 3600s as a safety net against orphaned keys from crashed runs.
6. **Two bugs found and fixed during verification** (both now covered by unit
   tests in `TestInterfaceOverrideConfig`):
   - `parse_target` originally extracted the entity as the bare instance
     number (`"0"`) instead of the full `"federate.instance"` id
     (`"spring_federate.0"`) that `BaseFederate` actually uses — overrides
     silently never matched. Fixed; regression-tested.
   - The output-override substitution in `BaseFederate._publish_outputs()`
     originally only affected the HELICS-published value, not
     `self.outputs[entity_id][var_name]` — so `update_storage()` (and thus any
     verification via the results JSON) never showed the override taking
     effect even though it correctly reached other HELICS federates. Fixed by
     also writing the clipped value back into `self.outputs`.
7. **`mode` is moot for `scope: output`/`param` bridges** — both `replace` and
   `passthrough` behave identically (no override present = computed value is
   used, which is what "passthrough" would mean anyway). Only `scope: input`
   bridges have a real HELICS fallback to distinguish the two modes.
8. **"Disabling restores computed behavior" is demonstrated, not separately
   engineered** — it happens because `_publish_override_bridges()` clears the
   registry entry whenever the adapter has no value (covers "never received a
   message" and, via `finalize()`, "bridge has shut down"), but a **live
   mid-run disable-then-resume** cycle (bridge keeps running, operator
   explicitly "un-overrides" one variable while others stay overridden) wasn't
   separately exercised — worth a dedicated check if that specific UX matters
   later.
9. **Redis logging noise (minor, not fixed):** `RedisClient.get_json` logs a
   WARNING every time a key is absent — which is the common case for every
   `override_enabled` federate's every param/output check on every tick with
   no active override. Didn't touch `redis_client.py` (shared utility, other
   legitimate uses want that warning); if this gets noisy in practice, consider
   a `log_missing: bool` param on `get_json` or a dedicated quiet path in
   `OverrideRegistry.get_override`.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # digitaltwin_interfaces; M4 files staged, not committed
git log --oneline -6                       # 0cd8d15, d9f0eba, b56fe1f, 868fb6c, 3db608d on top of 38948b3 (main)
git diff --staged --stat                   # M4's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface/override are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M4 check — output + param override, clipped, then reverts at shutdown:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m4_interface_override_smoke_test')" > /tmp/m4.log 2>&1 &
RUNPID=$!
sleep 12
mosquitto_pub -h localhost -p 11883 -t 'cosim/m4_smoke/override/velocity' -m '{"value": 2.0}'
mosquitto_pub -h localhost -p 11883 -t 'cosim/m4_smoke/override/damping' -m '{"value": 50.0}'
wait $RUNPID
python3 -c "
import json, glob
p = sorted(glob.glob('results/m4_interface_override_smoke_test/*/federation_1/spring_federate_test_storage.json'))[-1]
data = json.load(open(p))
print('velocity:', data['outputs']['spring_federate.0']['velocity'])
print('damping :', data['params']['spring_federate.0']['damping'])
"
# expect: damping 2.0 x10 then 10.0 (clipped from 50) for the rest; velocity
# natural growth x10 then pinned 0.5 (clipped from 2.0), reverting on the very
# last tick when the bridge finalizes.
# then: rm -rf results/m4_interface_override_smoke_test logs/m4_interface_override_smoke_test (gitignored, local hygiene only)

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 76 passed, 1 skipped
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch
> `digitaltwin_interfaces`. M0-M3 are committed; M4 (output/parameter override
> via a Redis-backed override registry) is implemented and verified but
> staged, not committed — review and commit it yourself first (see 'Staged,
> awaiting your commit' above), then tell the agent to tick the M4 box and
> continue to M5 (live dashboard, BK4 demo, docs — the final milestone of
> Plan 1) exactly as scoped in the plan and in this handoff's 'Next step'.
> Remember: the agent stages changes and never runs `git commit` — you do."
