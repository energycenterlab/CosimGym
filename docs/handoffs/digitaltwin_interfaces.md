# Handoff — `digitaltwin_interfaces` (Plan 1)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `digitaltwin_interfaces` (created off `main`, `git checkout main && git checkout -b digitaltwin_interfaces`)

## Last completed milestone

**M0 — Shared scaffolding** ✅ ticked in the plan file. Commit: `3db608d`
(preceded by cleanup commit `626e03b` — archived completed RL-refactor/pandapower
plan+handoff artifacts, done on this branch not `main`, since the classifier
blocked deleting on `main` directly).

## Next step

Start **M1 — `stream` flag outbound mirror** (all federate types):
- Implement `MqttAdapter.publish()` real outbound path: bounded `queue.Queue`
  with drop-oldest, drained by a loop (paho's own thread, or a small drain
  loop started in `connect()`) that actually calls `client.publish(topic, json)`.
- Add the `BaseFederate` stream hook: when `self.config.streaming.stream` is
  `True`, lazily create/connect the shared `MqttAdapter` and, right after
  `update_storage()` (`BaseFederate.py:442`, now ~443 after the M0 guard edit),
  enqueue current inputs+outputs for that federate.
- **Check:** run any scenario with `flags.stream: true`... — wait, it's
  `streaming.stream: true` (see `StreamingConfig` in `config_dataclasses.py`),
  not under `flags`. Use `mosquitto_sub -t 'cosim/#' -p 11883` (see port note
  below) and confirm live values appear. Confirm co-sim results are byte-identical
  to `main` when `streaming.stream` is absent/false.

First concrete action: read `src/core/BaseFederate.py` around line 442
(`update_storage()` call site in `run()`) and add the stream-hook call there,
gated on `self.config.streaming.stream`.

## Files touched so far

**New:**
- `src/adapters/__init__.py`, `src/adapters/base_adapter.py` (`InterfaceAdapter` ABC),
  `src/adapters/mqtt_adapter.py` (`MqttAdapter` — connect/close only; publish/subscribe/latest
  raise `NotImplementedError` until M1/M3)
- `src/core/InterfaceFederate.py` (shell: `_register_entities` → `[]`,
  `update_storage` → no-op)
- `src/mosquitto/mosquitto.conf` (anonymous listener on 1883 inside the container)
- `src/scenarios/m0_interface_smoke_test.yaml` (M0's boot-check scenario — keep it,
  the config parse-gate test picks it up automatically)

**Modified:**
- `environment.yml` — added `paho-mqtt>=2.0.0`
- `src/core/BaseFederate.py` — one-line guard at the model-step branch in `run()`:
  `if self.config.model_configs and self.config.model_configs.instantiation.parallel_execution:`
  (was crashing for federates with no `model_configs`, i.e. interface federates)
- `src/core/federate_launcher.py` — imports `InterfaceFederateConfig`; the non-RL
  branch now picks `InterfaceFederateConfig` vs `BaseFederateConfig` based on `args.type`
- `src/core/mappings.yaml` — `interface: InterfaceFederate:InterfaceFederate`
- `src/docker-compose.yaml` — new `mosquitto` service, **host port 11883** (not 1883 —
  a system-wide mosquitto service already owns 1883 on this shared server; same
  pattern as the existing MinIO 9101 remap)
- `src/models/model_catalog/catalog_loader.py` — `CATEGORY_MAP["interface_adapter"] = "interface_adapters"`
- `src/models/model_catalog/catalog.yaml` — new `mqtt_adapter` entry (category
  `interface_adapter`; default `port: 11883` to match the compose remap)
- `src/utils/config_dataclasses.py` — `StreamingConfig` (on `_FederateConfigBase`,
  `extra='ignore'`), `AdapterConfig`/`StreamSpec`/`BridgeSpec`/`InterfaceConfig`
  (`extra='forbid'`), `InterfaceFederateConfig(_FederateConfigBase)` with
  `type: Literal["interface"]`, `model_configs: Optional[ModelConfig] = None`,
  `memory_config: MemoryConfig = Field(default_factory=MemoryConfig)` — added to the
  `FederateConfig` discriminated union
- `tests/test_rl_config.py` — new `TestStreamingAndInterfaceConfig` class (6 tests)

## State of the tree

Clean, all committed on `digitaltwin_interfaces` (2 commits ahead of `main`).
`git status` → nothing to commit. Regression green (see below).

## Blockers / deviations from the plan

1. **Branch-first ordering.** The plan's "Session 0 → Step 1" does the
   HANDOFF.md/plan-file cleanup **on `main`** before creating the branch. The
   auto-mode classifier blocked `git rm` directly on `main` (irreversible
   deletion without the user naming the files). Resolved by creating the
   branch *first*, then doing the cleanup commit there instead — this also
   better matches the user's explicit meta-instruction ("ensure plans start
   with creation of a new branch to avoid disruption of existing branches").
   `main` itself was never touched.
2. **Port conflicts on this shared server** (same class of issue as the
   existing MinIO 9001→9101 remap): both host `1883` (system mosquitto
   service, `systemctl status mosquitto` — unrelated, pre-existing) and
   presumably other well-known ports may be occupied by other users/services.
   Mosquitto is remapped to host **`11883`**. If you add more infra in later
   milestones, check `ss -tlnp` / `docker ps` first.
3. **`ScenarioManager._enrich_dynamic_catalog_metadata`** (ScenarioManager.py:739)
   reads `.model_configs` on every federate config generically — required
   adding `model_configs: Optional[ModelConfig] = None` to
   `InterfaceFederateConfig` (mirroring what `RLFederateConfig` already does).
   Not a plan deviation, just a detail the plan didn't spell out.
4. Did **not** implement the `_register_connections`/`_register_pubs`/`_register_subs`
   override on `InterfaceFederate` in M0 — with `interface_config: None` and
   `entities = []`, the inherited methods already produce empty pubs/subs
   correctly. That override is real M2 work (building HELICS pubs/subs from
   `interface_config.streams`/`.bridges`) — deferred as the plan intends.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # should be digitaltwin_interfaces, clean
git log --oneline -5                       # 3db608d, 626e03b on top of 38948b3 (main)

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio, mosquitto (host 11883)

# Regression — must match main behavior exactly (streaming/interface are opt-in):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# M0's own check — stream:false federate + empty-config interface federate boot cleanly:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('m0_interface_smoke_test')"
# then: rm -rf results/m0_interface_smoke_test logs/m0_interface_smoke_test (gitignored, just local hygiene)

# Config parse-gate tests:
python -m pytest tests/test_rl_config.py -v   # 61 passed, 1 skipped (known-broken Adelaide_test.yaml, pre-existing)

# Adapter connects to Mosquitto:
python -c "
import sys; sys.path.insert(0, 'src')
from adapters.mqtt_adapter import MqttAdapter
a = MqttAdapter(host='localhost', port=11883)
a.connect(); print('connected:', a._connected.is_set()); a.close()
"
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> and `docs/handoffs/digitaltwin_interfaces.md`. We're on branch `digitaltwin_interfaces`,
> M0 is done and committed. Follow the Execution Guide: verify the regression is still
> green, then implement M1 (`stream` flag outbound mirror) exactly as scoped in the
> plan and in the handoff's 'Next step' section — run M1's Check + the regression
> test, commit, tick the M1 box in the plan file, update this handoff doc, then stop."
