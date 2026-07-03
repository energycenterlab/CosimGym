# CLAUDE.md

This file guide Claude Code (claude.ai/code) when working in this repo.

## What This Project Is

CosimGym = Python orchestration framework. Bridges HELICS co-simulation with Gymnasium RL. Define multi-federate scenarios declaratively in YAML, run as pure physics co-sim or RL training/testing env.

## Setup & Common Commands

**Prerequisites:** Conda, Docker, Docker Compose **v2** (`docker compose` plugin — verify `docker compose version` reports `v2.x`; legacy `docker-compose` v1 rejects Compose Spec file), Python 3.12

> Shared server, no sudo: install Compose v2 plugin into home only (no impact on other users) — drop `docker-compose` v2 binary in `~/.docker/cli-plugins/`, `chmod +x` it. See `docs/Installation_Setup.md`.

```bash
# Full setup
docker compose -f src/docker-compose.yaml up -d
conda activate cosim_gym
# Run simulations (activate env first: conda activate cosim_gym)
python src/test_script.py           # runs base co-simulation scenarios
python src/test_script_rl.py        # runs RL training scenarios (sets OMP_NUM_THREADS=1 etc.)
# Docker management
docker compose -f src/docker-compose.yaml logs -f redis
```

All sim scripts: run from project root, `cosim_gym` conda env active. Redis must run before any simulation.

## Architecture

### Execution Flow

1. **`ScenarioManager`** (`src/core/ScenarioManager.py`) reads YAML scenario config, starts HELICS brokers as subprocesses, spawns each federate as separate Python process via `federate_launcher.py`. Full scenario config serialized to Redis so each subprocess retrieves it.

2. **`federate_launcher.py`** (`src/core/federate_launcher.py`) = entry point per federate subprocess. Reads config from Redis, instantiates `BaseFederate` (type `"base"`) or `RLFederate` (type `"rl"`).

3. **`BaseFederate`** (`src/core/BaseFederate.py`) manages HELICS pub/sub lifecycle, time stepping, storage, reset. Instantiates model objects from Model Catalog, drives `_step()` loop.

4. **`RLFederate`** / `HelicsGymEnv` (`src/core/RL_Federate.py`) wraps `BaseFederate` as Gymnasium `Env`, routes observations/actions through HELICS. RL agents also loaded from catalog.

### YAML Scenario Config → Dataclasses → Runtime

Scenario YAML in `src/scenarios/`. Parsed by `src/utils/config_reader.py` into typed dataclasses in `src/utils/config_dataclasses.py`. Key dataclasses: `ScenarioConfig`, `FederationConfig`, `FederateConfig`, `FedTimingConfig`, `FedConnections`, `FedPublication`, `FedSubscription`.

RL scenarios: `ScenarioManager._modify_config_for_online_training()` injects synthetic `rl_federation` at runtime — creates `rl_agent` federate, pub/sub wiring derived from `reinforcement_learning_config` block.

### Model Catalog

`src/models/model_catalog/catalog.yaml` = static registry. Each entry: `model_name` key → `class_name` + `module_path` + I/O spec (inputs, outputs, parameters with bounds). `catalog_loader.py` loads into Redis at startup. Runtime: `RedisCatalog` (`src/models/model_catalog/RedisCatalog.py`) resolves model metadata so `BaseFederate` dynamically imports + instantiates correct class.

New model: create Python class inheriting `BaseModel` (implements `initialize`, `step`, `finalize`), add entry to `catalog.yaml`. Template: `src/models/model_catalog/model_template.yaml`.

Built-in physical models: `src/models/model_catalog/physical_models/`. RL agent classes: `src/models/model_catalog/RL_agents/`.

### Key Timing Concepts

- `real_period` (seconds): real-world time per federate step — only required timing field in YAML.
- HELICS time = unitless integer ticks; `ScenarioManager` normalizes all federates to minimum `real_period` as tick 1.
- `time_offset` in YAML shifts federate's first tick — avoids same-step circular deps. `auto_offset` in scenario's `synchronization` block computes automatically via topological sort.
- `subscription.causality: "same_step" | "next_step"` controls subscription value applied immediately or deferred one tick.

### Storage & Results

Federates buffer timeseries in memory in `self.storage` (partitioned `train`/`test`). End-of-run (or during-run) fate controlled by `memory_config.sink`:

- **`sink: json`** (default, unchanged behavior): nothing written until run ends, then `store_local_file()` dumps each partition to `results/<scenario_name>/<sim_id>/<federation_name>/<federate>_<mode>_storage.json`.
- **`sink: parquet`**: non-blocking, incremental. Each tick, `update_storage()` hands row snapshot to background `AsyncStorageWriter` (`src/utils/async_storage.py`) via queue — sim thread never blocks on I/O (queue blocks, not drops, if writer thread falls behind — result rows must never silently vanish). Writer batches rows (`memory_config.batch_size`), hands each batch to `ParquetStorageWriter` (`src/utils/parquet_storage.py`) — flattens to long/tidy schema (`time, federation, federate, model_instance, attribute, type, mode, value`), writes via `pyarrow.parquet.ParquetWriter` — one row group per batch, one file per mode — same `results/<scenario_name>/<sim_id>/<federation_name>/<federate>_<mode>_storage.parquet` layout as JSON sink. Parquet file finalized (`close()`) at run end, before `store_local_file()` (no-op for `sink: parquet` — data already on disk). Measured sim-thread cost negligible (~3µs/tick) vs `sink: json` on 3600-tick benchmark.
- **`sink: none`**: skips local file storage (good for throwaway runs).

`RLFederate` supports only `sink: json` / `sink: none` — `sink: parquet` raises `NotImplementedError` (storage schema differs from `BaseFederate`'s, not wired to async writer yet).

Streamlit dashboard `load_all_records()` (`src/dashboard/dashboard_data.py`) reads **JSON results only**. Parquet results not dashboard-readable yet (schema deliberately matches dashboard's existing columns — future addition low-risk, but not implemented).

### Multi-Federation Scenarios

Scenario with >1 federation: `ScenarioManager` auto-inserts hierarchy broker (`helics_broker --sub_brokers=N`) above per-federation brokers, assigns TCP ports dynamically.

### Digital-Twin Interfaces & Live Streaming (opt-in, off by default)

Two MQTT-backed mechanisms (Mosquitto broker, `src/adapters/mqtt_adapter.py`, background thread — never blocks sim) externalize data mid-run. Full reference: `docs/user_guide/digital_twin_interfaces.md`.

- **`streaming: { stream: true }`** on any `base`/`rl` federate (`StreamingConfig`) mirrors inputs/outputs to MQTT each step (`<prefix>/<inputs|outputs>/<entity_id>/<var>`), alongside normal HELICS traffic. Only for external observers/dashboards; changes nothing in co-sim.
- **`type: interface`** federate (`InterfaceFederate(BaseFederate)`, `InterfaceFederateConfig`, no physics model) relays wired connections to/from external world via `interface_config`:
  - `adapter`: catalog-resolved transport (`mqtt_adapter`).
  - `streams`: HELICS subscription → MQTT publish (co-sim → external).
  - `bridges`: `scope: input` registers normal HELICS global publication (`mode: replace` = external value only; `mode: passthrough` + `source_key` = real source until external value arrives, then follows it). `scope: output`/`param` have no HELICS representation — bridge writes bounds-clipped values into Redis-backed `OverrideRegistry` (`src/core/override_registry.py`); target federate opts in with `override_enabled: true` to substitute in `_publish_outputs()`/`BaseModel.set_parameter()`. Clear external value → computed behavior restored next step.
- **BK4 pattern (config-only sim-to-real):** model federate and interface federate register identical HELICS key names → swap simulated hardware for real = change *one* federate's block (`type: base` → `type: interface`) — every subscriber untouched. Demo pair: `src/scenarios/m5_bk4_demo_a_full_sim.yaml` / `m5_bk4_demo_b_digital_twin.yaml`.
- **Live dashboard:** "Live" page of `./src/dashboard/run_dashboard.sh` (`src/dashboard/live_dashboard.py`, combined into single Streamlit app via `st.navigation`) subscribes to `cosim/#`, shows both mechanisms' data as published — separate from "Results" page's post-run `dashboard_app.py` view.

## Config Reference

Scenario YAML top-level keys:
- `start_time`, `end_time`: ISO 8601 datetimes
- `log_level`: `ERROR | WARNING | INFO | DEBUG`
- `memory_config.attrs`: `"all"` or list of variable names to record
- `memory_config.sink`: `json` (default) | `parquet` | `none` — see Storage & Results above
- `memory_config.batch_size`: rows per batch for `parquet` sink's background writer (default `100`)
- `synchronization`: auto-offset + startup-sync policies
- `reinforcement_learning_config`: 4-axis RL config (below)
- `federations.<name>.broker_config`: `core_type`, `port`, `federates`
- `federations.<name>.federate_configs.<name>`: `type` (`base`|`rl`), `timing_configs.real_period`, `connections.publishes`, `connections.subscribes`, `model_configs.instantiation.model_name`
- `model_configs.instantiation.parallel_execution` (default `false`) + `max_parallel_workers` (default `min(n_instances, cpu_count)`): step federate's model instances in **persistent worker processes** (`src/core/parallel_executor.py`) instead of default sequential loop. CPU-heavy model `step()`s only (pure-Python/GIL-bound → processes, not threads; workers rebuild shard from config). Workers daemon + escalating `close()` (sentinel→join→terminate→kill) + atexit/SIGINT/SIGTERM → no orphans. Unsupported with `override_enabled` or `type: rl` (raises `NotImplementedError`). Benchmark pair: `src/scenarios/benchmark_parallel_{seq,par}.yaml` with CPU-heavy `heavy_compute_dummy` model. See `docs/user_guide/scenario_configuration/federate.md`.

Subscription target format: `<federate_name>.<instance_id>/<pub_key>` (same federation) or `<federation_name>.<federate_name>.<instance_id>/<pub_key>` (cross-federation).

RL observation/action keys use dot notation: `<federation>.<federate>.<instance>.<variable>`.

### RL Config Schema (`reinforcement_learning_config`)

Four top-level axes under `reinforcement_learning_config`:

- **`environment`** (MDP): `observations` (mapping: key → `ObservationSpec` with causality/history/reset_default/role/bounds), `actions` (mapping: key → `ActionSpec` with space/bounds/bins), `reward` (dotted path to reward fn), `reset` (mode: full|rolling|none, force_defaults)
- **`agent`** (solver): `model_name` (catalog key), `backend` (stable_baselines3|rllib), `algorithm`, `policy`, `hyperparameters` (all-Optional: learning_rate/gamma/batch_size/net_arch/train_frequency/gradient_steps), `params` (backend-specific escape hatch)
- **`run`** (schedule): `mode` (online|offline), `train` (episodes/episode_length → `total_steps` property), `eval`, `test` (episodes/episode_length/deterministic/checkpoint)
- **`experiment`** (infra): `name`, `checkpoint` (dir/best → `best_path` property), `logging`, `offline`

All RL Pydantic models use `extra='forbid'` — YAML typos raise validation errors. Hyperparameters default `None` (omit → backend applies own defaults).

Available RL agents: `rl_simple_SACsb3` (SB3 SAC), `rl_simple_DQN` (custom PyTorch DQN), `rl_simple_rllib` (RLlib PPO standalone module). New agents: subclass `RLAgent`, add catalog entry.

Tests: `pytest tests/test_rl_config.py` (parse-gate + extra='forbid' + validators).

## graphify

Project has knowledge graph at graphify-out/ — god nodes, community structure, cross-file relationships.

Rules:
- Codebase questions: first run `graphify query "<question>"` when graphify-out/graph.json exists. `graphify path "<A>" "<B>"` for relationships, `graphify explain "<concept>"` for focused concepts. Return scoped subgraph — much smaller than GRAPH_REPORT.md or raw grep.
- If graphify-out/wiki/index.md exists, use for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain surface too little.
- After code changes, run `graphify update .` — keeps graph current (AST-only, no API cost).