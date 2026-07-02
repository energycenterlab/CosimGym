# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

CosimGym is a Python orchestration framework that bridges HELICS co-simulation with Gymnasium-based Reinforcement Learning. It lets you define multi-federate simulation scenarios declaratively in YAML, then run them—either as pure physics co-simulations or as RL training/testing environments.

## Setup & Common Commands

**Prerequisites:** Conda, Docker, Docker Compose **v2** (`docker compose` plugin — verify `docker compose version` reports `v2.x`; legacy `docker-compose` v1 rejects the Compose Spec file), Python 3.12

> On a shared server without sudo, install the Compose v2 plugin into your home only (no impact on other users): drop the `docker-compose` v2 binary in `~/.docker/cli-plugins/` and `chmod +x` it. See `docs/Installation_Setup.md`.

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

All simulation scripts must be run from the project root with the `cosim_gym` conda environment active. Redis must be running before starting any simulation.

## Architecture

### Execution Flow

1. **`ScenarioManager`** (`src/core/ScenarioManager.py`) reads a YAML scenario config, starts HELICS brokers as subprocesses, then spawns each federate as a separate Python process via `federate_launcher.py`. The full scenario config is serialized to Redis so each subprocess can retrieve it.

2. **`federate_launcher.py`** (`src/core/federate_launcher.py`) is the entry point for each federate subprocess. It reads config from Redis and instantiates either `BaseFederate` (type `"base"`) or `RLFederate` (type `"rl"`).

3. **`BaseFederate`** (`src/core/BaseFederate.py`) manages the HELICS pub/sub lifecycle, time stepping, storage, and reset logic. It instantiates model objects from the Model Catalog and drives the `_step()` loop.

4. **`RLFederate`** / `HelicsGymEnv` (`src/core/RL_Federate.py`) wraps `BaseFederate` as a Gymnasium `Env`, routing observations and actions through HELICS. RL agents are also loaded from the catalog.

### YAML Scenario Config → Dataclasses → Runtime

Scenario YAML files live in `src/scenarios/`. They are parsed by `src/utils/config_reader.py` into typed dataclasses defined in `src/utils/config_dataclasses.py`. Key dataclasses: `ScenarioConfig`, `FederationConfig`, `FederateConfig`, `FedTimingConfig`, `FedConnections`, `FedPublication`, `FedSubscription`.

For RL scenarios, `ScenarioManager._modify_config_for_online_training()` injects a synthetic `rl_federation` into the config at runtime, creating an `rl_agent` federate whose pub/sub wiring is derived from the `reinforcement_learning_config` block.

### Model Catalog

`src/models/model_catalog/catalog.yaml` is the static registry. Each entry maps a `model_name` key to `class_name` + `module_path` plus I/O spec (inputs, outputs, parameters with bounds). `catalog_loader.py` loads this into Redis at startup. At runtime, `RedisCatalog` (`src/models/model_catalog/RedisCatalog.py`) resolves model metadata for `BaseFederate` to dynamically import and instantiate the correct class.

Adding a new model: create a Python class that inherits `BaseModel` (implements `initialize`, `step`, `finalize`) and add an entry to `catalog.yaml`. Use `src/models/model_catalog/model_template.yaml` as the template.

Built-in physical models are in `src/models/model_catalog/physical_models/`. RL agent classes are in `src/models/model_catalog/RL_agents/`.

### Key Timing Concepts

- `real_period` (seconds): real-world time per federate step — the only required timing field in YAML.
- HELICS time is unitless integer ticks; `ScenarioManager` normalizes all federates to the minimum `real_period` as tick 1.
- `time_offset` in YAML shifts a federate's first tick to avoid same-step circular dependencies. `auto_offset` in the scenario's `synchronization` block can compute these automatically via topological sort.
- `subscription.causality: "same_step" | "next_step"` controls whether a subscription value is applied immediately or deferred one tick.

### Storage & Results

Federates buffer timeseries in memory in `self.storage` (partitioned by `train`/`test`). What happens to that data at the end (or during) the run is controlled by `memory_config.sink`:

- **`sink: json`** (default, unchanged behavior): nothing is written until the run ends, when `store_local_file()` dumps each partition to `results/<scenario_name>/<sim_id>/<federation_name>/<federate>_<mode>_storage.json`.
- **`sink: parquet`**: non-blocking, incremental. Each tick, `update_storage()` hands a row snapshot to a background `AsyncStorageWriter` (`src/utils/async_storage.py`) via a queue — the sim thread never blocks on I/O (the queue only blocks, rather than drops, if the writer thread falls behind, since result rows must never be silently lost). The writer batches rows (`memory_config.batch_size`) and hands each batch to a `ParquetStorageWriter` (`src/utils/parquet_storage.py`), which flattens them into a long/tidy schema (`time, federation, federate, model_instance, attribute, type, mode, value`) and writes them via `pyarrow.parquet.ParquetWriter` — one row group per batch, one file per mode — to the same `results/<scenario_name>/<sim_id>/<federation_name>/<federate>_<mode>_storage.parquet` layout the JSON sink uses. The Parquet file is finalized (`close()`) at the end of the run, before `store_local_file()` runs (which for `sink: parquet` is then a no-op — the data is already on disk). Measured negligible sim-thread cost (~3µs/tick) vs `sink: json` on a 3600-tick benchmark.
- **`sink: none`**: skips local file storage entirely (useful for throwaway runs).

`RLFederate` currently only supports `sink: json` / `sink: none` — `sink: parquet` raises `NotImplementedError` there (its storage schema differs from `BaseFederate`'s and isn't wired to the async writer yet).

The Streamlit dashboard's `load_all_records()` (`src/dashboard/dashboard_data.py`) currently reads **JSON results only**; Parquet result files are not yet dashboard-readable (the schema was deliberately designed to match the dashboard's existing columns to make that addition low-risk later, but it isn't implemented).

### Multi-Federation Scenarios

When a scenario has more than one federation, `ScenarioManager` automatically inserts a hierarchy broker (`helics_broker --sub_brokers=N`) above the per-federation brokers and assigns TCP ports dynamically.

### Digital-Twin Interfaces & Live Streaming (opt-in, off by default)

Two MQTT-backed mechanisms (Mosquitto broker, `src/adapters/mqtt_adapter.py`, background thread — never blocks the sim) externalize data while a run is in progress. Full reference: `docs/user_guide/digital_twin_interfaces.md`.

- **`streaming: { stream: true }`** on any `base`/`rl` federate (`StreamingConfig`) mirrors its inputs/outputs to MQTT each step (`<prefix>/<inputs|outputs>/<entity_id>/<var>`), alongside normal HELICS traffic. Purely for external observers/dashboards; changes nothing in the co-sim.
- **`type: interface`** federate (`InterfaceFederate(BaseFederate)`, `InterfaceFederateConfig`, no physics model) relays its wired connections to/from the external world via `interface_config`:
  - `adapter`: catalog-resolved transport (`mqtt_adapter`).
  - `streams`: HELICS subscription → MQTT publish (co-sim → external).
  - `bridges`: `scope: input` registers a normal HELICS global publication (`mode: replace` = external value only; `mode: passthrough` + `source_key` = real source until an external value arrives, then follows it). `scope: output`/`param` have no HELICS representation — the bridge instead writes bounds-clipped values into the Redis-backed `OverrideRegistry` (`src/core/override_registry.py`); a target federate opts in with `override_enabled: true` to substitute them in `_publish_outputs()`/`BaseModel.set_parameter()`. Clearing the external value restores computed behavior next step.
- **BK4 pattern (config-only sim-to-real):** since a model federate and an interface federate register identical HELICS key names, swapping simulated hardware for real is a change to *one* federate's block (`type: base` → `type: interface`) — every subscriber is untouched. Demo pair: `src/scenarios/m5_bk4_demo_a_full_sim.yaml` / `m5_bk4_demo_b_digital_twin.yaml`.
- **Live dashboard:** `./src/dashboard/run_live_dashboard.sh` (`src/dashboard/live_dashboard.py`) subscribes to `cosim/#` and shows both mechanisms' data as it's published — separate from the post-run `dashboard_app.py`.

## Config Reference

Scenario YAML top-level keys:
- `start_time`, `end_time`: ISO 8601 datetimes
- `log_level`: `ERROR | WARNING | INFO | DEBUG`
- `memory_config.attrs`: `"all"` or list of variable names to record
- `memory_config.sink`: `json` (default) | `parquet` | `none` — see Storage & Results above
- `memory_config.batch_size`: rows per batch for the `parquet` sink's background writer (default `100`)
- `synchronization`: auto-offset and startup-sync policies
- `reinforcement_learning_config`: 4-axis RL config (see below)
- `federations.<name>.broker_config`: `core_type`, `port`, `federates`
- `federations.<name>.federate_configs.<name>`: `type` (`base`|`rl`), `timing_configs.real_period`, `connections.publishes`, `connections.subscribes`, `model_configs.instantiation.model_name`

Subscription target format: `<federate_name>.<instance_id>/<pub_key>` (same federation) or `<federation_name>.<federate_name>.<instance_id>/<pub_key>` (cross-federation).

RL observation/action keys use dot notation: `<federation>.<federate>.<instance>.<variable>`.

### RL Config Schema (`reinforcement_learning_config`)

Four top-level axes under `reinforcement_learning_config`:

- **`environment`** (MDP): `observations` (mapping: key → `ObservationSpec` with causality/history/reset_default/role/bounds), `actions` (mapping: key → `ActionSpec` with space/bounds/bins), `reward` (dotted path to reward fn), `reset` (mode: full|rolling|none, force_defaults)
- **`agent`** (solver): `model_name` (catalog key), `backend` (stable_baselines3|rllib), `algorithm`, `policy`, `hyperparameters` (all-Optional: learning_rate/gamma/batch_size/net_arch/train_frequency/gradient_steps), `params` (backend-specific escape hatch)
- **`run`** (schedule): `mode` (online|offline), `train` (episodes/episode_length → `total_steps` property), `eval`, `test` (episodes/episode_length/deterministic/checkpoint)
- **`experiment`** (infra): `name`, `checkpoint` (dir/best → `best_path` property), `logging`, `offline`

All RL Pydantic models use `extra='forbid'` — typos in YAML raise validation errors. Hyperparameters default to `None` (omit → backend applies own defaults).

Available RL agents: `rl_simple_SACsb3` (SB3 SAC), `rl_simple_DQN` (custom PyTorch DQN), `rl_simple_rllib` (RLlib PPO standalone module). Add new agents by subclassing `RLAgent` and adding a catalog entry.

Tests: `pytest tests/test_rl_config.py` (parse-gate + extra='forbid' + validators).
