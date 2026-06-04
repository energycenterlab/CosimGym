# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

CosimGym is a Python orchestration framework that bridges HELICS co-simulation with Gymnasium-based Reinforcement Learning. It lets you define multi-federate simulation scenarios declaratively in YAML, then run them—either as pure physics co-simulations or as RL training/testing environments.

## Setup & Common Commands

**Prerequisites:** Conda, Docker, Python 3.12

```bash
# Full setup
make setup                          # creates cosim_gym conda env + starts Redis via Docker

# Or step by step
make setup-env                      # conda env from environment.yml
make setup-docker                   # starts Redis container (src/docker-compose.yaml)
make validate                       # checks all components are up

# Run simulations (activate env first: conda activate cosim_gym)
python src/test_script.py           # runs base co-simulation scenarios
python src/test_script_rl.py        # runs RL training scenarios (sets OMP_NUM_THREADS=1 etc.)

# Dashboard
make run-dashboard                  # Streamlit at http://localhost:8501
# or
streamlit run src/dashboard/streamlit_dashboard.py

# Docker management
make clean                          # stop containers
make teardown                       # full cleanup (env + containers)
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

Federates buffer timeseries in memory in `self.storage` (partitioned by `train`/`test`). At simulation end, `store_local_file()` writes JSON to `results/<scenario_name>/<sim_id>/<federation_name>/`. The Streamlit dashboard reads these files (via `src/dashboard/dashboard_data.py` and parquet caching in `src/dashboard/dashboard_parquet_cache.py`).

### Multi-Federation Scenarios

When a scenario has more than one federation, `ScenarioManager` automatically inserts a hierarchy broker (`helics_broker --sub_brokers=N`) above the per-federation brokers and assigns TCP ports dynamically.

## Config Reference

Scenario YAML top-level keys:
- `start_time`, `end_time`: ISO 8601 datetimes
- `log_level`: `ERROR | WARNING | INFO | DEBUG`
- `memory_config.attrs`: `"all"` or list of variable names to record
- `synchronization`: auto-offset and startup-sync policies
- `reinforcement_learning_config`: RL agent, env (observations/actions), training, test blocks
- `federations.<name>.broker_config`: `core_type`, `port`, `federates`
- `federations.<name>.federate_configs.<name>`: `type` (`base`|`rl`), `timing_configs.real_period`, `connections.publishes`, `connections.subscribes`, `model_configs.instantiation.model_name`

Subscription target format: `<federate_name>.<instance_id>/<pub_key>` (same federation) or `<federation_name>.<federate_name>.<instance_id>/<pub_key>` (cross-federation).

RL observation/action keys use dot notation: `<federation>.<federate>.<instance>.<variable>`.
