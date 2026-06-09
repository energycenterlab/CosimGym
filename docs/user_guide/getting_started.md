# Getting Started

This guide walks you through setting up and running your first simulation with CosimGym.

## 1. Prerequisites

Before starting, ensure you have completed the installation steps listed in the [Installation Setup](../Installation_Setup.md) section. The recommended setup uses **Conda** (Python 3.12 environment) plus **Docker**, which provides the backend services: **Redis** (mandatory — distributes the scenario config and model catalog to federate processes) and **MinIO** (optional — object store for remotely-hosted FMUs).

## 2. Basic Setup Validation

Once installed, ensure your backend infrastructure is running. The repository provides an environment wrapper handling everything.

Navigate to the repository root:
```bash
make setup
make validate
```
*(If you are choosing the Python setup script route, use `python setup.py --auto` and `python setup.py --validate`)*.

## 3. Review a Basic Scenario

A scenario defines the models, timeline, and connections. In the `src/scenarios/` directory, open `simple_test.yaml`.
This file defines two federates inside one federation: a `spring_federate` running the `spring_mass_damper` physics model (2 instances) and an `input_federate` running the `inputs4spring` signal model that publishes the driving `force` and `disturbance` signals.

```yaml
name: "simple_test"
start_time: "2024-01-01T00:00:00"
end_time:   "2024-01-01T01:00:00"

federations:
  federation_1:                 # dict keyed by federation name (not a list)
    broker_config:
      core_type: "zmq"
      port: 23404
    federate_configs:
      spring_federate:          # ... broker and federate definitions
      input_federate:           # detailed in "Scenario Configuration"
```

> Full field-by-field reference is in [Scenario Configuration](scenario_configuration/overview.md).

## 4. Run the Scenario

All scenarios are launched via the main entry point `src/test_script.py`. There is **no command-line flag** — you select the scenario by editing the file and calling `main('<scenario_name>')`:

```python
# src/test_script.py
from core.ScenarioManager import main

main('simple_test')      # scenario filename without the .yaml extension
```

Then, with the conda environment active, run it from the repository root:

```bash
conda activate cosim_gym
python src/test_script.py
```

Or use the Makefile wrapper, which validates the setup first:

```bash
make run
```

You will see logs streaming as the Scenario Manager spawns brokers and advances time. Output artifacts are saved under `logs/` and `results/<scenario_name>/`.

## 5. Visualize Results in Dashboard

CosimGym features a built-in interactive Streamlit dashboard. It reads the JSON result files written under `results/` (and builds a local Parquet cache for fast re-loading).

Start the dashboard:
```bash
make run-dashboard
```
Your default browser will launch pointing to `localhost:8501`. Here you can select the run from the dropdown menu and inspect all published tags graphically!
