# Getting Started

This guide walks you through setting up and running your first simulation with CosimGym.

## 1. Prerequisites

Before starting, ensure you have completed the installation steps listed in the [Installation Setup](../Installation_Setup.md) section. Our recommended setup uses the unified Makefile which relies on **Docker** (for backend services like Redis and InfluxDB) and **Conda**.

## 2. Basic Setup Validation

Once installed, ensure your backend infrastructure is running. The repository provides an environment wrapper handling everything.

Navigate to the repository root:
```bash
make setup
make validate
```
*(If you are choosing the Python setup script route, use `python setup.py --auto` and `python setup.py --validate`)*.

## 3. Review a Basic Scenario

A scenario defines the models, timeline, and connections. In the `src/scenarios` directory, open `simple_test.yaml`. 
This file defines a single `spring_mass_damper` physics model and an `inputs4spring` signal model publishing into it.

```yaml
name: "case0: simple_test"
start_time: "2024-01-01 00:00:00"
end_time: "2024-01-02 00:00:00"

federations:
  - name: "federation1"
    # ... broker and federate definitions
    # Detailed in "Scenario Configuration"
```

## 4. Run the Scenario

All scenarios in CosimGym are launched via the main entry point: `src/test_script.py`. Let's execute the `case0` (simple test).

Make sure your conda environment is activated:
```bash
conda activate cosim_gym
python src/test_script.py --scenario simple_test
```
You will see logs streaming showing the Scenario Manager spawning brokers and advancing time. The output artifacts are saved inside the `logs/` and `results/` directories.

## 5. Visualize Results in Dashboard

CosimGym features a built-in interactive Streamlit dashboard. It connects directly to the resulting output files (Parquet + InfluxDB).

Start the dashboard:
```bash
make run-dashboard
```
Your default browser will launch pointing to `localhost:8501`. Here you can select the run from the dropdown menu and inspect all published tags graphically!
