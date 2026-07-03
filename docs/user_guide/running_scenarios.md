# Running a Scenario

CosimGym has **no `--scenario` command-line flag**. You pick which scenario runs by editing the entry-point script and calling `main('<scenario_name>')`, where `<scenario_name>` is the YAML filename in `src/scenarios/` without the `.yaml` extension.

## Two entry points

| Script | Use for | Notes |
|---|---|---|
| `src/test_script.py` | Plain co-simulation (no RL) | |
| `src/test_script_rl.py` | RL training / testing | Also sets `OMP_NUM_THREADS=1` and similar, to keep PyTorch single-threaded per process |

## Steps

1. **Start the backend** (Redis is mandatory; it distributes the config and model catalog to each federate process):

   ```bash
   docker compose -f src/docker-compose.yaml up -d
   ```

2. **Load the model catalog into Redis** (after any edit to `catalog.yaml`):

   ```bash
   python src/models/model_catalog/catalog_loader.py
   ```

3. **Select the scenario** by editing the entry point:

   ```python
   # src/test_script.py
   from core.ScenarioManager import main

   main('simple_test')        # ← filename without .yaml
   ```

4. **Run it** from the repository root, with the conda environment active:

   ```bash
   conda activate cosim_gym
   python src/test_script.py
   ```

## Makefile shortcut

`make run` validates the setup, then runs whichever scenario is currently uncommented in `test_script.py`:

```bash
make run
make run-dashboard      # launch the Streamlit dashboard afterwards
```

## Where output goes

- **Logs:** `logs/`
- **Results:** `results/<scenario_name>/<sim_id>/<federation_name>/`, format controlled by `memory_config.sink` — JSON by default (`<federate>_<train|test>_storage.json`) or Parquet (`<federate>_<train|test>_storage.parquet`, written incrementally by a non-blocking background writer). Both are read by the dashboard. See [`memory_config`](scenario_configuration/general.md#memory_config).

> The `main()` function takes a single argument (`scenario_name`). It does **not** accept an `enable_progress_bar` keyword — that option was removed.

## Digital-twin / streaming scenarios

Scenarios using `streaming.stream: true` or a `type: interface` federate additionally need Mosquitto running (already included in `docker compose -f src/docker-compose.yaml up -d`). See [Digital-Twin Interfaces & Live Streaming](digital_twin_interfaces.md) for the config reference, and `src/scenarios/m5_bk4_demo_a_full_sim.yaml` / `m5_bk4_demo_b_digital_twin.yaml` for a runnable example pair.
