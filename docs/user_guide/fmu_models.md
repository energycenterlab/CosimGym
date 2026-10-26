# FMU Models

CosimGym runs **Functional Mock-up Units (FMUs)** as ordinary models through a single wrapper class, `BaseFMUModel` (`src/models/base_FMU_model.py`). It supports **FMI 2.0 and 3.0** co-simulation (with partial FMI 1.0 support) via [fmpy](https://github.com/CATIA-Systems/FMPy), and reads the FMU's input/output variable names directly from the FMU's `modelDescription.xml`.

You do not write any Python to use an FMU — you register a catalog entry pointing `class_name` at `BaseFMUModel` and declare where the FMU file lives.

## 1. Catalog entry

```yaml
models:
  my_building_fmu:
    class_name: BaseFMUModel
    module_path: models.base_FMU_model
    version: 1.0.0
    description: My building FMU.
    category: physical_model
    time_step: 600          # must match the FMU's fixed communication step (see below)
    max_time_step: 600
    min_time_step: 600
    user_defined:
      fmu_source:
        type: local         # local | minio | http
        path: /abs/path/to/MyBuilding.fmu
    inputs:                 # names MUST match the FMU's input variables
      ZoneSetPoint:
        type: float
        default_value: 20.0
        unit: degC
    outputs:                # names MUST match the FMU's output variables
      TBuilding:
        type: float
        default_value: 0.0
        unit: degC
```

> The `inputs`/`outputs` keys are not free-form: they must equal the variable names declared inside the FMU. `BaseFMUModel` looks each scenario input/output up by name and sets/gets it on the FMU via its value reference.

### FMU sources

`fmu_source.type` selects how the binary is resolved. Downloaded FMUs are cached under `~/.cosimgym/fmu_cache/<model_name>/<version>/` and reused.

```yaml
# Local file
fmu_source: {type: local, path: /abs/path/to/Model.fmu}

# MinIO / S3 (the docker-compose stack ships a MinIO service)
fmu_source:
  type: minio
  endpoint: http://localhost:9000
  bucket: fmus
  object_key: building/MyBuilding.fmu
  # access_key/secret_key optional — fall back to MINIO_ACCESS_KEY/MINIO_SECRET_KEY env vars

# HTTP download
fmu_source: {type: http, url: https://example.com/MyBuilding.fmu}
```

## 2. Communication step must match the FMU

Many co-sim FMUs declare `canHandleVariableCommunicationStepSize="false"`, meaning the communication step is **fixed**. You must set the federate's `real_period` (and the catalog `time_step`) to that fixed step. For an EnergyPlus FMU with `Timestep = 6` per hour, the step is `3600 / 6 = 600` seconds.

```yaml
timing_configs:
  real_period: 600     # match the FMU's required communication step
```

## 3. EnergyPlus FMUs: defined stop time

FMUs generated from EnergyPlus IDF files (`idf-to-fmu-export-prep`) require a **defined stop time**. `BaseFMUModel` derives it automatically from the scenario `start_time`/`end_time` and passes it to `setupExperiment` (FMI 2.0) / `enterInitializationMode` (FMI 3.0). If the stop time is left undefined, EnergyPlus clamps it to 0 and the second `doStep` fails with `fmi2Error`. The stop time is the span until the next restart (end of the scenario, or the next RL reset), rounded **up to whole days** and never past the end of the RunPeriod.

An EnergyPlus FMU's calendar lives in the IDF inside the FMU. The FMU clock counts seconds from 1 January of the scenario's start year; `BaseFMUModel` starts the slave at the model's current position on its horizon window (`horizon_pos_s`, see [§4](#4-simulation-horizon-and-restart)). A scenario whose `start_time` differs from the FMU's RunPeriod begin date has the building simulating a different calendar date than the scenario says; every model is still at the same *elapsed* simulated time.

EnergyPlus FMUs also create runtime working directories named `Output_EPExport_<federate>.<n>/`. These are git-ignored — do not commit them.

## 4. Simulation horizon and restart

`BaseFMUModel` wraps **any** FMU, whatever engine exported it (EnergyPlus, Modelica tools, …) and whatever it models. EnergyPlus exports are the ones used most in this project, but they are one case among several, and anything below that is specific to EnergyPlus says so.

Two things are called "reset" or "restart" in CosimGym. They are **independent concepts**:

| | Model horizon restart | RL episode reset |
| --- | --- | --- |
| **Why** | The model has an intrinsic limit on the span of simulated time it can cover (an EnergyPlus FMU only covers its RunPeriod; a CSV only covers the period of its data) | A reinforcement-learning episode ends and the next one must start from a defined state |
| **When** | Whenever the model reaches that limit — in plain co-simulation, in training and in testing | Only in RL runs, at `reset.period` |
| **Configured by** | the model's own `max_sim_time` / `sim_start_date` (this section) | `reinforcement_learning_config.environment.reset` ([RL Configuration](scenario_configuration/rl.md)) |

The horizon restart is an edge case. It only matters when a scenario (or a training run) lasts longer than a model's limit, and the normal answer is to **choose models whose limit covers the whole scenario**. It is a safety net, not a feature to design studies around.

**Mechanics.** Every restart (RL `full` / `rolling` reset, or horizon) is the same operation: terminate and free the slave, then instantiate a new one whose start time is the model's current position on its horizon window and whose stop time covers the span to the next restart (the federate tells the model that span). Nothing is saved or replayed, so a rolling reset costs one instantiation (~0.5 s for BUI0), independent of where the episode starts. The model's relative clock is not folded back at the horizon (after December comes the next January), only the slave restarts at the window origin; the weekday of the model can therefore differ from a schedule feeder's after a wrap. Each slave writes into its own `fmu_output/restart_<n>/` directory.

### Declaring the limit

A model that has a limit declares it in its catalog entry:

```yaml
models:
  my_building_fmu:
    class_name: BaseFMUModel
    module_path: models.base_FMU_model
    max_sim_time: 31536000      # seconds of simulated time before a restart is required
    sim_start_date: '01-01'     # calendar date the model restarts from
```

| Field | Meaning |
| ----- | ------- |
| `max_sim_time` | Longest span of **model-local** simulated time the model can run in one go, in seconds. `null` or absent means **unbounded**: the model is stepped for the whole scenario and is never restarted. |
| `sim_start_date` | Calendar date that model-local time 0 corresponds to, as `MM-DD` or `YYYY-MM-DD`. For an EnergyPlus FMU it is the RunPeriod begin date, which is not necessarily 1 January. Recorded in the catalog; nothing currently uses it to label results. |

A scenario can override both for one model under that model's `user_defined` block.

> **If your FMU has a limit, you must declare it.** The default is unbounded, so an undeclared limit is not detected: the run simply fails when the FMU is stepped past its run period. The registration script (below) fills both fields in for you.

### EnergyPlus FMUs

An EnergyPlus export is recognised by `user_defined.fmu_engine: energyplus` in its catalog entry (`--engine energyplus` in the register script; not auto-detected yet).

- It can be started at any **midnight** within its maximum time. A start in the middle of a day is silently rounded down to midnight by EnergyPlus, so `BaseFMUModel` refuses it, and `ScenarioManager` requires `rolling_window` to be a whole number of days for such a federate.
- The total duration of an instance (stop minus start) must be a **multiple of 86400 s**, and the smallest step is **60 s**. `max_sim_time` must be a whole number of days, and the scenario must start at 00:00.

### Letting the register script fill this in

`fmu_catalog_register.py` detects both fields so you do not have to compute them:

```bash
# horizon read from the EnergyPlus RunPeriod
python src/models/model_catalog/fmu_catalog_register.py \
    --fmu path/to/BUI0.fmu --name bui0_building_fmu --local \
    --idf path/to/BUI0.idf

# or stated explicitly
python src/models/model_catalog/fmu_catalog_register.py \
    --fmu path/to/model.fmu --name my_fmu --local \
    --max-sim-time 31536000 --sim-start-date 03-01 --engine energyplus
```

Precedence: `--max-sim-time`/`--sim-start-date` override `--idf`, which overrides the FMU's own `DefaultExperiment`. If none of them yields a horizon the model is registered as unbounded and the script warns.

## 5. Worked example — BUI0 building FMU

The repository ships a complete two-federate example wiring a Python schedule feeder into an EnergyPlus building FMU.

- **Scenario:** `src/scenarios/bui0_fmu_test.yaml`
- **FMU:** `src/models/model_catalog/physical_models/resources/BUI0.fmu` (FMI 2.0)
- **Feeder model:** `bui0_input_feeder` — generates six schedule signals (`PeopleNumber`, `LightsWatt`, `EEquipWatt`, `OthEquRadWatt`, `OthEquFCWatt`, `ZoneSetPoint`) from an hour-of-day occupancy profile, plus a `HeatingSeason` flag (1.0/0.0).
- **Season-aware set-point:** the heating season spans `heating_season_start_month`..`heating_season_end_month` (inclusive, wraps across the new year; default `10`..`4`). Inside it the feeder publishes `heating_setpoint_day_c` / `heating_setpoint_night_c` (21/18 °C). Outside it, `cooling_season_mode` decides:
  - `setback` (default) — publish `cooling_season_setback_c` (12 °C). Correct for BUI0, whose thermostat is a `ThermostatSetpoint:SingleHeating` with zero cooling capacity: a summer set-point of 26 °C would make the heating coil chase 26 °C.
  - `cooling` — publish `cooling_setpoint_day_c` / `cooling_setpoint_night_c` (26/28 °C), for zones that really have a cooling thermostat.
  
  `setpoint_day_c` / `setpoint_night_c` still work as deprecated aliases of the heating-season pair. Unit tests: `pytest tests/test_bui0_feeder_season.py`.
- **FMU outputs:** `TBuilding` (zone air temperature), `HeatingLoadTarget`.

Data flow:

```
feeder_federate (bui0_input_feeder)
    ──6 schedule signals──▶ building_federate (bui0_building_fmu, BaseFMUModel)
                                ──▶ TBuilding, HeatingLoadTarget
```

Both federates use `real_period: 600`; one simulated day = 144 steps. Run it:

```python
# src/test_script.py
main('bui0_fmu_test')
```

```bash
docker compose -f src/docker-compose.yaml up -d
python src/models/model_catalog/catalog_loader.py
conda activate cosim_gym
python src/test_script.py
```

Inspect `results/bui0_fmu_test/<sim_id>/federation_1/` — `TBuilding` and `HeatingLoadTarget` should be populated after the first step.
