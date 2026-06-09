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

FMUs generated from EnergyPlus IDF files (`idf-to-fmu-export-prep`) require a **defined stop time**. `BaseFMUModel` derives it automatically from the scenario `start_time`/`end_time` and passes it to `setupExperiment` (FMI 2.0) / `enterInitializationMode` (FMI 3.0). If the stop time is left undefined, EnergyPlus clamps it to 0 and the second `doStep` fails with `fmi2Error`.

EnergyPlus FMUs also create runtime working directories named `Output_EPExport_<federate>.<n>/`. These are git-ignored — do not commit them.

## 4. Worked example — BUI0 building FMU

The repository ships a complete two-federate example wiring a Python schedule feeder into an EnergyPlus building FMU.

- **Scenario:** `src/scenarios/bui0_fmu_test.yaml`
- **FMU:** `src/models/model_catalog/physical_models/resources/BUI0.fmu` (FMI 2.0)
- **Feeder model:** `bui0_input_feeder` — generates six schedule signals (`PeopleNumber`, `LightsWatt`, `EEquipWatt`, `OthEquRadWatt`, `OthEquFCWatt`, `ZoneSetPoint`) from an hour-of-day occupancy profile.
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
