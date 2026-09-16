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

FMUs generated from EnergyPlus IDF files (`idf-to-fmu-export-prep`) require a **defined stop time**. `BaseFMUModel` derives it automatically from the scenario `start_time`/`end_time` and passes it to `setupExperiment` (FMI 2.0) / `enterInitializationMode` (FMI 3.0). If the stop time is left undefined, EnergyPlus clamps it to 0 and the second `doStep` fails with `fmi2Error`. When the model declares a simulation horizon (next section) that horizon is used instead, because the slave is restarted at it rather than stepped past it.

EnergyPlus FMUs also create runtime working directories named `Output_EPExport_<federate>.<n>/`. These are git-ignored — do not commit them.

## 4. Simulation horizon and automatic restart

Some FMUs cannot run for an unlimited span of simulated time. An EnergyPlus export stops at the end of the **RunPeriod** it was built with — normally one year — and fails on the first step past it. An RL training that runs for many simulated years therefore has to restart the FMU whenever it reaches that limit.

CosimGym does this for you, but **only if the catalog entry says what the limit is**. Two fields declare it:

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
| `sim_start_date` | Calendar date that model-local time 0 corresponds to, as `MM-DD` or `YYYY-MM-DD`. This is the date the model restarts *from* — for an EnergyPlus FMU it is the RunPeriod begin date, which is not necessarily 1 January. |

> **If your FMU has a limit, you must declare it.** The default is unbounded, so an undeclared limit is not detected: the run simply fails when the FMU is stepped past its run period. The registration script (below) fills both fields in for you.

### What a restart does

At a restart the slave is freed, a fresh one is instantiated from the cached unzip directory, and the configured initial state is pushed back into it. Each restart writes to its own `fmu_output/restart_<n>/` directory so successive instances do not overwrite each other's EnergyPlus output.

**Every model in the scenario restarts at the same tick, not just the FMU.** The horizon is resolved once by `ScenarioManager` — from the catalog entries of all the models in the scenario, or from the scenario's `simulation_horizon` key — and handed to every federate, which then restarts all of its models together exactly as it does at an episode boundary. A schedule feeder and the building it feeds go back to their first step together and stay at the same simulated moment:

```
tick          1 ....... 144 | 145 ...... 288 | 289 ......
FMU           t=0 ..... 1 d | t=0 ...... 1 d | t=0 ......
feeder      Jan 1 ..... Jan 2| Jan 1 .... Jan 2| Jan 1 ....
```

`state.time` on every model follows this restarted clock, so a model that decides anything from the date — a schedule, a season, a weather row — rewinds with it. The federate's own clock, which timestamps the results, keeps running forward, so nothing is lost from the record. A model that is never restarted sees exactly the elapsed time it always saw.

The **physical state does not carry over**: an EnergyPlus building restarts from its IDF initial conditions, so indoor temperatures jump at the restart. The restart fires whatever the RL reset policy is, and with no RL at all.

### Interaction with RL episode resets

Episode resets (`reinforcement_learning_config.environment.reset`) and the horizon restart are independent. The horizon is always armed; the reset mode only decides where each episode starts:

| `reset.mode` | What the FMU does at an episode boundary |
| ------------ | ---------------------------------------- |
| `none` | nothing, the slave keeps running between episodes |
| `full` | restarts at its own start date, like the horizon restart |
| `rolling` | restarts and replays forward to the episode's start tick |

The horizon restart is independent of all three: it fires every `max_sim_time` of simulated time regardless, and every federate performs it together.

### How a restart is performed

Two mechanisms, picked automatically from what the FMU itself declares in its `modelDescription`:

| FMU declares | Mechanism | Cost of a restart or a rewind |
| ------------ | --------- | ----------------------------- |
| `canGetAndSetFMUstate="true"` | **save / restore state** | instant, exact, any direction |
| `canGetAndSetFMUstate="false"` | **restart and replay** | a new slave, plus one step per tick replayed |

**Save / restore.** The FMU's state is captured with `fmi2GetFMUstate` (FMI 3: `fmi3GetFMUState`) and put back with the matching setter. The saved blob carries the model's *entire* internal state, so a restore needs no replay and nothing needs to be remembered about the inputs the FMU saw. The state at the first tick is saved once at startup and reused by every full reset and every horizon restart; under `rolling`, the next episode's start point is saved in passing while the current episode runs, so a rewind is a restore rather than a re-simulation. Only one rolling snapshot is kept at a time.

This is not something a scenario can emulate by saving parameters and variables itself: the FMI interface exposes only the declared I/O variables, while the model's real state (an EnergyPlus building's zone thermal mass, surface temperatures, HVAC and warm-up history) lives inside the simulator and is never published as variables.

**Restart and replay.** EnergyPlus exports report `canGetAndSetFMUstate="false"`, so for them the only way to reach any point is to free the slave, instantiate a fresh one from the cached unzip directory and step it forward to the target. Reaching the first tick is cheap; anything later costs one real FMU step per tick replayed. Since the rolling start point slides forward every episode, the total grows with the **square of the episode count**:

```
restarts     = episodes
replay steps ≈ rolling_window × episodes × (episodes − 1) / 2
```

100 episodes with `rolling_window: 10` is about 50 000 replayed steps — minutes. The same 100 episodes with `rolling_window: 2880` is about 14 million — days. The model logs this estimate at startup and then proceeds; nothing is blocked, because a long training is often worth waiting for. If the number is not what you expected, reduce `rolling_window`, reduce the episode count, or set `rolling_window` equal to `episode_length` so each episode continues where the last ended and no rewind is needed at all.

During a replay the FMU is fed the inputs it originally saw at those ticks, recorded as the run went along, so it arrives at the start point in the state it really had. If that history does not cover the requested span the initial inputs are held constant and a warning says the replayed span is approximate. Set `user_defined.fmu_reset.replay_inputs: hold` to skip the recording entirely and always hold.

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
    --max-sim-time 31536000 --sim-start-date 03-01
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
