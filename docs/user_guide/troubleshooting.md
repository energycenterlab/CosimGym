# Troubleshooting

Common failures when running CosimGym scenarios and how to fix them.

## `helics_broker executable not found`

`ScenarioManager` launches `helics_broker` as a subprocess and needs it on `PATH`. It ships inside the conda environment, so this usually means the environment is not active (or you launched from a shell that didn't inherit it).

```bash
conda activate cosim_gym
# if it persists, ensure the env's bin is on PATH:
export PATH="$CONDA_PREFIX/bin:$PATH"
```

## Redis connection refused / federates hang at startup

Redis must be running **before** you start a simulation — every federate process pulls its config and the model catalog from Redis.

```bash
docker compose -f src/docker-compose.yaml up -d
docker compose -f src/docker-compose.yaml logs -f redis    # check it is healthy
```

## `model_name '<x>' not found` / stale model after editing the catalog

The catalog lives in Redis, not on disk at runtime. After editing `catalog.yaml`, reload it:

```bash
python src/models/model_catalog/catalog_loader.py
```

## Port already in use / broker fails to bind

A previous run may have left brokers behind, or another process holds the port. Kill leftover brokers and retry; if you set `broker_config.port` explicitly, pick a free one (default base is `23404`).

```bash
pkill -f helics_broker      # clear stale brokers from a crashed run
```

## `fmi2DoStep failed with status 3` on an EnergyPlus FMU

Two usual causes:

1. **Communication step mismatch** — the FMU has a fixed step (`canHandleVariableCommunicationStepSize=false`). Set `real_period` to that exact step (EnergyPlus `Timestep=6/hr` → `600` s).
2. **Undefined stop time** — handled automatically by `BaseFMUModel` from `start_time`/`end_time`. Make sure both are valid ISO 8601 datetimes in the scenario.

See [FMU Models](fmu_models.md).

## `Output_EPExport_*/` folders appearing in the repo

EnergyPlus FMUs create these runtime working directories. They are git-ignored (`Output_EPExport_*/`) and safe to delete between runs.

## Algebraic-loop / cycle error at launch

If two federates subscribe to each other at the same step, `validate_causality_cycles` raises a `RuntimeError` describing the cycle. Break it by marking one subscription `causality: "next_step"`. See [Synchronization & Causality](scenario_configuration/synchronization.md).

## Startup-sync warnings about missing or invalid inputs

A federate's inputs were not populated before its first step (often a stale `-1.0e49` HELICS default, or an upstream federate that hasn't published). Check the wiring of `subscribes.targets`, and review `startup_sync` policy. See [Synchronization & Causality](scenario_configuration/synchronization.md).

## `TypeError: main() got an unexpected keyword argument 'enable_progress_bar'`

`main()` now takes only `scenario_name`. Remove the `enable_progress_bar=...` argument from your `main(...)` call.

## `FileNotFoundError: 'src/core/mappings.yaml'` in a federate's `.stdio.log`, manager hangs at "Monitoring N federates"

Caused by running from inside `src/` (e.g. `cd src && python -c "..."`). Federate subprocesses resolve config paths (like `src/core/mappings.yaml`) relative to the **repository root**, not `src/`. Always launch from the repo root — either `python src/test_script.py`-style (script lives under `src/`, cwd stays root) or `PYTHONPATH=src python -c "from core.ScenarioManager import main; main('...')"` from the root. If you hit this, the federates die immediately but the broker (started from the correct cwd) stays up — kill stray processes and retry:
```bash
pkill -f helics_broker
```

## MQTT / digital-twin features silently produce no data

`streaming.stream: true` or an interface federate's `interface_config` needs Mosquitto running (`docker compose -f src/docker-compose.yaml up -d`; check with `docker compose -f src/docker-compose.yaml logs -f mosquitto`). Confirm the host port: the compose file maps Mosquitto to `11883` (not the default `1883`, which may already be used by a system-wide broker). `MqttAdapter` defaults to `localhost:11883`; override via `MQTT_HOST`/`MQTT_PORT` env vars or `interface_config.adapter.params`. See [Digital-Twin Interfaces & Live Streaming](digital_twin_interfaces.md).
