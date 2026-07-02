# Dashboard & Analytics

Co-simulation outputs high-frequency time-series data across dozens of models. To help rapidly inspect these massive traces and evaluate AI training success, CosimGym features an automated runtime Streamlit Dashboard.

## The Data Pipeline

1. **JSON results (default, `memory_config.sink: json`):** At the end of a run, each federate writes its buffered timeseries to JSON files under `results/<scenario_name>/<sim_id>/<federation_name>/` (`<federate>_<train|test>_storage.json`). **This is the only sink the dashboard currently reads.**
2. **Dashboard's own Parquet cache:** The first time the dashboard opens a run, it converts those JSON files into a compressed `.parquet` cache (`~/.cosim_dashboard_cache/`, via `src/dashboard/dashboard_parquet_cache.py`) for fast subsequent re-loading. This is a **read-side acceleration cache built by the dashboard**, unrelated to point 3.
3. **`memory_config.sink: parquet` (non-blocking federate-level storage):** an alternative to the JSON sink for large/long-running scenarios — federates write their results incrementally, via a background thread, directly to `<federate>_<train|test>_storage.parquet` files (same `results/...` layout, same tidy schema the dashboard already uses internally). **The dashboard does not yet read these files** — `load_all_records()` only globs `*.json`. Use `sink: json` if you need to view a run's results in this dashboard. See [General Scenario Configuration](scenario_configuration/general.md#memory_config) for the full `sink` reference.
4. **InfluxDB (optional/legacy):** A real-time metric-push path exists in `src/utils/influxdb_client.py`, but it is not part of the default Docker stack and is disabled unless explicitly configured.

## How To Use The Dashboard

If you used the Makefile setup:
```bash
make run-dashboard
```

Once running on `localhost:8501`, the interface provides several core panels:

1. **Test Selector:** A dropdown scanning the `results/` and `logs/` folders automatically allowing you to switch between separate runs (e.g. comparing `simple_test` to `pv_batt_SAC`).
2. **Federation Viewer:** Select a specific Federation to examine.
3. **Data Sub-Selections:** 
   - Choose which `Models` you want to overlay.
   - Select variables. Standard models often output variables spanning differing orders of magnitude (e.g. HVAC Power (kW) vs Temperature (C)). The application graphs dynamically split axes to keep visualizations clear.
4. **Reinforcement Learning Performance:** If the dashboard detects agent action files or reward logs in the test's directory footprint, it renders separate plotting tabs dedicated to Policy Loss metrics and Episode Reward traces.

### Headless Environments

Because Streamlit binds dynamically to local disk folders, if you are running CosimGym on a remote AWS/SLURM cluster, ensure you forward port `8501` to your local machine:
`ssh -L 8501:localhost:8501 target_host`

## Live View (during a run)

The panels above read `results/` files written **after** a run finishes. To watch data as it is
produced, run the separate live dashboard, which subscribes to the Mosquitto broker instead of the
filesystem:

```bash
./src/dashboard/run_live_dashboard.sh    # http://localhost:8053
```

It shows whatever is being published to `cosim/#` — any federate's `streaming.stream: true`
telemetry, or an interface federate's `interface_config` bridges. See
[Digital-Twin Interfaces & Live Streaming](digital_twin_interfaces.md) for the full picture.

The live dashboard's sidebar can also **launch a scenario directly** — pick any file from
`src/scenarios/` and click **▶ Start** to run it as a background process (**■ Stop** to kill it),
with a log tail so you can see it boot, all without a separate terminal. Only scenarios with
`streaming.stream: true` or a `type: interface` federate will actually produce data to watch.