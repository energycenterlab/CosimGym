# Dashboard & Analytics

Co-simulation outputs high-frequency time-series data across dozens of models. To help rapidly inspect these massive traces and evaluate AI training success, CosimGym features an automated runtime Streamlit Dashboard.

## The Data Pipeline

1. **JSON results (default):** At the end of a run, each federate writes its buffered timeseries to JSON files under `results/<scenario_name>/<sim_id>/<federation_name>/` (`<federate>_<train|test>_storage.json`).
2. **Parquet cache:** The first time the dashboard opens a run, it converts those JSON files into a compressed `.parquet` cache for fast subsequent re-loading.
3. **InfluxDB (optional/legacy):** A real-time metric-push path exists in `src/utils/influxdb_client.py`, but it is not part of the default Docker stack and is disabled unless explicitly configured.

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