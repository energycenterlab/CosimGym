# Dashboard Notes

`streamlit_dashboard.py` is the single Streamlit entrypoint (`./run_dashboard.sh`). It uses
`st.navigation` to combine two pages in one app/process:

- **Results** (`dashboard_app.py`): the post-run historical explorer, reading `results/`.
- **Live** (`live_dashboard.py`): the MQTT live view, subscribing to `cosim/#`.

The refactor keeps dashboard-only code local to `src/dashboard`:

- `dashboard_app.py`: Streamlit page composition, sidebar state, and cached wrappers.
- `dashboard_data.py`: pure JSON/Parquet parsing, filtering, and dataframe builders.
- `dashboard_charts.py`: Plotly figure builders reused by the page layer.
- `live_dashboard.py`: MQTT-backed live view (`render_live_dashboard()`), used as the "Live" page.

Current dashboard behavior:

- Time-series comparison plots are stacked vertically and share the same time axis.
- Plot rows can be reordered with `Up` and `Down` in the UI.
- RL episode reward charts can overlay a rolling mean to inspect convergence.

Targeted checks:

```bash
python -m unittest discover -s src/dashboard/tests -p "test_*.py"
python -m compileall src/dashboard
```
