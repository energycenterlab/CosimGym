# Dashboard Notes

`streamlit_dashboard.py` remains the Streamlit entrypoint.

The refactor keeps dashboard-only code local to `src/dashboard`:

- `dashboard_app.py`: Streamlit page composition, sidebar state, and cached wrappers.
- `dashboard_data.py`: pure JSON parsing, filtering, and dataframe builders.
- `dashboard_charts.py`: Plotly figure builders reused by the page layer.

Current dashboard behavior:

- Time-series comparison plots are stacked vertically and share the same time axis.
- Plot rows can be reordered with `Up` and `Down` in the UI.
- RL episode reward charts can overlay a rolling mean to inspect convergence.

Targeted checks:

```bash
python -m unittest discover -s src/dashboard/tests -p "test_*.py"
python -m compileall src/dashboard
```
