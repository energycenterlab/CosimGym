"""Streamlit UI composition for the simulation results dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.dashboard_charts import (
    TIME_SERIES_GROUP_COLUMNS,
    build_episode_metric_figure,
    build_stacked_time_series_figure,
)
from src.dashboard.dashboard_data import (
    RESULTS_PATH,
    build_dataframe as build_dataframe_from_records,
    build_episode_dataframe as build_episode_dataframe_from_records,
    get_available_tags as get_available_tags_from_records,
    list_federations as list_federations_from_disk,
    list_scenarios as list_scenarios_from_disk,
    list_simulation_ids as list_simulation_ids_from_disk,
    load_all_records,
    load_rl_episode_records,
    load_simulation_metadata as load_simulation_metadata_from_disk,
)
from src.dashboard.dashboard_cache import get_global_metadata_index
from src.dashboard.dashboard_parquet_cache import (
    load_records_from_parquet,
    save_records_to_parquet,
)

# ============================================================================
# PERSISTENT FINGERPRINT CACHE
# ============================================================================

FINGERPRINT_CACHE_DIR = Path.home() / ".cosim_dashboard_cache"
FINGERPRINT_CACHE_FILE = FINGERPRINT_CACHE_DIR / "fingerprints.json"


def _get_persistent_fingerprints() -> dict[str, float]:
    """Load persisted fingerprint cache from disk."""
    if not FINGERPRINT_CACHE_FILE.exists():
        return {}
    try:
        with open(FINGERPRINT_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_persistent_fingerprints(fingerprints: dict[str, float]) -> None:
    """Save fingerprint cache to disk."""
    try:
        FINGERPRINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(FINGERPRINT_CACHE_FILE, encoding="utf-8", mode="w") as f:
            json.dump(fingerprints, f)
    except OSError:
        pass  # Silent failure if cache write fails


PAGE_CSS = """
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
    }
    h1, h2 {
        font-weight: 500;
        margin-bottom: 0.3rem;
        font-size: 1.1rem;
    }
    code {
        background-color: #f0f0f0;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
</style>
"""

ALL_OPTION = "All"
PLOT_GROUP_ORDER_KEY = "plot_group_order"
NEXT_PLOT_GROUP_ID_KEY = "next_plot_group_id"
PLOT_STATE_PREFIXES = ("group_", PLOT_GROUP_ORDER_KEY, NEXT_PLOT_GROUP_ID_KEY)


@dataclass(frozen=True)
class PlotFilters:
    federations: tuple[str, ...] | None = None
    federates: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    attributes: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PlotGroup:
    group_id: int
    label: str
    dataframe: pd.DataFrame
    filters: PlotFilters


@st.cache_data(ttl=300)
def get_scenarios() -> list[str]:
    """Get scenarios using metadata index cache for faster lookups."""
    index = get_global_metadata_index()
    scenarios_dict = index.get_scenarios(RESULTS_PATH)
    return list(scenarios_dict.keys())


@st.cache_data(ttl=300)
def get_simulation_ids(scenario: str) -> list[str]:
    """Get simulation IDs using metadata index cache."""
    index = get_global_metadata_index()
    scenarios_dict = index.get_scenarios(RESULTS_PATH)
    scenario_data = scenarios_dict.get(scenario, {})
    return scenario_data.get("sim_ids", [])


@st.cache_data(ttl=300)
def get_simulation_metadata(scenario: str, sim_id: str):
    return load_simulation_metadata_from_disk(scenario, sim_id)


@st.cache_data(ttl=300)
def get_federations(scenario: str, sim_id: str) -> list[str]:
    return list_federations_from_disk(scenario, sim_id)


def get_all_records(scenario: str, sim_id: str) -> list[dict]:
    key = f"records_{scenario}_{sim_id}"
    fingerprint_key = f"records_fp_{scenario}_{sim_id}"
    run_path = RESULTS_PATH / scenario / sim_id
    fingerprint = _compute_run_fingerprint(run_path)
    
    if (
        fingerprint == -1
        or st.session_state.get(fingerprint_key) != fingerprint
        or key not in st.session_state
    ):
        # Try to load from Parquet cache first (5-10x faster)
        records = load_records_from_parquet(scenario, sim_id, RESULTS_PATH, "timeseries")
        
        # Fall back to JSON if Parquet cache doesn't exist or is stale
        if records is None:
            records = load_all_records(scenario, sim_id)
            # Save to Parquet for future loads
            save_records_to_parquet(records, scenario, sim_id, "timeseries")
        
        st.session_state[key] = records
        st.session_state[fingerprint_key] = fingerprint
    return st.session_state[key]


def get_all_episode_records(scenario: str, sim_id: str) -> list[dict]:
    key = f"episodes_{scenario}_{sim_id}"
    fingerprint_key = f"episodes_fp_{scenario}_{sim_id}"
    run_path = RESULTS_PATH / scenario / sim_id
    fingerprint = _compute_run_fingerprint(run_path)
    
    if (
        fingerprint == -1
        or st.session_state.get(fingerprint_key) != fingerprint
        or key not in st.session_state
    ):
        # Try to load from Parquet cache first
        records = load_records_from_parquet(scenario, sim_id, RESULTS_PATH, "episodes")
        
        # Fall back to JSON if Parquet cache doesn't exist or is stale
        if records is None:
            records = load_rl_episode_records(scenario, sim_id)
            # Save to Parquet for future loads
            save_records_to_parquet(records, scenario, sim_id, "episodes")
        
        st.session_state[key] = records
        st.session_state[fingerprint_key] = fingerprint
    return st.session_state[key]


def get_available_tags(
    scenario: str,
    sim_id: str,
    tag: str,
    federations=None,
    federates=None,
    models=None,
    attributes=None,
    modes=None,
) -> list[str]:
    """Get available tag values with memoization during session."""
    # Create a cache key from the filter parameters
    cache_key = (
        f"tags_{scenario}_{sim_id}_{tag}_{federations}_{federates}_{models}_{attributes}_{modes}"
    )
    
    # Check if we have a cached result in session state
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Compute the result
    result = get_available_tags_from_records(
        get_all_records(scenario, sim_id),
        tag,
        federations=federations,
        federates=federates,
        models=models,
        attributes=attributes,
        modes=modes,
    )
    
    # Cache it for the session
    st.session_state[cache_key] = result
    return result


def build_dataframe(
    scenario: str,
    sim_id: str,
    federations=None,
    federates=None,
    models=None,
    attributes=None,
    modes=None,
):
    """Build filtered dataframe with session-local memoization."""
    # Create cache key from filter parameters (tuples are hashable for caching)
    cache_key = (
        f"df_{scenario}_{sim_id}_"
        f"{tuple(federations) if federations else None}_"
        f"{tuple(federates) if federates else None}_"
        f"{tuple(models) if models else None}_"
        f"{tuple(attributes) if attributes else None}_"
        f"{tuple(modes) if modes else None}"
    )
    
    # Check session cache first (per-browser-session cache)
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Compute the result
    result = build_dataframe_from_records(
        get_all_records(scenario, sim_id),
        federations=federations,
        federates=federates,
        models=models,
        attributes=attributes,
        modes=modes,
    )
    
    # Cache it for the session
    st.session_state[cache_key] = result
    return result


def build_episode_dataframe(scenario: str, sim_id: str):
    """Build episode dataframe with session-local memoization."""
    cache_key = f"episode_df_{scenario}_{sim_id}"
    
    # Check session cache first
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Compute the result
    result = build_episode_dataframe_from_records(get_all_episode_records(scenario, sim_id))
    
    # Cache it for the session
    st.session_state[cache_key] = result
    return result


def _compute_run_fingerprint(run_path: Path) -> float:
    """
    Compute a fingerprint (max mtime) for a run, using persistent cache.
    This avoids re-scanning all JSON files on every page refresh.
    """
    run_key = str(run_path)
    
    if not run_path.exists():
        return -1
    
    # Load persistent cache
    persistent_fp = _get_persistent_fingerprints()
    cached_fp = persistent_fp.get(run_key)
    
    # Check if cached fingerprint is still valid by quickly checking the directory
    # (stat on directory is much cheaper than rglob on all files)
    try:
        dir_mtime = run_path.stat().st_mtime
        if cached_fp is not None and cached_fp >= dir_mtime:
            # Cache is still valid (directory hasn't been modified)
            return cached_fp
    except OSError:
        pass
    
    # Cache miss or invalid; recompute
    max_mtime = -1.0
    for json_file in run_path.rglob("*.json"):
        try:
            mtime = json_file.stat().st_mtime
        except OSError:
            continue
        if mtime > max_mtime:
            max_mtime = mtime
    
    # Update persistent cache
    if max_mtime > 0:
        persistent_fp[run_key] = max_mtime
        _save_persistent_fingerprints(persistent_fp)
    
    return max_mtime


def run_dashboard() -> None:
    initialize_session_state()
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    selected_scenario, selected_sim = render_sidebar()

    metadata = get_simulation_metadata(selected_scenario, selected_sim)
    if not metadata:
        st.error(f"`metadata.json` not found for run `{selected_sim}`.")
        st.stop()

    st.title("Simulation Results Explorer")
    
    # Performance diagnostics (optional expander)
    with st.expander("⚡ Performance Info", expanded=False):
        import sys
        import os
        from pathlib import Path
        
        # Calculate memory usage
        session_cache_size = sum(
            sys.getsizeof(v) for k, v in st.session_state.items() 
            if k.startswith(("records_", "tags_", "df_", "episode_"))
        )
        
        # Check disk cache
        cache_dir = Path.home() / ".cosim_dashboard_cache"
        parquet_dir = cache_dir / "parquet"
        if parquet_dir.exists():
            parquet_size = sum(f.stat().st_size for f in parquet_dir.rglob("*") if f.is_file())
        else:
            parquet_size = 0
        
        cache_size = len([k for k in st.session_state.keys() if k.startswith(("records_", "tags_", "df_", "episode_"))])
        parquet_cache_exists = (Path.home() / ".cosim_dashboard_cache" / "parquet").exists()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Session RAM (cached data)", f"{session_cache_size / 1024 / 1024:.1f} MB")
            st.metric("Cache Entries", cache_size)
        with col2:
            st.metric("Disk Cache (Parquet)", f"{parquet_size / 1024 / 1024:.1f} MB")
            st.metric("Parquet Status", "✓ Active" if parquet_cache_exists else "✗ None yet")
        
        st.divider()
        st.caption(
            "**💡 Cache Behavior:**\n"
            "- Session RAM clears on browser refresh or tab close\n"
            "- Disk cache persists (speeds up repeat loads)\n"
            "- Clear disk cache: `rm -rf ~/.cosim_dashboard_cache/parquet/`"
        )
        
        st.info(
            f"**Performance Tips:**\n"
            f"- Filter by federation/mode to reduce data size\n"
            f"- Repeat loads will use Parquet (5-10x faster)\n"
            f"- Uncheck 'Summary' if only viewing time series"
        )
    
    render_simulation_overview(metadata, selected_scenario, selected_sim)
    federations = render_data_overview(selected_scenario, selected_sim)
    render_time_series_section(selected_scenario, selected_sim, federations)
    render_rl_episode_section(selected_scenario, selected_sim)
    render_footer()


def initialize_session_state() -> None:
    if PLOT_GROUP_ORDER_KEY not in st.session_state:
        st.session_state[PLOT_GROUP_ORDER_KEY] = [0]
    if NEXT_PLOT_GROUP_ID_KEY not in st.session_state:
        st.session_state[NEXT_PLOT_GROUP_ID_KEY] = 1


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## Results Explorer")
        st.markdown("---")

        scenarios = get_scenarios()
        if not scenarios:
            st.info(
                "No results found yet. Run `make run` or copy your federate JSON outputs into "
                f"`{RESULTS_PATH}` so the dashboard can list scenarios."
            )
            st.stop()

        selected_scenario = st.selectbox("Scenario", options=scenarios)
        simulation_ids = get_simulation_ids(selected_scenario)
        if not simulation_ids:
            st.warning("No simulation runs found for this scenario.")
            st.stop()

        selected_sim = st.selectbox("Simulation Run", options=simulation_ids)
        _reset_plot_state_if_selection_changed(selected_scenario, selected_sim)

        st.markdown("---")
        
        # Cache management
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔄 Refresh Data",
                help="Clear all in-memory and persistent caches, rescan results folder",
                use_container_width=True,
            ):
                st.cache_data.clear()
                get_global_metadata_index().invalidate()
                from src.dashboard.dashboard_parquet_cache import clear_parquet_cache
                clear_parquet_cache()
                st.rerun()
        
        with col2:
            if st.button(
                "🗑️ Clear Session",
                help="Clear only in-memory cache (frees RAM)",
                use_container_width=True,
            ):
                st.session_state.clear()
                st.rerun()
        
        st.caption("Source: `results/`")

    return selected_scenario, selected_sim


def render_simulation_overview(metadata: dict, selected_scenario: str, selected_sim: str) -> None:
    scenario_name = metadata.get('scenario_name', selected_scenario)
    start_date = metadata.get('start_date', '—')
    end_date = metadata.get('end_date', '—')
    dur = metadata.get("duration_seconds")
    duration_text = f"{dur:.0f} s" if dur is not None else "—"
    
    # Compact info box
    info_line = f"<strong>{scenario_name}</strong> · <code style='font-size:0.85rem'>{selected_sim}</code> · {start_date} to {end_date} · {duration_text}"
    
    st.markdown(
        f"<div style='padding:0.6rem;background:#f8f8f8;border-left:3px solid #888;font-size:0.95rem;margin-bottom:0.5rem'>{info_line}</div>",
        unsafe_allow_html=True,
    )


def render_data_overview(selected_scenario: str, selected_sim: str) -> list[str]:
    federations = get_federations(selected_scenario, selected_sim)
    all_federates = get_available_tags(selected_scenario, selected_sim, "federate")
    all_models = get_available_tags(selected_scenario, selected_sim, "model_instance")
    all_attributes = get_available_tags(selected_scenario, selected_sim, "attribute")
    all_modes = get_available_tags(selected_scenario, selected_sim, "mode")

    # Compact single-line summary
    summary = (
        f"<strong>Data:</strong> "
        f"{len(federations)} federations · "
        f"{len(all_federates)} federates · "
        f"{len(all_models)} models · "
        f"{len(all_attributes)} attributes · "
        f"Modes: {', '.join(all_modes) if all_modes else '—'}"
    )
    
    st.markdown(
        f"<div style='padding:0.5rem;background:#f8f8f8;font-size:0.9rem;margin-bottom:1rem'>{summary}</div>",
        unsafe_allow_html=True,
    )
    
    return federations


def render_time_series_section(
    selected_scenario: str,
    selected_sim: str,
    federations: list[str],
) -> None:
    col1, col2 = st.columns([0.12, 0.88])
    with col1:
        if st.button("➕ Plot", key="add_plot_group", width="stretch"):
            _add_plot_group()
            st.rerun()
    with col2:
        st.markdown("**Time Series** – Configure plot groups below (reorder with buttons)")

    plot_groups = [
        render_plot_group(
            position,
            group_id,
            selected_scenario,
            selected_sim,
            federations,
            total_groups=len(get_plot_group_ids()),
        )
        for position, group_id in enumerate(get_plot_group_ids())
    ]

    st.markdown("### Comparison View")

    populated_groups = [group for group in plot_groups if not group.dataframe.empty]
    if not populated_groups:
        st.info("Choose filters for at least one plot to render the shared time-axis comparison view.")
        st.markdown("---")
        return
    
    # Performance info: show data size
    total_rows = sum(len(g.dataframe) for g in populated_groups)
    if total_rows > 100000:
        st.warning(
            f"⚠️ **Large dataset**: {total_rows:,} records will render. "
            f"Use federation/mode filters to reduce data size for faster rendering."
        )
    elif total_rows > 50000:
        st.info(f"📊 Rendering {total_rows:,} records across {len(populated_groups)} plot(s)")

    st.plotly_chart(
        build_stacked_time_series_figure(
            [(group.label, group.dataframe) for group in populated_groups]
        ),
    )

    export_dataframe = pd.concat(
        [
            group.dataframe.assign(plot_group=group.label)
            for group in populated_groups
        ],
        ignore_index=True,
    )
    st.download_button(
        label="Download plotted comparison data (CSV)",
        data=export_dataframe.to_csv(index=False),
        file_name=f"{selected_sim}_stacked_plots.csv",
        mime="text/csv",
        key="download_stacked_plots",
    )
    st.markdown("---")


def render_plot_group(
    position: int,
    group_id: int,
    selected_scenario: str,
    selected_sim: str,
    federations: list[str],
    total_groups: int,
) -> PlotGroup:
    default_label = f"Plot {position + 1}"
    label_key = f"group_{group_id}_label"

    title_column, spacer, up_column, down_column, remove_column = st.columns([4.5, 1, 0.7, 0.7, 0.7])
    with title_column:
        custom_label = st.text_input(
            "Plot label",
            value=st.session_state.get(label_key, ""),
            placeholder=default_label,
            key=label_key,
        )

    with up_column:
        if st.button("⬆️", key=f"group_{group_id}_up", disabled=(position == 0), help="Move plot up"):
            _move_plot_group(group_id, -1)
            st.rerun()

    with down_column:
        if st.button("⬇️", key=f"group_{group_id}_down", disabled=(position == total_groups - 1), help="Move plot down"):
            _move_plot_group(group_id, 1)
            st.rerun()

    with remove_column:
        if st.button("🗑️", key=f"group_{group_id}_remove", disabled=(total_groups == 1), help="Remove this plot"):
            _remove_plot_group(group_id)
            st.rerun()

    plot_label = custom_label.strip() or default_label
    filters = render_plot_filters(group_id, selected_scenario, selected_sim, federations)
    
    # Keys for plot generation state
    data_key = f"group_{group_id}_data"
    filters_key = f"group_{group_id}_filters"
    render_key = f"group_{group_id}_render"
    current_filters_key = f"group_{group_id}_current_filters"
    
    # Check if filters have changed (dirty state)
    last_rendered_filters = st.session_state.get(current_filters_key)
    is_dirty = last_rendered_filters is not None and filters != last_rendered_filters
    
    # Generate Plot button with visual feedback for dirty state
    button_label = "🔴 Generate Plot" if is_dirty else "📊 Generate Plot"
    button_help = f"Filters changed since last render" if is_dirty else "Render plot with current filters"
    
    col_gen, col_status = st.columns([1.5, 2])
    with col_gen:
        if st.button(button_label, key=f"group_{group_id}_generate", help=button_help):
            # User clicked Generate - build dataframe and mark as rendered
            dataframe = build_dataframe(
                selected_scenario,
                selected_sim,
                federations=filters.federations,
                federates=filters.federates,
                models=filters.models,
                attributes=filters.attributes,
                modes=filters.modes,
            )
            st.session_state[data_key] = dataframe
            st.session_state[current_filters_key] = filters
            st.session_state[filters_key] = filters
            st.rerun()
    
    # Show status
    with col_status:
        if data_key in st.session_state and not is_dirty:
            st.caption("✅ Plot up-to-date")
        elif is_dirty:
            st.caption("🔄 Filters changed – click Generate Plot to update")
        else:
            st.caption("⏸️ Click Generate Plot to render")
    
    # Only show plot if it's been generated
    dataframe = st.session_state.get(data_key, pd.DataFrame())
    
    if not dataframe.empty:
        show_summary = st.checkbox(
            "Show summary",
            value=True,
            key=f"group_{group_id}_summary",
        )
        if show_summary:
            curve_count = dataframe.groupby(TIME_SERIES_GROUP_COLUMNS).ngroups
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.5])
            c1.metric("Curves", curve_count)
            c2.metric("Points", f"{len(dataframe):,}")
            c3.metric("Mean", f"{dataframe['value'].mean():.4g}")
            c4.metric("Max", f"{dataframe['value'].max():.4g}")
            c5.download_button(
                label=f"Download {plot_label}",
                data=dataframe.to_csv(index=False),
                file_name=f"{selected_sim}_{make_safe_filename(plot_label)}.csv",
                mime="text/csv",
                key=f"group_{group_id}_download",
            )

    st.divider()

    return PlotGroup(
        group_id=group_id,
        label=plot_label,
        dataframe=dataframe,
        filters=filters,
    )


def render_plot_filters(
    group_id: int,
    selected_scenario: str,
    selected_sim: str,
    federations: list[str],
) -> PlotFilters:
    filter_columns = st.columns(5)

    with filter_columns[0]:
        selected_federations = select_filter_values(
            "Federation",
            federations,
            key=f"group_{group_id}_pf0",
        )

    available_federates = get_available_tags(
        selected_scenario,
        selected_sim,
        "federate",
        federations=selected_federations,
    )
    with filter_columns[1]:
        selected_federates = select_filter_values(
            "Federate",
            available_federates,
            key=f"group_{group_id}_pf1",
        )

    available_models = get_available_tags(
        selected_scenario,
        selected_sim,
        "model_instance",
        federations=selected_federations,
        federates=selected_federates,
    )
    with filter_columns[2]:
        selected_models = select_filter_values(
            "Model Instance",
            available_models,
            key=f"group_{group_id}_pf2",
        )

    available_attributes = get_available_tags(
        selected_scenario,
        selected_sim,
        "attribute",
        federations=selected_federations,
        federates=selected_federates,
        models=selected_models,
    )
    with filter_columns[3]:
        selected_attributes = select_filter_values(
            "Attribute",
            available_attributes,
            key=f"group_{group_id}_pf3",
        )

    available_modes = get_available_tags(
        selected_scenario,
        selected_sim,
        "mode",
        federations=selected_federations,
        federates=selected_federates,
        models=selected_models,
        attributes=selected_attributes,
    )

    with filter_columns[4]:
        selected_modes = select_filter_values(
            "Mode",
            available_modes,
            key=f"group_{group_id}_pf5",
            help_text="train / test",
        )

    return PlotFilters(
        federations=selected_federations,
        federates=selected_federates,
        models=selected_models,
        attributes=selected_attributes,
        modes=selected_modes,
    )


def render_rl_episode_section(selected_scenario: str, selected_sim: str) -> None:
    episode_dataframe = build_episode_dataframe(selected_scenario, selected_sim)
    if episode_dataframe.empty:
        return

    for mode in sorted(episode_dataframe["mode"].unique()):
        mode_slice = episode_dataframe[episode_dataframe["mode"] == mode]
        
        # Compact stats display
        stats_line = (
            f"<strong>RL Episodes ({mode}):</strong> "
            f"{len(mode_slice)} episodes · "
            f"Avg Reward: {mode_slice['episode_reward'].mean():.4g} · "
            f"Best: {mode_slice['episode_reward'].max():.4g} · "
            f"Avg Length: {mode_slice['episode_length'].mean():.1f}"
        )
        st.markdown(
            f"<div style='padding:0.5rem;background:#f8f8f8;font-size:0.9rem;margin-bottom:0.8rem'>{stats_line}</div>",
            unsafe_allow_html=True,
        )

    max_episode_count = int(episode_dataframe["episode"].max()) + 1
    rolling_window = None
    if max_episode_count >= 2:
        control_column, info_column = st.columns([2, 4])
        with control_column:
            rolling_window = st.slider(
                "Reward rolling mean window",
                min_value=2,
                max_value=max_episode_count,
                value=min(10, max_episode_count),
                help="Overlay a rolling mean on the episode reward chart.",
            )
        with info_column:
            st.caption("Rolling mean overlaid for convergence inspection")

    st.plotly_chart(
        build_episode_metric_figure(
            episode_dataframe,
            value_column="episode_reward",
            yaxis_title="Episode Reward",
            height=420,
            rolling_window=rolling_window,
            show_rolling_mean=(rolling_window is not None),
        ),
    )
    st.plotly_chart(
        build_episode_metric_figure(
            episode_dataframe,
            value_column="episode_length",
            yaxis_title="Episode Length (steps)",
            height=320,
        ),
    )

    st.download_button(
        label="Download RL Episode Data (CSV)",
        data=episode_dataframe.to_csv(index=False),
        file_name=f"{selected_sim}_rl_episodes.csv",
        mime="text/csv",
        key="download_rl_episodes",
    )
    st.markdown("---")


def render_footer() -> None:
    st.markdown("---")
    st.caption("Simulation Results Explorer - local file viewer for co-simulation experiments")


def select_filter_values(
    label: str,
    options: list[str],
    *,
    key: str,
    help_text: str | None = None,
) -> tuple[str, ...] | None:
    selected_values = st.multiselect(
        label,
        options=[ALL_OPTION, *options],
        default=[ALL_OPTION],
        key=key,
        help=help_text,
    )
    normalized_values = tuple(value for value in selected_values if value != ALL_OPTION)
    return normalized_values or None


def make_safe_filename(name: str) -> str:
    sanitized = re.sub(r"\s+", "_", name.strip())
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "", sanitized)
    return sanitized or "plot"


def get_plot_group_ids() -> list[int]:
    return list(st.session_state[PLOT_GROUP_ORDER_KEY])


def _add_plot_group() -> None:
    new_group_id = st.session_state[NEXT_PLOT_GROUP_ID_KEY]
    st.session_state[PLOT_GROUP_ORDER_KEY] = [
        *st.session_state[PLOT_GROUP_ORDER_KEY],
        new_group_id,
    ]
    st.session_state[NEXT_PLOT_GROUP_ID_KEY] = new_group_id + 1


def _move_plot_group(group_id: int, direction: int) -> None:
    plot_group_order = get_plot_group_ids()
    current_index = plot_group_order.index(group_id)
    next_index = current_index + direction
    if next_index < 0 or next_index >= len(plot_group_order):
        return

    plot_group_order[current_index], plot_group_order[next_index] = (
        plot_group_order[next_index],
        plot_group_order[current_index],
    )
    st.session_state[PLOT_GROUP_ORDER_KEY] = plot_group_order


def _remove_plot_group(group_id: int) -> None:
    st.session_state[PLOT_GROUP_ORDER_KEY] = [
        existing_group_id
        for existing_group_id in get_plot_group_ids()
        if existing_group_id != group_id
    ]
    _clear_group_state(group_id)


def _clear_group_state(group_id: int) -> None:
    group_prefix = f"group_{group_id}_"
    for key in list(st.session_state):
        if key.startswith(group_prefix):
            del st.session_state[key]


def _reset_plot_state_if_selection_changed(selected_scenario: str, selected_sim: str) -> None:
    simulation_key = (selected_scenario, selected_sim)
    if st.session_state.get("_sim_key") == simulation_key:
        return

    for key in list(st.session_state):
        if key.startswith(PLOT_STATE_PREFIXES):
            del st.session_state[key]

    st.session_state[PLOT_GROUP_ORDER_KEY] = [0]
    st.session_state[NEXT_PLOT_GROUP_ID_KEY] = 1
    st.session_state["_sim_key"] = simulation_key
    st.rerun()
