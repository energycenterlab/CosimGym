"""Plotly chart builders for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_COLORS = px.colors.qualitative.T10
TIME_SERIES_GROUP_COLUMNS = [
    "federation",
    "federate",
    "model_instance",
    "attribute",
    "type",
    "mode",
]

# Maximum traces per chart to avoid browser/rendering performance issues
MAX_TRACES_PER_CHART = 50


def build_stacked_time_series_figure(plot_frames: list[tuple[str, pd.DataFrame]]) -> go.Figure:
    """Build vertically stacked time-series subplots sharing the same time axis."""
    figure = make_subplots(
        rows=len(plot_frames),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[label for label, _dataframe in plot_frames],
    )

    total_trace_count = 0
    traces_skipped = 0
    
    for row_index, (_plot_label, dataframe) in enumerate(plot_frames, start=1):
        groups = dataframe.groupby(TIME_SERIES_GROUP_COLUMNS, sort=True)
        group_list = list(groups)
        
        # Calculate skip rate if we have too many traces
        skip_rate = 1
        remaining_traces = MAX_TRACES_PER_CHART - total_trace_count
        if len(group_list) > remaining_traces:
            skip_rate = max(1, len(group_list) // remaining_traces)

        for trace_index, (
            (_federation, federate, model, attribute, data_type, mode),
            group,
        ) in enumerate(group_list):
            # Skip traces if we exceed max
            if trace_index % skip_rate != 0:
                traces_skipped += 1
                continue
            
            total_trace_count += 1
            if total_trace_count > MAX_TRACES_PER_CHART:
                traces_skipped += len(group_list) - trace_index
                break
            
            color = PLOT_COLORS[(trace_index + row_index - 1) % len(PLOT_COLORS)]
            label = f"{federate} / {model} / {attribute} [{data_type}] ({mode})"
            figure.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=group["value"],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.7),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Time: %{x}<br>Value: %{y:.4g}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )

        figure.update_yaxes(
            title_text="Value",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            row=row_index,
            col=1,
        )
        figure.update_xaxes(
            showgrid=True,
            gridcolor="#f0f0f0",
            row=row_index,
            col=1,
        )

    figure.update_xaxes(title_text="Time", row=len(plot_frames), col=1)
    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=max(360, 280 * len(plot_frames)),
        margin=dict(l=60, r=220, t=60, b=60),
        legend=dict(
            font=dict(size=11),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            tracegroupgap=8,
        ),
    )
    return figure


def build_episode_metric_figure(
    dataframe: pd.DataFrame,
    value_column: str,
    yaxis_title: str,
    height: int,
    *,
    rolling_window: int | None = None,
    show_rolling_mean: bool = False,
) -> go.Figure:
    """Build an RL episode metrics figure for reward or episode length."""
    figure = go.Figure()

    for index, ((_federation, federate, mode), group) in enumerate(
        dataframe.groupby(["federation", "federate", "mode"], sort=True)
    ):
        group = group.sort_values("episode").reset_index(drop=True)
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        label = f"{federate} ({mode})"
        hover_label = "Reward" if value_column == "episode_reward" else "Length"
        hover_format = ":.4g" if value_column == "episode_reward" else ""

        figure.add_trace(
            go.Scatter(
                x=group["episode"],
                y=group[value_column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=4),
                opacity=0.5 if show_rolling_mean and rolling_window else 1.0,
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b><br>Episode: %{{x}}<br>{hover_label}: %{{y{hover_format}}}<extra></extra>"
                ),
            )
        )

        if show_rolling_mean and rolling_window:
            rolling_series = group[value_column].rolling(window=rolling_window, min_periods=1).mean()
            figure.add_trace(
                go.Scatter(
                    x=group["episode"],
                    y=rolling_series,
                    mode="lines",
                    name=f"{label} mean ({rolling_window})",
                    line=dict(color=color, width=3),
                    hovertemplate=(
                        f"<b>%{{fullData.name}}</b><br>Episode: %{{x}}<br>{hover_label} rolling mean: %{{y{hover_format}}}<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        template="plotly_white",
        xaxis_title="Episode",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=180, t=30, b=60),
        legend=dict(
            font=dict(size=11),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )
    return figure
