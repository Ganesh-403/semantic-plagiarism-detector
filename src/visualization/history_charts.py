"""
src/visualization/history_charts.py
-----------------------------------
Plotly chart generators for the Document Similarity History Dashboard.

Provides functions to visualize historical scan trends, similarity distributions,
and frequently flagged documents over time.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_DEFAULT_BG = "#FFFFFF"
_DEFAULT_INK = "#0F172A"
_DEFAULT_PRIMARY = "#3B82F6"
_DEFAULT_DANGER = "#EF4444"
_DEFAULT_WARNING = "#F59E0B"


def _resolve_color(theme_colors: dict | None, key: str, default: str) -> str:
    """Resolve a theme color with a fallback to the default color."""
    return theme_colors.get(key, default) if theme_colors else default


def plot_similarity_trend_line(
    history_data: list[dict],
    theme_colors: Optional[dict] = None,
) -> go.Figure:
    """Generate a line chart showing average and max similarity trends over time.

    Args:
        history_data: List of scan history dictionaries from get_scan_history().
        theme_colors: Optional theme color dictionary for dark/light mode support.

    Returns:
        Plotly Figure object with dual-line trend chart.
    """
    if not history_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No scan history data available yet.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    df = pd.DataFrame(history_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    bg_color = _resolve_color(theme_colors, "background", _DEFAULT_BG)
    ink_color = _resolve_color(theme_colors, "ink", _DEFAULT_INK)
    danger_color = _resolve_color(theme_colors, "danger", _DEFAULT_DANGER)
    warning_color = _resolve_color(theme_colors, "warning", _DEFAULT_WARNING)

    fig = go.Figure()

    # Average similarity line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["avg_similarity"],
            mode="lines+markers",
            name="Avg Similarity",
            line=dict(color=warning_color, width=3),
            marker=dict(size=8),
        )
    )

    # Max similarity line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["max_similarity"],
            mode="lines+markers",
            name="Max Similarity",
            line=dict(color=danger_color, width=3),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Similarity Trends Over Time",
        xaxis_title="Scan Date",
        yaxis_title="Similarity Score",
        yaxis=dict(range=[0, 1.0], tickformat=".0%"),
        hovermode="x unified",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_flagged_documents_bar(
    history_data: list[dict],
    theme_colors: Optional[dict] = None,
) -> go.Figure:
    """Generate a bar chart showing the number of flagged documents per scan.

    Args:
        history_data: List of scan history dictionaries.
        theme_colors: Optional theme color dictionary.

    Returns:
        Plotly Figure object with bar chart.
    """
    if not history_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No scan history data available yet.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    df = pd.DataFrame(history_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    bg_color = _resolve_color(theme_colors, "background", _DEFAULT_BG)
    ink_color = _resolve_color(theme_colors, "ink", _DEFAULT_INK)
    primary_color = _resolve_color(theme_colors, "primary", _DEFAULT_PRIMARY)

    fig = px.bar(
        df,
        x="date_str",
        y="flagged_count",
        title="Flagged Plagiarism Incidents per Scan",
        labels={"date_str": "Scan Date", "flagged_count": "Flagged Pairs"},
        color_discrete_sequence=[primary_color],
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color),
        xaxis_tickangle=-45,
    )

    return fig
