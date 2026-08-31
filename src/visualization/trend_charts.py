"""
Plagiarism Trend Analytics Visualizations
==========================================
Plotly and Matplotlib visualizations for the trend analytics engine.
Provides interactive charts for the Streamlit dashboard and static
charts for PDF report generation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.debug("plotly not installed — interactive charts unavailable")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.debug("matplotlib not installed — static charts unavailable")

from src.core.plagiarism_trends import (
    AnalyticsReport,
    StatisticalSummary,
    SeverityDistribution,
    TimeSeriesPoint,
    TrendResult,
    TrendWindow,
)


# ── Color Palette ─────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#1e3a8a",
    "secondary": "#3b82f6",
    "accent": "#f59e0b",
    "danger": "#ef4444",
    "success": "#10b981",
    "warning": "#f97316",
    "info": "#06b6d4",
    "bg": "#ffffff",
    "grid": "#e5e7eb",
    "text": "#1f2937",
    "text_light": "#6b7280",
}

SEVERITY_COLORS = {
    "low": "#10b981",
    "medium": "#f59e0b",
    "high": "#ef4444",
    "critical": "#7c2d12",
}

TREND_COLORS = {
    "increasing": "#ef4444",
    "decreasing": "#10b981",
    "stable": "#6b7280",
    "insufficient_data": "#d1d5db",
}


# ── Plotly Visualizations ─────────────────────────────────────────────────────

def create_incident_timeline(
    windows: List[TrendWindow],
    forecast_values: Optional[List[float]] = None,
    forecast_timestamps: Optional[List[datetime]] = None,
    title: str = "Plagiarism Incidents Over Time",
) -> Optional[Any]:
    """Create an interactive timeline of incidents per time window.

    Returns a Plotly Figure or None if plotly is unavailable.
    """
    if not HAS_PLOTLY:
        return None
    if not windows:
        return _empty_figure("No data available for timeline")

    labels = [w.window_label for w in windows]
    counts = [w.incident_count for w in windows]

    fig = go.Figure()

    # Bar chart for actual incidents
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        name="Incidents",
        marker_color=COLORS["primary"],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Incidents: %{y}<extra></extra>",
    ))

    # Line overlay for moving average
    if len(counts) >= 3:
        ma_values = _compute_moving_average(counts, 3)
        fig.add_trace(go.Scatter(
            x=labels,
            y=ma_values,
            name="3-Period Moving Avg",
            mode="lines",
            line=dict(color=COLORS["accent"], width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>MA: %{y:.1f}<extra></extra>",
        ))

    # Forecast bars
    if forecast_values and forecast_timestamps:
        forecast_labels = [f"Forecast {i + 1}" for i in range(len(forecast_values))]
        fig.add_trace(go.Bar(
            x=labels[-1:] + forecast_labels,
            y=[0] * max(0, len(labels) - 1) + [counts[-1]] + forecast_values,
            name="Forecast",
            marker_color=COLORS["info"],
            marker_opacity=0.5,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Predicted: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        xaxis_title="Time Period",
        yaxis_title="Number of Incidents",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], gridwidth=1)
    fig.update_yaxes(gridcolor=COLORS["grid"], gridwidth=1, rangemode="tozero")

    return fig


def create_similarity_trend_chart(
    windows: List[TrendWindow],
    title: str = "Average Similarity Score Trend",
) -> Optional[Any]:
    """Create a line chart of average similarity scores over time."""
    if not HAS_PLOTLY:
        return None
    if not windows:
        return _empty_figure("No data available for similarity trend")

    labels = [w.window_label for w in windows]
    avg_scores = [w.avg_similarity for w in windows]
    max_scores = [w.max_similarity for w in windows]

    fig = go.Figure()

    # Average similarity area
    fig.add_trace(go.Scatter(
        x=labels,
        y=avg_scores,
        name="Avg Similarity",
        mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f}<extra></extra>",
    ))

    # Max similarity line
    fig.add_trace(go.Scatter(
        x=labels,
        y=max_scores,
        name="Max Similarity",
        mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Max: %{y:.3f}<extra></extra>",
    ))

    # Threshold line at 0.59
    fig.add_hline(
        y=0.59,
        line_dash="dot",
        line_color=COLORS["warning"],
        annotation_text="Plagiarism Threshold (0.59)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        xaxis_title="Time Period",
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )

    return fig


def create_severity_donut(
    dist: SeverityDistribution,
    title: str = "Severity Distribution",
) -> Optional[Any]:
    """Create a donut chart of severity distribution."""
    if not HAS_PLOTLY:
        return None
    if dist.total == 0:
        return _empty_figure("No incidents to display")

    labels = ["Low", "Medium", "High", "Critical"]
    values = [dist.low, dist.medium, dist.high, dist.critical]
    colors = [SEVERITY_COLORS["low"], SEVERITY_COLORS["medium"],
              SEVERITY_COLORS["high"], SEVERITY_COLORS["critical"]]

    # Filter out zero values for cleaner chart
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return _empty_figure("No incidents to display")

    labels_f, values_f, colors_f = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=labels_f,
        values=values_f,
        hole=0.45,
        marker=dict(colors=colors_f, line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))

    # Add center annotation
    fig.add_annotation(
        text=f"<b>{dist.total}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=16, color=COLORS["text"]),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        paper_bgcolor=COLORS["bg"],
    )

    return fig


def create_offender_bar_chart(
    offenders: List[Any],
    top_n: int = 10,
    title: str = "Top Plagiarism Offenders",
) -> Optional[Any]:
    """Create a horizontal bar chart of top offenders."""
    if not HAS_PLOTLY:
        return None
    if not offenders:
        return _empty_figure("No offender data available")

    top = offenders[:top_n]
    names = [o.document_name[:40] for o in reversed(top)]
    counts = [o.incident_count for o in reversed(top)]
    max_sims = [o.max_similarity for o in reversed(top)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=counts,
        orientation="h",
        name="Incidents",
        marker_color=COLORS["primary"],
        hovertemplate="<b>%{y}</b><br>Incidents: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis_title="Number of Incidents",
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(top) * 35 + 80),
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )

    return fig


def create_trend_summary_card(
    trend: TrendResult,
    title: str = "Trend Analysis",
) -> Optional[Any]:
    """Create a compact trend summary indicator chart."""
    if not HAS_PLOTLY:
        return None

    direction = trend.direction.value
    color = TREND_COLORS.get(direction, COLORS["text_light"])

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=trend.confidence,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": f"{title}<br><span style='font-size:14px;color:{color}'>"
                        f"{direction.upper()}</span>",
               "font": {"size": 16, "color": COLORS["text"]}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 50], color="#fee2e2"),
                dict(range=[50, 80], color="#fef3c7"),
                dict(range=[80, 100], color="#d1fae5"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=trend.confidence,
            ),
        ),
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor=COLORS["bg"],
    )

    return fig


def create_window_comparison_heatmap(
    windows: List[TrendWindow],
    title: str = "Window Comparison Matrix",
) -> Optional[Any]:
    """Create a heatmap comparing metrics across time windows."""
    if not HAS_PLOTLY:
        return None
    if len(windows) < 2:
        return _empty_figure("Need at least 2 windows for comparison")

    labels = [w.window_label for w in windows]
    metrics = ["Incidents", "Avg Sim", "Max Sim", "Unique Docs"]
    matrix = [
        [w.incident_count for w in windows],
        [w.avg_similarity for w in windows],
        [w.max_similarity for w in windows],
        [w.unique_documents for w in windows],
    ]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=metrics,
        colorscale="Blues",
        hovertemplate="<b>%{y}</b> in %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        height=280,
        margin=dict(l=100, r=20, t=50, b=40),
        paper_bgcolor=COLORS["bg"],
    )

    return fig


def create_severity_timeline(
    windows: List[TrendWindow],
    title: str = "Severity Distribution Over Time",
) -> Optional[Any]:
    """Create a stacked area chart of severity levels over time."""
    if not HAS_PLOTLY:
        return None
    if not windows:
        return _empty_figure("No data available")

    labels = [w.window_label for w in windows]
    low = [w.severity_dist.low for w in windows]
    med = [w.severity_dist.medium for w in windows]
    high = [w.severity_dist.high for w in windows]
    crit = [w.severity_dist.critical for w in windows]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels, y=crit, name="Critical", mode="lines",
        stackgroup="one", line=dict(width=0.5),
        fillcolor=SEVERITY_COLORS["critical"],
        line_color=SEVERITY_COLORS["critical"],
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=high, name="High", mode="lines",
        stackgroup="one", line=dict(width=0.5),
        fillcolor=SEVERITY_COLORS["high"],
        line_color=SEVERITY_COLORS["high"],
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=med, name="Medium", mode="lines",
        stackgroup="one", line=dict(width=0.5),
        fillcolor=SEVERITY_COLORS["medium"],
        line_color=SEVERITY_COLORS["medium"],
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=low, name="Low", mode="lines",
        stackgroup="one", line=dict(width=0.5),
        fillcolor=SEVERITY_COLORS["low"],
        line_color=SEVERITY_COLORS["low"],
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis_title="Time Period",
        yaxis_title="Incident Count",
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )

    return fig


# ── Matplotlib Static Charts ──────────────────────────────────────────────────

def create_static_severity_pie(
    dist: SeverityDistribution,
    output_path: str = "severity_distribution.png",
    dpi: int = 150,
) -> Optional[str]:
    """Create a static severity pie chart using matplotlib.

    Returns:
        Path to saved PNG or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None
    if dist.total == 0:
        return None

    labels = []
    values = []
    colors = []
    for label, count, color in [
        ("Low", dist.low, SEVERITY_COLORS["low"]),
        ("Medium", dist.medium, SEVERITY_COLORS["medium"]),
        ("High", dist.high, SEVERITY_COLORS["high"]),
        ("Critical", dist.critical, SEVERITY_COLORS["critical"]),
    ]:
        if count > 0:
            labels.append(label)
            values.append(count)
            colors.append(color)

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100. * sum(values)))})",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    centre_circle = plt.Circle((0, 0), 0.35, fc="white")
    ax.add_artist(centre_circle)
    ax.text(0, 0, f"{dist.total}\nTotal", ha="center", va="center",
            fontsize=14, fontweight="bold")

    ax.set_title("Severity Distribution", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


def create_static_incident_bar(
    windows: List[TrendWindow],
    output_path: str = "incident_trend.png",
    dpi: int = 150,
) -> Optional[str]:
    """Create a static bar chart of incidents per window."""
    if not HAS_MATPLOTLIB:
        return None
    if not windows:
        return None

    labels = [w.window_label for w in windows]
    counts = [w.incident_count for w in windows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bars = ax.bar(labels, counts, color=COLORS["primary"], edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Moving average line
    if len(counts) >= 3:
        ma = _compute_moving_average(counts, 3)
        ax.plot(labels, ma, color=COLORS["accent"], linewidth=2,
                linestyle="--", marker="o", markersize=4, label="3-Period MA")
        ax.legend(loc="upper left", fontsize=9)

    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_ylabel("Number of Incidents", fontsize=11)
    ax.set_title("Plagiarism Incidents Over Time", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_moving_average(values: List[float], period: int) -> List[float]:
    """Compute a simple moving average."""
    result = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def _empty_figure(message: str) -> Optional[Any]:
    """Create a minimal Plotly figure with an info message."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS["text_light"]),
    )
    fig.update_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor=COLORS["bg"],
    )
    return fig


def render_analytics_summary_metrics(
    report: AnalyticsReport,
) -> Dict[str, Any]:
    """Extract key metrics from a report for Streamlit metric cards.

    Returns a dict suitable for st.metric() calls.
    """
    return {
        "total_incidents": report.total_incidents,
        "avg_similarity": report.statistical_summary.mean,
        "high_rate": round(report.severity_distribution.high_rate, 1),
        "repeat_offense_rate": round(report.repeat_offense_rate * 100, 1),
        "monthly_growth": round(report.monthly_growth_rate * 100, 1),
        "trend_direction": report.trend.direction.value,
        "trend_confidence": report.trend.confidence,
        "date_range_start": str(report.date_range_start.date()),
        "date_range_end": str(report.date_range_end.date()),
        "forecast_next": report.trend.forecast_values[0] if report.trend.forecast_values else 0,
    }
