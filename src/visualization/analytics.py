from __future__ import annotations

"""
analytics.py
-----------
Plotly visualizations for plagiarism analytics dashboard.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


FigureT = TypeVar("FigureT")


def _annotation_color(
    theme_colors: dict[str, str] | None, fallback: str = "gray"
) -> str:
    """Return a readable annotation text color for the active theme."""
    if not theme_colors:
        return fallback
    return theme_colors.get("muted", fallback)


def _apply_theme_colors(fig: go.Figure, theme_colors: dict[str, str] | None) -> None:
    """Apply light/dark theme colors to a Plotly figure layout.

    Matches the ``theme_colors`` palette produced by ``app.theme.get_colors()``
    so charts render on dark backgrounds in Dark mode. When ``theme_colors``
    is ``None`` the default Plotly template is left untouched.

    Args:
        fig: Plotly figure to style.
        theme_colors: Optional dict with ``background``, ``surface``, ``ink``,
            ``muted`` and ``border`` color keys.
    """
    if not theme_colors:
        return

    background = theme_colors.get("background", "#FFFFFF")
    surface = theme_colors.get("surface", background)
    ink = theme_colors.get("ink", "#0F172A")
    muted = theme_colors.get("muted", "#64748B")
    border = theme_colors.get("border", "#E2E8F0")

    fig.update_layout(
        paper_bgcolor=background,
        plot_bgcolor=surface,
        font=dict(color=ink),
    )
    fig.update_xaxes(
        gridcolor=border,
        tickfont=dict(color=muted),
        title_font=dict(color=ink),
    )
    fig.update_yaxes(
        gridcolor=border,
        tickfont=dict(color=muted),
        title_font=dict(color=ink),
    )



def build_visualization_lazily(
    enabled: bool,
    factory: Callable[[], FigureT],
) -> FigureT | None:
    """Build a visualization only after the user explicitly enables it.

    Streamlit evaluates the bodies of all tabs during a script rerun. Merely
    placing a chart inside a tab therefore does not defer expensive figure
    construction. This helper keeps the figure factory uncalled until the UI
    control for that visualization is enabled.

    Args:
        enabled: Whether the user requested the visualization.
        factory: Zero-argument callable that creates the figure.

    Returns:
        The created figure when enabled, otherwise ``None``.
    """
    if not enabled:
        return None

    return factory()
def get_top_similar_pairs(
    similarity_df: pd.DataFrame,
    top_n: int = 5,
) -> list[tuple[str, str, float]]:
    """
    Return the top-N highest similarity document pairs.

    Extracts only the upper triangle of the similarity matrix to avoid
    duplicate pairs and excludes self-similarity.

    Args:
        similarity_df: Square DataFrame containing pairwise similarity scores.
        top_n: Number of highest similarity pairs to return.

    Returns:
        List of tuples in the form:
        (document_a, document_b, similarity_score)
        sorted by similarity score in descending order.
    """
    if similarity_df.empty or similarity_df.shape[0] < 2:
        return []

    pairs: list[tuple[str, str, float]] = []

    doc_names = list(similarity_df.index)

    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            score = float(similarity_df.iloc[i, j])

            pairs.append(
                (
                    doc_names[i],
                    doc_names[j],
                    score,
                )
            )

    pairs.sort(key=lambda pair: pair[2], reverse=True)

    return pairs[:top_n]
def plot_high_severity_trends(
    trend_data: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Create an interactive line chart showing High severity plagiarism incidents over time.

    Args:
        trend_data: List of dicts with 'date' and 'count' keys
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object
    """
    if not trend_data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No High severity incidents recorded in the specified period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="High Severity Plagiarism Trends (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Number of High Severity Incidents",
            height=400,
            autosize=True,
        )
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    df = pd.DataFrame(trend_data)
    df["date"] = pd.to_datetime(df["date"])

    fig = px.line(
        df,
        x="date",
        y="count",
        title="High Severity Plagiarism Trends (Last 30 Days)",
        labels={"date": "Date", "count": "Number of High Severity Incidents"},
        markers=True,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of High Severity Incidents",
        hovermode="x unified",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        line=dict(color="#ff4b4b", width=3), marker=dict(size=8, color="#ff4b4b")
    )

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_most_plagiarized_documents(
    doc_data: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Create a bar chart showing the most frequently plagiarized documents.

    Args:
        doc_data: List of dicts with 'document_name' and 'incident_count' keys
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object
    """
    if not doc_data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No plagiarism incidents recorded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="Most Frequently Plagiarized Documents",
            xaxis_title="Document Name",
            yaxis_title="Number of Incidents",
            height=400,
            autosize=True,
        )
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    df = pd.DataFrame(doc_data)

    # Truncate long document names for display
    df["display_name"] = df["document_name"].apply(
        lambda x: x[:30] + "..." if len(x) > 30 else x
    )

    fig = px.bar(
        df,
        x="display_name",
        y="incident_count",
        title="Most Frequently Plagiarized Documents",
        labels={
            "display_name": "Document Name",
            "incident_count": "Number of Incidents",
        },
        orientation="v",
    )

    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Number of Incidents",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        marker_color="#ffa500",
        marker_line_color="#cc8400",
        marker_line_width=1.5,
    )

    # Add hover template with full document name
    full_names = df["document_name"].tolist()
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Incidents: %{y}<extra></extra>",
        customdata=full_names,
    )

    # Update hover to show full name
    fig.update_traces(
        hovertemplate="<b>%{customdata}</b><br>Incidents: %{y}<extra></extra>"
    )

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_similarity_distribution(
    sim_matrix: pd.DataFrame,
    title: str = "Distribution of Similarity Scores",
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Create a histogram showing the distribution of all pairwise similarity scores.

    Extracts the upper triangle (excluding the diagonal) from the symmetric
    similarity matrix and visualises the bell curve with Plotly Express.

    Args:
        sim_matrix: NxN DataFrame of pairwise similarity scores (0.0–1.0).
        title: Chart title.
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object with a histogram trace.
    """
    if sim_matrix.empty or sim_matrix.shape[0] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough documents to compute a similarity distribution",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Similarity Score Range (%)",
            yaxis_title="Number of Document Pairs",
            height=400,
            autosize=True,
        )
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    mask = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    scores = sim_matrix.where(mask).stack().values

    fig = px.histogram(
        scores,
        nbins=30,
        title=title,
        labels={"value": "Similarity Score Range (%)", "count": "Number of Document Pairs"},
        range_x=[0.0, 1.0],
    )

    fig.update_layout(
        xaxis_title="Similarity Score Range (%)",
        yaxis_title="Number of Document Pairs",
        bargap=0.05,
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        marker_color="#636efa",
        marker_line_color="#4a4dba",
        marker_line_width=1,
        hovertemplate="Score: %{x:.2f}<br>Pairs: %{y}<extra></extra>",
    )

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_document_sizes(
    word_counts: dict[str, int],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a bar chart visualizing document word counts.

    Args:
        word_counts: Dictionary mapping document names to word counts.
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object.
    """
    if not word_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No documents currently in the database",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(title="Document Word Counts", height=400, autosize=True)
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    doc_names = list(word_counts.keys())
    counts = list(word_counts.values())

    display_names = [
        name[:30] + "..." if len(name) > 30 else name for name in doc_names
    ]

    fig = px.bar(
        x=display_names,
        y=counts,
        title="Document Word Counts",
        labels={"x": "Document Name", "y": "Word Count"},
    )

    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Word Count",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        marker_color="#00cc96",
        customdata=doc_names,
        hovertemplate="<b>%{customdata}</b><br>Words: %{y}<extra></extra>",
    )

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_similarity_boxplot(
    incidents: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a box plot showing similarity score distributions per assignment.

    Renders one box (whiskers, median, quartiles, and statistical outliers)
    for every assignment title found in the incidents. Incidents without an
    assignment title or with a non-numeric similarity score are ignored.

    Args:
        incidents: List of dicts with 'assignment_title' and 'similarity_score'
            keys. A bare 'title'/'similarity' fallback is also accepted.
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object with one box trace per assignment title.
    """
    rows: list[dict[str, Any]] = []
    for incident in incidents:
        title = incident.get("assignment_title") or incident.get("title")
        score = incident.get("similarity_score")
        if score is None:
            score = incident.get("similarity")
        if title is None or score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        rows.append({"assignment_title": str(title), "similarity_score": score})

    if not rows:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity scores recorded for the selected incidents",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="Similarity Score Distribution by Assignment",
            xaxis_title="Assignment Title",
            yaxis_title="Similarity Score",
            height=400,
            autosize=True,
        )
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["assignment_title"], []).append(
            row["similarity_score"]
        )

    fig = go.Figure()
    for title, scores in grouped.items():
        fig.add_trace(
            go.Box(
                y=scores,
                name=title,
                boxpoints="outliers",
                marker_color="#636efa",
                line_color="#4a4dba",
                hovertemplate=(
                    "<b>%{name}</b><br>"
                    "Similarity Score: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Similarity Score Distribution by Assignment",
        xaxis_title="Assignment Title",
        yaxis_title="Similarity Score",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid, range=[0.0, 1.0])

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_severity_donut_chart(
    incidents: list[dict[str, Any]],
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Create a donut chart showing the distribution of plagiarism incident severities.

    Args:
        incidents: List of dicts, each representing an incident, expected to contain a 'severity' key.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object
    """
    if not incidents:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No plagiarism incidents recorded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="Plagiarism Incident Severity Distribution",
            height=400,
        )
        _apply_theme_colors(fig, theme_colors)
        return fig

    df = pd.DataFrame(incidents)
    
    # If no severity column exists, create it with a default value to prevent errors
    if "severity" not in df.columns:
        df["severity"] = "Unknown"

    # Count frequencies of each severity
    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]

    # Define the custom colors
    color_map = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#10b981"
    }

    # Map the colors ensuring that the order matches the plotted categories
    colors = [color_map.get(sev, "#cccccc") for sev in counts["severity"]]

    fig = go.Figure(data=[go.Pie(
        labels=counts["severity"],
        values=counts["count"],
        hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>Severity: %{label}</b><br>Incidents: %{value}<extra></extra>"
    )])

    fig.update_layout(
        title="Plagiarism Incident Severity Distribution",
        height=400,
        showlegend=True,
    )

    _apply_theme_colors(fig, theme_colors)

    return fig


def plot_similarity_histogram(
    scores: list[float],
    n_bins: int = 20,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Create an interactive histogram of pairwise similarity scores, with bars
    colored on a gradient based on how many pairs fall into each bin.

    Args:
        scores: List of pairwise similarity scores (0.0-1.0).
        n_bins: Number of histogram bins to split the 0.0-1.0 range into.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object with a gradient-colored bar histogram.
    """
    if not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity scores available to plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="Similarity Score Distribution",
            height=400,
            autosize=True,
        )
        _apply_theme_colors(fig, theme_colors)
        return fig

    counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(
        data=go.Bar(
            x=bin_centers,
            y=counts,
            marker=dict(
                color=counts,
                colorscale="Viridis",
                colorbar=dict(title="Pair Count"),
                line=dict(color="#4a4dba", width=1),
            ),
            hovertemplate="Score: %{x:.2f}<br>Pairs: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Number of Document Pairs",
        bargap=0.05,
        height=400,
        showlegend=False,
        autosize=True,
    )

    _apply_theme_colors(fig, theme_colors)

    return fig

def plot_similarity_percentiles(
    similarity_scores: list[float],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a horizontal bar chart of the similarity score percentile breakdown.

    Computes the 25th, 50th (median), 75th, and 90th percentiles of the given
    similarity scores with np.percentile() to summarise the overall similarity
    distribution. Non-numeric values are ignored.

    Args:
        similarity_scores: List of similarity scores (0.0–1.0).
        show_grid: Whether to show chart gridlines.
        theme_colors: Optional theme palette for light/dark backgrounds.

    Returns:
        Plotly Figure object with one horizontal bar per percentile.
    """
    scores: list[float] = []
    for value in similarity_scores:
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue

    if not scores:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity scores available to compute percentiles",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=_annotation_color(theme_colors)),
        )
        fig.update_layout(
            title="Similarity Score Percentile Breakdown",
            xaxis_title="Similarity Score",
            yaxis_title="Percentile",
            height=400,
            autosize=True,
        )
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        _apply_theme_colors(fig, theme_colors)
        return fig

    percentile_values = np.percentile(scores, [25, 50, 75, 90])
    percentile_labels = ["25th", "50th (Median)", "75th", "90th"]

    fig = px.bar(
        x=percentile_values,
        y=percentile_labels,
        orientation="h",
        title="Similarity Score Percentile Breakdown",
        labels={
            "x": "Similarity Score",
            "y": "Percentile",
        },
        range_x=[0.0, 1.0],
    )

    fig.update_layout(
        xaxis_title="Similarity Score",
        yaxis_title="Percentile",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        marker_color="#636efa",
        marker_line_color="#4a4dba",
        marker_line_width=1,
        hovertemplate="<b>%{y}</b><br>Similarity Score: %{x:.2f}<extra></extra>",
    )

    _apply_theme_colors(fig, theme_colors)

    return fig
