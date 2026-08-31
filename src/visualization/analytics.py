"""
analytics.py
-----------
Plotly visualizations for plagiarism analytics dashboard.
Supports dynamic light and dark mode theme switching (#1619).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.config import DEFAULT_THRESHOLDS

FigureT = TypeVar("FigureT")


def apply_plotly_theme(
    fig: go.Figure,
    theme_colors: dict[str, str] | None = None,
    show_grid: bool = True,
) -> go.Figure:
    """Apply matching light/dark theme colors (paper_bgcolor, plot_bgcolor, font_color) to a Plotly figure."""
    if not theme_colors or not isinstance(theme_colors, dict):
        return fig

    paper_bg = theme_colors.get("background", "white")
    plot_bg = theme_colors.get("surface", "white")
    font_color = theme_colors.get("ink", "#0f172a")
    grid_color = theme_colors.get("border", "#e2e8f0")

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
    )
    if show_grid:
        fig.update_xaxes(gridcolor=grid_color)
        fig.update_yaxes(gridcolor=grid_color)

    return fig


def get_chart_theme_colors(theme_mode: str) -> dict[str, str]:
    """Return a dictionary of Plotly-compatible theme colors based on the UI mode.

    This helper synchronizes Plotly chart background and font colors with the
    current Streamlit UI theme mode (Light vs Dark). It ensures that charts
    rendered in the analytics dashboard remain legible and visually consistent
    regardless of the user's selected theme.

    The returned dictionary is structured to be passed directly into the
    ``theme_colors`` parameter of :func:`apply_plotly_theme` or used manually
    in Plotly layout updates.

    Args:
        theme_mode: The current UI theme mode. Expected values are "Light"
                    or "Dark" (case-insensitive). Any other value defaults
                    to the Light theme palette.

    Returns:
        A dictionary containing the following keys:
        - ``background``: The main paper/canvas background color.
        - ``surface``: The plot area background color.
        - ``ink``: The primary text/font color.
        - ``muted``: Secondary text color for subtitles and annotations.
        - ``border``: Gridline and axis border color.
    """
    normalized_mode = (theme_mode or "light").strip().lower()

    if normalized_mode == "dark":
        return {
            "background": "#1e293b",  # Slate 800
            "surface": "#0f172a",  # Slate 900
            "ink": "#f8fafc",  # Slate 50
            "muted": "#94a3b8",  # Slate 400
            "border": "#334155",  # Slate 700
            "grid": "#475569",  # Slate 600
        }
    else:
        return {
            "background": "#ffffff",  # Pure white
            "surface": "#f8fafc",  # Slate 50
            "ink": "#0f172a",  # Slate 900
            "muted": "#64748b",  # Slate 500
            "border": "#e2e8f0",  # Slate 200
            "grid": "#cbd5e1",  # Slate 300
        }


def _create_boxplot_trace(name: str, scores: list[float], **kwargs: Any) -> go.Box:
    """Create a standardized Plotly Box trace with uniform styling."""
    return go.Box(
        y=scores,
        name=name,
        boxpoints="outliers",
        marker_color="#636efa",
        line_color="#4a4dba",
        hovertemplate="<b>%{name}</b><br>Similarity Score: %{y:.2f}<extra></extra>",
        **kwargs,
    )


def plot_similarity_boxplot_by_group(
    scores_dict: dict[str, list[float]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a box plot of similarity score quartiles, grouped by assignment."""
    if not scores_dict:
        return _empty_chart(
            title="Similarity Score Quartile Distribution",
            message="No similarity scores available to plot",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Assignment",
            yaxis_title="Similarity Score",
        )
    fig = go.Figure()
    for group_name, scores in scores_dict.items():
        fig.add_trace(_create_boxplot_trace(name=str(group_name), scores=scores))

    fig.update_layout(
        title="Similarity Score Quartile Distribution",
        xaxis_title="Assignment",
        yaxis_title="Similarity Score",
        height=400,
        showlegend=False,
        autosize=True,
    )

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid, range=[0.0, 1.0])

    _apply_theme_colors(fig, theme_colors)

    return fig


def _apply_theme_colors(
    fig: go.Figure,
    theme_colors: dict[str, str] | None,
    theme_override: str | None = None,
) -> None:
    """Apply light/dark theme colors to a Plotly figure layout."""
    if theme_override == "light":
        fig.update_layout(template="plotly_white")
    elif theme_override == "dark":
        fig.update_layout(template="plotly_dark")

    if not theme_colors or not isinstance(theme_colors, dict):
        return

    apply_plotly_theme(fig, theme_colors)


def calculate_severity_ratios(incidents: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate the percentage breakdown of High, Medium, and Low severity incidents."""
    counts = {"High": 0, "Medium": 0, "Low": 0}
    total = 0

    for incident in incidents:
        score = incident.get("similarity_score")
        if score is None:
            score = incident.get("similarity")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        total += 1
        if score >= DEFAULT_THRESHOLDS.high:
            counts["High"] += 1
        elif score >= DEFAULT_THRESHOLDS.medium:
            counts["Medium"] += 1
        else:
            counts["Low"] += 1

    if total == 0:
        return {"High": 0.0, "Medium": 0.0, "Low": 0.0}

    return {label: round((count / total) * 100, 2) for label, count in counts.items()}


def _annotation_color(theme_colors: dict[str, str] | None) -> str:
    """Pick a readable annotation color for the given theme."""
    if theme_colors and isinstance(theme_colors, dict):
        return theme_colors.get("ink", "#64748b")
    return "#64748b"


def _empty_chart(
    title: str,
    message: str,
    theme_colors: dict[str, str] | None = None,
    show_grid: bool = True,
    height: int = 400,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Build a themed placeholder figure for an empty-state chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=_annotation_color(theme_colors)),
    )

    layout_kwargs: dict[str, Any] = {"title": title, "height": height, "autosize": True}
    if xaxis_title is not None:
        layout_kwargs["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        layout_kwargs["yaxis_title"] = yaxis_title
    fig.update_layout(**layout_kwargs)

    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def build_visualization_lazily(
    enabled: bool,
    factory: Callable[[], FigureT],
) -> FigureT | None:
    """Build a visualization only after the user explicitly enables it."""
    if not enabled:
        return None

    return factory()


def get_top_similar_pairs(
    similarity_df: pd.DataFrame,
    top_n: int = 5,
) -> list[tuple[str, str, float]]:
    """Return the top-N highest similarity document pairs."""
    if similarity_df.empty or similarity_df.shape[0] < 2:
        return []

    doc_names = list(similarity_df.index)
    n = len(doc_names)

    row_indices, col_indices = np.triu_indices(n, k=1)
    sim_matrix = similarity_df.to_numpy(dtype=float)
    scores = sim_matrix[row_indices, col_indices]

    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:top_n]

    pairs: list[tuple[str, str, float]] = []
    for idx in top_indices:
        i = row_indices[idx]
        j = col_indices[idx]
        score = float(scores[idx])
        pairs.append((doc_names[i], doc_names[j], score))

    return pairs


def plot_high_severity_trends(
    trend_data: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
    theme_override: str | None = None,
) -> go.Figure:
    """Create an interactive line chart showing High severity plagiarism incidents over time."""
    if not trend_data:
        return _empty_chart(
            title="High Severity Plagiarism Trends (Last 30 Days)",
            message="No High severity incidents recorded in the specified period",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Date",
            yaxis_title="Number of High Severity Incidents",
        )
    df = pd.DataFrame(trend_data)
    df["date"] = pd.to_datetime(df["date"])
    df["cumulative"] = df["count"].cumsum()

    fig = px.line(
        df,
        x="date",
        y="count",
        title="High Severity Plagiarism Trends (Last 30 Days)",
        labels={"date": "Date", "count": "Number of High Severity Incidents"},
        markers=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative"],
            mode="lines+markers",
            name="Cumulative Incidents",
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of High Severity Incidents",
        yaxis2=dict(
            title="Cumulative Incidents",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        height=400,
        showlegend=True,
        autosize=True,
    )
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        line=dict(color="#ff4b4b", width=3), marker=dict(size=8, color="#ff4b4b")
    )
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_most_plagiarized_documents(
    doc_data: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
    theme_override: str | None = None,
    max_name_len: int = 30,
) -> go.Figure:
    """Create a bar chart showing the most frequently plagiarized documents."""
    if not doc_data:
        return _empty_chart(
            title="Most Frequently Plagiarized Documents",
            message="No plagiarism incidents recorded",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Document Name",
            yaxis_title="Number of Incidents",
        )
    df = pd.DataFrame(doc_data)
    df["display_name"] = df["document_name"].apply(
        lambda x: x[:max_name_len] + "..." if len(x) > max_name_len else x
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

    full_names = df["document_name"].tolist()
    fig.update_traces(
        customdata=full_names,
        hovertemplate="<b>%{customdata}</b><br>Incidents: %{y}<extra></extra>",
    )
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_similarity_distribution(
    sim_matrix: pd.DataFrame,
    title: str = "Distribution of Similarity Scores",
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a histogram showing the distribution of all pairwise similarity scores."""
    if sim_matrix.empty or sim_matrix.shape[0] < 2:
        return _empty_chart(
            title=title,
            message="Not enough documents to compute a similarity distribution",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Similarity Score",
            yaxis_title="Number of Document Pairs",
        )
    mask = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    scores = sim_matrix.where(mask).stack().values

    fig = px.histogram(
        scores,
        nbins=30,
        title=title,
        labels={
            "value": "Similarity Score",
            "count": "Number of Document Pairs",
        },
        range_x=[0.0, 1.0],
    )

    fig.update_layout(
        xaxis_title="Similarity Score",
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
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_document_sizes(
    word_counts: dict[str, int],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
    max_name_len: int = 30,
) -> go.Figure:
    """Create a bar chart visualizing document word counts."""
    if not word_counts:
        return _empty_chart(
            title="Document Word Counts",
            message="No documents currently in the database",
            theme_colors=theme_colors,
            show_grid=show_grid,
        )
    doc_names = list(word_counts.keys())
    counts = list(word_counts.values())

    display_names = [
        name[:max_name_len] + "..." if len(name) > max_name_len else name
        for name in doc_names
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
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_similarity_boxplot(
    incidents: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a box plot showing similarity score distributions per assignment."""
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
        return _empty_chart(
            title="Similarity Score Distribution by Assignment",
            message="No similarity scores recorded for the selected incidents",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Assignment Title",
            yaxis_title="Similarity Score",
        )
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["assignment_title"], []).append(row["similarity_score"])

    fig = go.Figure()
    for title, scores in grouped.items():
        fig.add_trace(_create_boxplot_trace(name=title, scores=scores))

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
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_severity_donut_chart(
    incidents: list[dict[str, Any]],
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a donut chart showing the distribution of plagiarism incident severities."""
    if not incidents:
        return _empty_chart(
            title="Plagiarism Incident Severity Distribution",
            message="No plagiarism incidents recorded",
            theme_colors=theme_colors,
            show_grid=False,
        )
    df = pd.DataFrame(incidents)
    if "severity" not in df.columns:
        df["severity"] = "Unknown"

    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]

    color_map = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#10b981",
    }
    colors = [color_map.get(sev, "#cccccc") for sev in counts["severity"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["severity"],
                values=counts["count"],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>Severity: %{label}</b><br>Incidents: %{value}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Plagiarism Incident Severity Distribution",
        height=400,
        showlegend=True,
        autosize=True,
    )
    return apply_plotly_theme(fig, theme_colors, show_grid=False)


def plot_similarity_histogram(
    scores: list[float],
    n_bins: int = 20,
    colorscale: str = "Viridis",
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create an interactive histogram of pairwise similarity scores with gradient coloring."""
    if not scores:
        return _empty_chart(
            title="Similarity Score Distribution",
            message="No similarity scores available to plot",
            theme_colors=theme_colors,
            show_grid=False,
        )
    counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(
        data=go.Bar(
            x=bin_centers,
            y=counts,
            marker=dict(
                color=counts,
                colorscale=colorscale,
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
    return apply_plotly_theme(fig, theme_colors, show_grid=True)


def plot_similarity_percentiles(
    similarity_scores: list[float],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a horizontal bar chart of the similarity score percentile breakdown."""
    scores: list[float] = []
    for value in similarity_scores:
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue

    if not scores:
        return _empty_chart(
            title="Similarity Score Percentile Breakdown",
            message="No similarity scores available to compute percentiles",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Similarity Score",
            yaxis_title="Percentile",
        )
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
    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_hierarchical_dendrogram(
    similarity_matrix: pd.DataFrame,
    title: str = "Hierarchical Clustering Dendrogram",
    height: int = 500,
    theme_colors: dict[str, str] | None = None,
    show_grid: bool = True,
) -> go.Figure:
    """Create an interactive hierarchical clustering dendrogram."""
    fig = go.Figure()

    if similarity_matrix is None or similarity_matrix.empty:
        return _empty_chart(
            title=title,
            message="No similarity data available to build a dendrogram",
            theme_colors=theme_colors,
            show_grid=show_grid,
            height=height,
            xaxis_title="Document",
            yaxis_title="Merge Distance (1 − similarity)",
        )

    if similarity_matrix.shape[0] < 2:
        return _empty_chart(
            title=title,
            message="At least two documents are required to build a dendrogram",
            theme_colors=theme_colors,
            show_grid=show_grid,
            height=height,
            xaxis_title="Document",
            yaxis_title="Merge Distance (1 − similarity)",
        )

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    doc_names = list(similarity_matrix.index)
    sim_values = np.clip(similarity_matrix.to_numpy(dtype=float), 0.0, 1.0)

    distance_matrix = 1.0 - sim_values
    np.fill_diagonal(distance_matrix, 0.0)

    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="ward")

    xs: list[float] = []
    ys: list[float] = []
    hover_texts: list[str] = []

    n_leaves = len(doc_names)
    cluster_x: dict[int, float] = {
        leaf_idx: float(leaf_idx) for leaf_idx in range(n_leaves)
    }

    def _cluster_members(cluster_id: int) -> list[int]:
        members = []
        stack = [cluster_id]
        while stack:
            curr_id = stack.pop()
            if curr_id < n_leaves:
                members.append(int(curr_id))
            else:
                row = linkage_matrix[curr_id - n_leaves]
                stack.append(int(row[0]))
                stack.append(int(row[1]))
        return members

    for step, row in enumerate(linkage_matrix, start=1):
        left_id = int(row[0])
        right_id = int(row[1])
        merge_distance = float(row[2])
        new_id = n_leaves + step - 1

        left_x = cluster_x[left_id]
        right_x = cluster_x[right_id]

        left_y = (
            float(linkage_matrix[left_id - n_leaves][2]) if left_id >= n_leaves else 0.0
        )
        right_y = (
            float(linkage_matrix[right_id - n_leaves][2])
            if right_id >= n_leaves
            else 0.0
        )

        xs.extend([left_x, left_x, right_x, right_x, None])
        ys.extend([left_y, merge_distance, merge_distance, right_y, None])

        left_members = _cluster_members(left_id)
        right_members = _cluster_members(right_id)
        left_names = ", ".join(doc_names[i] for i in left_members)
        right_names = ", ".join(doc_names[i] for i in right_members)
        tooltip = (
            f"<b>Merge #{step}</b><br>"
            f"Distance: {merge_distance:.3f} "
            f"(similarity: {1.0 - merge_distance:.3f})<br>"
            f"Cluster A ({len(left_members)}): {left_names}<br>"
            f"Cluster B ({len(right_members)}): {right_names}"
        )
        hover_texts.extend([tooltip, tooltip, tooltip, tooltip, ""])

        cluster_x[new_id] = (left_x + right_x) / 2.0

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#636efa", width=2),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_leaves)),
        ticktext=doc_names,
        tickangle=-45,
    )
    fig.update_yaxes(
        title="Merge Distance (1 − similarity)",
        autorange="reversed",
        range=[1.0, 0.0],
    )

    fig.update_layout(
        title=title,
        xaxis_title="Document",
        height=height,
        autosize=True,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=120, l=60, r=40, t=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=show_grid)

    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_precision_recall_curve(
    evaluations: list[dict[str, Any]],
    current_threshold: float | None = None,
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create a precision / recall / F1 calibration curve from a threshold sweep."""
    if not evaluations:
        return _empty_chart(
            title="Precision / Recall Calibration Curve",
            message="No calibration sweep data available to plot",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Similarity Threshold",
            yaxis_title="Score",
        )

    df = pd.DataFrame(evaluations)
    df = df.sort_values("threshold")

    fig = go.Figure()
    for column, name, color, line in [
        ("precision", "Precision", "#636efa", "solid"),
        ("recall", "Recall", "#00cc96", "solid"),
        ("f1", "F1", "#ef4444", "dash"),
    ]:
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["threshold"],
                y=df[column],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2, dash=line),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<br>Threshold: %{{x:.3f}}<extra></extra>",
            )
        )

    if current_threshold is not None:
        fig.add_vline(
            x=float(current_threshold),
            line_dash="dot",
            line_color="#64748b",
            annotation_text=f"Current {current_threshold:.3f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Precision / Recall Calibration Curve",
        xaxis_title="Similarity Threshold",
        yaxis_title="Score",
        height=400,
        showlegend=True,
        autosize=True,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=show_grid, range=[0.0, 1.0])
    fig.update_yaxes(showgrid=show_grid, range=[0.0, 1.0])

    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)


def plot_monthly_incident_trends(
    incidents: list[dict[str, Any]],
    show_grid: bool = True,
    theme_colors: dict[str, str] | None = None,
    months_to_show: int = 12,
) -> go.Figure:
    """Create a vertical bar chart showing monthly plagiarism incident trends."""
    if not incidents or not isinstance(incidents, list):
        return _empty_chart(
            title="Monthly Plagiarism Incident Trends",
            message="No plagiarism incidents recorded to display trends",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Month (YYYY-MM)",
            yaxis_title="Number of Incidents",
        )

    monthly_counts: dict[str, int] = {}
    valid_dates_found = False

    for incident in incidents:
        date_str = (
            incident.get("date_flagged")
            or incident.get("timestamp")
            or incident.get("created_at")
        )

        if not date_str:
            continue

        try:
            if isinstance(date_str, str):
                if "T" in date_str:
                    dt = pd.to_datetime(date_str, utc=True)
                else:
                    dt = pd.to_datetime(date_str)
            else:
                dt = pd.to_datetime(date_str)

            month_key = dt.strftime("%Y-%m")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            valid_dates_found = True

        except (ValueError, TypeError, pd.errors.ParserError):
            continue

    if not valid_dates_found or not monthly_counts:
        return _empty_chart(
            title="Monthly Plagiarism Incident Trends",
            message="No incidents with valid dates found in the dataset",
            theme_colors=theme_colors,
            show_grid=show_grid,
            xaxis_title="Month (YYYY-MM)",
            yaxis_title="Number of Incidents",
        )

    sorted_months = sorted(monthly_counts.keys())

    if len(sorted_months) > months_to_show:
        display_months = sorted_months[-months_to_show:]
    else:
        display_months = sorted_months

    start_date = pd.to_datetime(display_months[0] + "-01")
    end_date = pd.to_datetime(display_months[-1] + "-01")

    complete_range = pd.date_range(start=start_date, end=end_date, freq="MS")

    chart_data = []
    for dt in complete_range:
        month_key = dt.strftime("%Y-%m")
        count = monthly_counts.get(month_key, 0)
        chart_data.append(
            {
                "month": month_key,
                "incident_count": count,
                "display_label": dt.strftime("%b %Y"),
            }
        )

    df = pd.DataFrame(chart_data)

    bar_color = "#636efa"
    if theme_colors and isinstance(theme_colors, dict):
        bar_color = theme_colors.get("primary", "#636efa")

    fig = px.bar(
        df,
        x="display_label",
        y="incident_count",
        title="Monthly Plagiarism Incident Trends",
        labels={
            "display_label": "Month",
            "incident_count": "Number of Incidents",
        },
        orientation="v",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Incidents",
        height=450,
        showlegend=False,
        autosize=True,
        bargap=0.2,
        yaxis=dict(
            rangemode="tozero",
            dtick=1,
        ),
    )

    fig.update_xaxes(
        showgrid=show_grid,
        tickangle=-45,
    )
    fig.update_yaxes(showgrid=show_grid)

    fig.update_traces(
        marker_color=bar_color,
        marker_line_color="#4a4dba",
        marker_line_width=1.5,
        customdata=df["month"],
        hovertemplate=("<b>%{customdata}</b><br>Incidents: %{y}<br><extra></extra>"),
    )

    fig.update_traces(
        text=df["incident_count"],
        textposition="outside",
        textfont=dict(size=11),
    )

    return apply_plotly_theme(fig, theme_colors, show_grid=show_grid)
