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

def plot_high_severity_trends(trend_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create an interactive line chart showing High severity plagiarism incidents over time.

    Args:
        trend_data: List of dicts with 'date' and 'count' keys

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
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="High Severity Plagiarism Trends (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Number of High Severity Incidents",
            height=400,
        )
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
    )

    fig.update_traces(
        line=dict(color="#ff4b4b", width=3), marker=dict(size=8, color="#ff4b4b")
    )

    return fig


def plot_most_plagiarized_documents(doc_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create a bar chart showing the most frequently plagiarized documents.

    Args:
        doc_data: List of dicts with 'document_name' and 'incident_count' keys

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
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Most Frequently Plagiarized Documents",
            xaxis_title="Document Name",
            yaxis_title="Number of Incidents",
            height=400,
        )
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
    )

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

    return fig


def plot_similarity_distribution(sim_matrix: pd.DataFrame, title: str = "Distribution of Similarity Scores") -> go.Figure:
    """
    Create a histogram showing the distribution of all pairwise similarity scores.

    Extracts the upper triangle (excluding the diagonal) from the symmetric
    similarity matrix and visualises the bell curve with Plotly Express.

    Args:
        sim_matrix: NxN DataFrame of pairwise similarity scores (0.0–1.0).
        title: Chart title.

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
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(title=title, height=400)
        return fig

    mask = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    scores = sim_matrix.where(mask).stack().values

    fig = px.histogram(
        scores,
        nbins=30,
        title=title,
        labels={"value": "Similarity Score", "count": "Number of Pairs"},
        range_x=[0.0, 1.0],
    )

    fig.update_layout(
        xaxis_title="Similarity Score",
        yaxis_title="Number of Document Pairs",
        bargap=0.05,
        height=400,
        showlegend=False,
    )

    fig.update_traces(
        marker_color="#636efa",
        marker_line_color="#4a4dba",
        marker_line_width=1,
        hovertemplate="Score: %{x:.2f}<br>Pairs: %{y}<extra></extra>",
    )

    return fig


def plot_document_sizes(word_counts: dict[str, int]) -> go.Figure:
    """Create a bar chart visualizing document word counts.

    Args:
        word_counts: Dictionary mapping document names to word counts.

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
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(title="Document Word Counts", height=400)
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
    )

    fig.update_traces(
        marker_color="#00cc96",
        customdata=doc_names,
        hovertemplate="<b>%{customdata}</b><br>Words: %{y}<extra></extra>",
    )

    return fig

