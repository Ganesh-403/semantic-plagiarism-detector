"""
src/visualization/drift_heatmap.py
----------------------------------
Plotly visualizations for Intra-Document Style Drift.

Generates heatmaps and timeline charts highlighting authorship shifts
and change-points over the length of a document.
"""

import plotly.graph_objects as go
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_drift_heatmap(
    features_list: List[Dict[str, float]],
    changepoints: List[Dict[str, Any]],
    title: str = "Intra-Document Style Drift Heatmap",
) -> go.Figure:
    """Generate a Plotly heatmap showing stylometric feature evolution.

    Args:
        features_list: List of feature dictionaries from sliding windows.
        changepoints: List of detected change-points.
        title: Title for the Plotly figure.

    Returns:
        A configured Plotly Figure object.
    """
    if not features_list:
        fig = go.Figure()
        fig.add_annotation(text="No feature data available.", showarrow=False)
        return fig

    # Extract feature values for heatmap
    feature_keys = ["ttr", "yules_k", "sent_len_var", "avg_sent_len"]
    z_data = []
    y_labels = []

    for key in feature_keys:
        row = [f.get(key, 0.0) for f in features_list]
        z_data.append(row)
        y_labels.append(key)

    # X-axis labels (word positions)
    x_labels = [f"Word {f.get('start_word', 0)}" for f in features_list]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Feature Value"),
        )
    )

    # Add vertical lines for change-points
    for cp in changepoints:
        word_pos = cp.get("start_word", 0)
        fig.add_vline(
            x=f"Word {word_pos}",
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Drift ({cp.get('feature', '')})",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Document Position (Words)",
        yaxis_title="Stylometric Feature",
        height=500,
    )
    return fig
