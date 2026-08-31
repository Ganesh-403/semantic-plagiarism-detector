"""
src/visualization/stylometry_radar.py
-------------------------------------
Plotly visualizations for Stylometric Authorship Attribution.

Generates radar charts comparing a submission's stylometric profile
against the student's historical baseline, highlighting deviations
that may indicate ghostwriting or AI generation.
"""

import plotly.graph_objects as go
from typing import Dict, List
import logging

from src.core.stylometry_engine import StylometricProfile

logger = logging.getLogger(__name__)


def generate_stylometry_radar_chart(
    current_profile: StylometricProfile,
    baseline_profile: StylometricProfile,
    title: str = "Stylometric Authorship Analysis",
) -> go.Figure:
    """Generate a Plotly radar chart comparing current vs baseline profiles.

    Args:
        current_profile: The stylometric profile of the new submission.
        baseline_profile: The historical baseline profile of the student.
        title: Title for the Plotly figure.

    Returns:
        A configured Plotly Figure object.
    """
    # Define the features to plot
    features = [
        "Type-Token Ratio",
        "Avg Sentence Length",
        "Sentence Variance",
        "Avg Word Length",
        "Punctuation Freq",
        "Yule's K",
    ]

    # Map feature names to dataclass attributes
    attr_map = {
        "Type-Token Ratio": "type_token_ratio",
        "Avg Sentence Length": "avg_sentence_length",
        "Sentence Variance": "sentence_length_variance",
        "Avg Word Length": "avg_word_length",
        "Punctuation Freq": "punctuation_frequency",
        "Yule's K": "yules_k",
    }

    current_vals = [getattr(current_profile, attr_map[f]) for f in features]
    baseline_vals = [getattr(baseline_profile, attr_map[f]) for f in features]

    # Normalize values for the radar chart (0 to 1 scale for visual comparison)
    # We find the max of each feature across both profiles to scale them
    max_vals = [max(c, b, 1e-6) for c, b in zip(current_vals, baseline_vals)]

    norm_current = [(c / m) for c, m in zip(current_vals, max_vals)]
    norm_baseline = [(b / m) for b, m in zip(baseline_vals, max_vals)]

    # Close the radar polygon
    features_closed = features + [features[0]]
    norm_current_closed = norm_current + [norm_current[0]]
    norm_baseline_closed = norm_baseline + [norm_baseline[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=norm_baseline_closed,
            theta=features_closed,
            fill="toself",
            name="Historical Baseline",
            line_color="#2563eb",
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=norm_current_closed,
            theta=features_closed,
            fill="toself",
            name="Current Submission",
            line_color="#ef4444",
            opacity=0.6,
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        showlegend=True,
        title=title,
        height=500,
    )

    return fig
