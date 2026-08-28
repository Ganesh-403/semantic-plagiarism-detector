# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/visualization/reliability_dashboard.py
------------------------------------------
Plotly visualizations for Inter-Rater Reliability and Calibration.

Generates heatmaps and charts to visualize reviewer agreement, bias,
and calibration weights across review committees.
"""

import logging

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def generate_reviewer_agreement_heatmap(
    reviewer_matrix: list[list[float]],
    reviewer_names: list[str],
    title: str = "Reviewer Agreement Matrix",
) -> go.Figure:
    """Generate a Plotly heatmap showing pairwise Cohen's Kappa between reviewers.

    Args:
        reviewer_matrix: N x N matrix of Kappa scores between N reviewers.
        reviewer_names: List of reviewer names/IDs.
        title: Title for the Plotly figure.

    Returns:
        A configured Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=reviewer_matrix,
            x=reviewer_names,
            y=reviewer_names,
            colorscale="RdYlGn",  # Red (low agreement) to Green (high agreement)
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Kappa Score"),
            text=[[f"{val:.2f}" for val in row] for row in reviewer_matrix],
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Reviewer",
        yaxis_title="Reviewer",
        height=600,
        width=800,
    )

    return fig


def generate_calibration_weight_chart(
    reviewer_weights: dict[str, float], title: str = "Reviewer Calibration Weights"
) -> go.Figure:
    """Generate a bar chart showing the calibration weight of each reviewer.

    Args:
        reviewer_weights: Dictionary mapping reviewer ID to their weight (0.0-1.0).
        title: Title for the Plotly figure.

    Returns:
        A configured Plotly Figure object.
    """
    names = list(reviewer_weights.keys())
    weights = list(reviewer_weights.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=weights,
                marker_color=[
                    "#ef4444" if w < 0.5 else "#f59e0b" if w < 0.8 else "#10b981"
                    for w in weights
                ],
                text=[f"{w:.2f}" for w in weights],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Reviewer",
        yaxis_title="Calibration Weight (Trust Score)",
        yaxis=dict(range=[0, 1.05]),
        height=400,
    )

    return fig
