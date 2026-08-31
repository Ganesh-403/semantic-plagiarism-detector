"""
src/visualization/reliability_dashboard.py
------------------------------------------
Plotly visualizations for Inter-Rater Reliability and Calibration.

Generates heatmaps and charts to visualize reviewer agreement, bias,
and calibration weights across review committees.
"""

import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def generate_reviewer_agreement_heatmap(
    reviewer_matrix: list[list[float]],
    reviewer_names: list[str],
    title: str = "Reviewer Agreement Matrix"
) -> go.Figure:
    """Generate a Plotly heatmap showing pairwise Cohen's Kappa between reviewers.
    
    Args:
        reviewer_matrix: N x N matrix of Kappa scores between N reviewers.
        reviewer_names: List of reviewer names/IDs.
        title: Title for the Plotly figure.
        
    Returns:
        A configured Plotly Figure object.
    """
    fig = go.Figure(data=go.Heatmap(
        z=reviewer_matrix,
        x=reviewer_names,
        y=reviewer_names,
        colorscale='RdYlGn', # Red (low agreement) to Green (high agreement)
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Kappa Score"),
        text=[[f"{val:.2f}" for val in row] for row in reviewer_matrix],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Reviewer",
        yaxis_title="Reviewer",
        height=600,
        width=800
    )
    
    return fig


def generate_calibration_weight_chart(
    reviewer_weights: dict[str, float],
    title: str = "Reviewer Calibration Weights"
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
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=weights,
            marker_color=['#ef4444' if w < 0.5 else '#f59e0b' if w < 0.8 else '#10b981' for w in weights],
            text=[f"{w:.2f}" for w in weights],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Reviewer",
        yaxis_title="Calibration Weight (Trust Score)",
        yaxis=dict(range=[0, 1.05]),
        height=400
    )
    
    return fig
