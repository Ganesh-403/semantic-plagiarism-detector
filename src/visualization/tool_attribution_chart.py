"""
src/visualization/tool_attribution_chart.py
---------------------------------------------
Plotly visualizations for Paraphrase Tool Attribution.

Generates probability charts showing the likelihood of specific tool usage
based on the extracted statistical fingerprint.
"""

import plotly.graph_objects as go
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_attribution_chart(
    attribution_result: dict[str, Any],
    title: str = "Paraphrase Tool Attribution"
) -> go.Figure:
    """Generate a Plotly bar chart showing tool attribution probabilities.
    
    Args:
        attribution_result: Dictionary from attribute_paraphrase_tool().
        title: Title for the Plotly figure.
        
    Returns:
        A configured Plotly Figure object.
    """
    scores = attribution_result.get("scores", {})
    if not scores:
        # Return empty figure if no scores
        fig = go.Figure()
        fig.add_annotation(text="No attribution data available.", showarrow=False)
        return fig
        
    tools = list(scores.keys())
    confidences = list(scores.values())
    best_tool = attribution_result.get("attributed_tool")
    
    # Highlight the best match
    colors = ['#ef4444' if t == best_tool else '#94a3b8' for t in tools]
    
    fig = go.Figure(data=[
        go.Bar(
            x=tools,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"{title} (Best Match: {best_tool})",
        xaxis_title="Paraphrase Tool",
        yaxis_title="Confidence Score (%)",
        yaxis=dict(range=[0, 105]),
        height=400
    )
    
    return fig


# semantic-plagiarism-detector/src/visualization/tool_attribution_chart.py

import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

def generate_tool_probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """
    Generates a Plotly probability chart showing the likelihood of specific paraphrasing tool usage.
    """
    tools = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = px.bar(
        x=tools,
        y=probs,
        labels={"x": "Paraphrasing Tool / Engine", "y": "Probability Likelihood"},
        title="Automated Paraphrase Tool Fingerprinting & Attribution",
        color=probs,
        color_continuousScale="Viridis"
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI", size=12),
        yaxis=dict(range=[0, 1])
    )
    return fig
