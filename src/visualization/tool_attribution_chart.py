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
src/visualization/tool_attribution_chart.py
---------------------------------------------
Plotly visualizations for Paraphrase Tool Attribution.

Generates probability charts showing the likelihood of specific tool usage
based on the extracted statistical fingerprint.
"""

import logging
from typing import Any, Dict

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def generate_attribution_chart(
    attribution_result: dict[str, Any], title: str = "Paraphrase Tool Attribution"
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
    colors = ["#ef4444" if t == best_tool else "#94a3b8" for t in tools]

    fig = go.Figure(
        data=[
            go.Bar(
                x=tools,
                y=confidences,
                marker_color=colors,
                text=[f"{c:.1f}%" for c in confidences],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"{title} (Best Match: {best_tool})",
        xaxis_title="Paraphrase Tool",
        yaxis_title="Confidence Score (%)",
        yaxis=dict(range=[0, 105]),
        height=400,
    )

    return fig


# semantic-plagiarism-detector/src/visualization/tool_attribution_chart.py

from typing import Dict

import plotly.express as px
import plotly.graph_objects as go


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
        color_continuousScale="Viridis",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI", size=12),
        yaxis=dict(range=[0, 1]),
    )
    return fig
