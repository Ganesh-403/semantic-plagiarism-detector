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
src/visualization/citation_network.py
-------------------------------------
Plotly visualizations for Citation Network Analysis.

Generates network graphs highlighting shared citation clusters between
students, making it easy to spot bibliography rings and citation laundering.
"""

import logging
from typing import Dict, List, Set, Tuple

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def generate_citation_network_graph(
    doc_citations: dict[str, set[str]], title: str = "Citation Network Analysis"
) -> go.Figure:
    """Generate a Plotly network graph showing shared citations.

    Args:
        doc_citations: Dictionary mapping document_id to a set of citation node keys.
        title: Title for the Plotly figure.

    Returns:
        A configured Plotly Figure object.
    """
    # Build a bipartite graph: Documents on one side, Citations on the other
    # For simplicity in Plotly, we'll just plot documents as nodes and
    # draw edges between documents that share citations.

    doc_ids = list(doc_citations.keys())
    n = len(doc_ids)

    # Compute shared citation matrix
    edges = []
    edge_weights = []

    for i in range(n):
        for j in range(i + 1, n):
            doc_a = doc_ids[i]
            doc_b = doc_ids[j]
            shared = doc_citations[doc_a].intersection(doc_citations[doc_b])
            if shared:
                edges.append((i, j))
                edge_weights.append(len(shared))

    # Simple circular layout for nodes
    import math

    node_x = []
    node_y = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        node_x.append(math.cos(angle))
        node_y.append(math.sin(angle))

    # Build edge traces
    edge_x = []
    edge_y = []
    for i, j in edges:
        edge_x.extend([node_x[i], node_x[j], None])
        edge_y.extend([node_y[i], node_y[j], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=doc_ids,
        textposition="top center",
        marker=dict(showscale=False, color="#2563eb", size=20, line_width=2),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig
