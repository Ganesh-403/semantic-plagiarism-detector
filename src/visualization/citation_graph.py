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
src/visualization/citation_graph.py
-----------------------------------
Plotly rendering utilities for bibliography citation networks.

Visualizes shared citations between documents as a bipartite network
graph, highlighting potential "citation lifting" or ghost citations.

Recent Additions (Issue #1958):
- Added plot_citation_network() to render shared bibliography overlaps.
"""

import logging
from typing import Dict, List, Optional

import networkx as nx
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def plot_citation_network(
    doc_a: str,
    doc_b: str,
    shared_citations: list[dict[str, str]],
    theme_colors: Optional[dict[str, str]] = None,
) -> go.Figure:
    """Render a bipartite network graph of shared citations.

    Creates a visual representation where Document A and Document B are
    nodes on the left and right, and shared citations are nodes in the
    middle, connected by edges.

    Args:
        doc_a: Name of the first document.
        doc_b: Name of the second document.
        shared_citations: List of shared citation dictionaries.
        theme_colors: Optional theme dictionary for dark/light mode.

    Returns:
        Plotly Figure object containing the network graph.
    """
    if not shared_citations:
        fig = go.Figure()
        fig.add_annotation(
            text="No shared citations found between these documents.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    G = nx.Graph()

    # Add document nodes
    G.add_node(doc_a, bipartite=0, type="doc")
    G.add_node(doc_b, bipartite=0, type="doc")

    # Add citation nodes and edges
    for i, cit in enumerate(shared_citations):
        cit_id = f"cit_{i}"
        G.add_node(
            cit_id, bipartite=1, type="citation", title=cit.get("title", "Unknown")
        )
        G.add_edge(doc_a, cit_id)
        G.add_edge(doc_b, cit_id)

    # Calculate layout (shell layout works well for bipartite graphs)
    pos = nx.shell_layout(
        G,
        nlist=[
            [doc_a, doc_b],
            [n for n, d in G.nodes(data=True) if d["type"] == "citation"],
        ],
    )

    bg_color = theme_colors.get("background", "#FFFFFF") if theme_colors else "#FFFFFF"
    ink_color = theme_colors.get("ink", "#0F172A") if theme_colors else "#0F172A"

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = G.nodes[node]
        if data["type"] == "doc":
            node_text.append(node)
            node_color.append("#3B82F6")  # Blue for documents
            node_size.append(30)
        else:
            title = data.get("title", "Citation")
            # Truncate long titles for display
            display_title = (title[:40] + "...") if len(title) > 40 else title
            node_text.append(display_title)
            node_color.append("#F59E0B")  # Amber for citations
            node_size.append(15)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            line_color=bg_color,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Shared Bibliography Network ({len(shared_citations)} citations)",
            titlefont=dict(size=16, color=ink_color),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
        ),
    )

    return fig
