"""
src/visualization/citation_network.py
-------------------------------------
Plotly visualizations for Citation Network Analysis.

Generates network graphs highlighting shared citation clusters between
students, making it easy to spot bibliography rings and citation laundering.
"""

import plotly.graph_objects as go
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def generate_citation_network_graph(
    doc_citations: dict[str, set[str]],
    title: str = "Citation Network Analysis"
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
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=doc_ids,
        textposition="top center",
        marker=dict(
            showscale=False,
            color='#2563eb',
            size=20,
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
    )
    return fig
