"""
network_graph.py
----------------
Generates interactive document plagiarism network graphs using networkx and Plotly.
Documents are represented as nodes, and similarities above the threshold are edges.
"""

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_network_data(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    theme_colors: Optional[dict] = None,
) -> dict:
    """
    Processes similarity matrix data, constructs NetworkX graph layout, and formats node and edge traces.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold are connected.
        theme_colors: Optional dictionary containing theme colors.

    Returns:
        Dictionary containing shapes, edge_hover_trace, node_trace, graph, and pos coordinates.
    """
    # Create networkx graph
    G = nx.Graph()

    # Add all documents as nodes
    doc_names = list(similarity_df.columns)

    for name in doc_names:
        G.add_node(name)

    # Add edges for pairs exceeding threshold
    n = len(doc_names)
    edge_similarities = {}

    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_df.iloc[i, j])

            if score >= threshold:
                G.add_edge(doc_names[i], doc_names[j])

                edge_similarities[(doc_names[i], doc_names[j])] = score

    # Compute layout coordinates
    # Seed layout for reproducibility
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.0 / np.sqrt(max(1, len(G.nodes()))),
    )

    # ── Draw Edges ─────────────────────────────────────────────────────────────

    shapes = []

    # For hover info, add a transparent trace over each edge
    edge_trace_x = []
    edge_trace_y = []
    edge_hover_texts = []

    for edge in G.edges():
        doc_a, doc_b = edge

        x0, y0 = pos[doc_a]
        x1, y1 = pos[doc_b]

        # Get similarity score
        score = edge_similarities.get(
            (doc_a, doc_b),
            edge_similarities.get(
                (doc_b, doc_a),
                threshold,
            ),
        )

        # Line width based on similarity
        line_width = max(1.5, score * 6.0)

        # Color based on severity
        if score >= 0.90:
            color = (
                theme_colors.get("danger", "#ff4b4b")
                if theme_colors
                else "#ff4b4b"
            )
        elif score >= 0.75:
            color = (
                theme_colors.get("warning", "#ffa500")
                if theme_colors
                else "#ffa500"
            )
        else:
            color = (
                theme_colors.get("success", "#21c55d")
                if theme_colors
                else "#21c55d"
            )


        shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(
                    color=color,
                    width=line_width,
                ),
                layer="below",
            )
        )

        # Add to hover trace
        edge_trace_x.extend([x0, x1, None])

        edge_trace_y.extend([y0, y1, None])

        edge_hover_texts.append(
            f"<b>Match:</b> {doc_a} ↔ {doc_b}<br>" f"<b>Similarity:</b> {score:.1%}"
        )

    # ── Edge Hover Trace ──────────────────────────────────────────────────────

    edge_hover_x = []
    edge_hover_y = []

    for edge in G.edges():
        doc_a, doc_b = edge

        x0, y0 = pos[doc_a]
        x1, y1 = pos[doc_b]

        # Midpoint coordinate
        edge_hover_x.append((x0 + x1) / 2.0)

        edge_hover_y.append((y0 + y1) / 2.0)

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker=dict(
            size=8,
            color="rgba(0,0,0,0)",
        ),
        text=edge_hover_texts,
        hoverinfo="text",
        name="Connections",
    )

    # ── Draw Nodes ─────────────────────────────────────────────────────────────

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_size = []

    # Store the document ID for each Plotly node.
    # The order matches node_x, node_y, and node_text.
    node_document_ids = []

    for node in G.nodes():
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # The current graph uses document names as node identifiers.
        # These values are passed through Plotly's customdata so that
        # streamlit-plotly-events can identify the clicked document.
        node_document_ids.append(node)

        deg = G.degree(node)

        # Size based on degree
        node_size.append(20 + deg * 6)

        # Color based on degree
        if deg == 0:
            node_color.append(
                theme_colors.get(
                    "success",
                    "#2e7d32",
                )
                if theme_colors
                else "#2e7d32"
            )

        elif deg == 1:
            node_color.append(
                theme_colors.get(
                    "warning",
                    "#f9a825",
                )
                if theme_colors
                else "#f9a825"
            )

        else:
            node_color.append(
                theme_colors.get(
                    "danger",
                    "#c62828",
                )
                if theme_colors
                else "#c62828"
            )

        node_hover.append(
            f"<b>📄 Document:</b> {node}<br>"
            f"<b>🚨 Flagged connections:</b> "
            f"{deg} / {len(doc_names) - 1}"
        )

    # ── Plotly Node Trace ──────────────────────────────────────────────────────

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        # Store document ID with every Plotly node.
        # streamlit-plotly-events can retrieve this value
        # when the user clicks a node.
        customdata=node_document_ids,
        text=[name.split(".")[0] for name in node_text],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        textfont=dict(
            color=(
                theme_colors.get(
                    "ink",
                    "#0F172A",
                )
                if theme_colors
                else "#0F172A"
            ),
            size=10,
            family="Arial Black",
        ),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(
                width=2,
                color=(
                    theme_colors.get(
                        "background",
                        "#ffffff",
                    )
                    if theme_colors
                    else "#ffffff"
                ),
            ),
        ),
        name="Documents",
    )

    return {
        "shapes": shapes,
        "edge_hover_trace": edge_hover_trace,
        "node_trace": node_trace,
        "graph": G,
        "pos": pos,
    }


def render_network_plotly(
    network_data: dict,
    title: str = "Document Plagiarism Network",
    theme_colors: Optional[dict] = None,
) -> go.Figure:
    """
    Renders an interactive Plotly figure layout using preformatted graph data.

    Args:
        network_data: Dictionary containing shapes, edge_hover_trace, and node_trace.
        title: Title of the graph.
        theme_colors: Optional dictionary containing theme colors.

    Returns:
        Plotly Graph Objects Figure.
    """
    shapes = network_data.get("shapes", [])
    edge_hover_trace = network_data.get("edge_hover_trace")
    node_trace = network_data.get("node_trace")

    bg_color = (
        theme_colors.get(
            "background",
            "#FFFFFF",
        )
        if theme_colors
        else "#FFFFFF"
    )

    ink_color = (
        theme_colors.get(
            "ink",
            "#0F172A",
        )
        if theme_colors
        else "#0F172A"
    )

    traces = []
    if edge_hover_trace is not None:
        traces.append(edge_hover_trace)
    if node_trace is not None:
        traces.append(node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(
                    size=16,
                    family="Arial Black",
                ),
            ),
            showlegend=False,
            hovermode="closest",
            autosize=True,
            width=None,
            margin=dict(
                b=40,
                l=40,
                r=40,
                t=50,
            ),
            shapes=shapes,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(
                color=ink_color,
            ),
        ),
    )

    return fig


def plot_similarity_network(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    title: str = "Document Plagiarism Network",
    theme_colors: Optional[dict] = None,
) -> go.Figure:
    """
    Builds a networkx graph from the similarity matrix and returns an interactive Plotly figure.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold are connected.
        title: Title of the graph.
        theme_colors: Optional dictionary containing theme colors.

    Returns:
        Plotly Graph Objects Figure.
    """
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        theme_colors=theme_colors,
    )
    return render_network_plotly(
        network_data=network_data,
        title=title,
        theme_colors=theme_colors,
    )

