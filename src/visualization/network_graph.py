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


DEFAULT_TAG_COLORS = [
    "#3B82F6",  # Blue
    "#10B981",  # Emerald / Green
    "#F59E0B",  # Amber / Yellow
    "#EF4444",  # Red
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#F97316",  # Orange
    "#6366F1",  # Indigo
    "#14B8A6",  # Teal
]


def _parse_document_tags(tags_val: object) -> list[str]:
    """Extracts a list of normalized tag strings from string, list, set or tuple input."""
    if not tags_val:
        return []
    if isinstance(tags_val, str):
        raw_list = [t.strip() for t in tags_val.replace(" ", ",").split(",") if t.strip()]
    elif isinstance(tags_val, (list, set, tuple)):
        raw_list = [str(t).strip() for t in tags_val if str(t).strip()]
    else:
        return []

    normalized = []
    for tag in raw_list:
        clean = tag.lower()
        if not clean.startswith("#"):
            clean = "#" + clean
        if clean not in normalized:
            normalized.append(clean)
    return normalized


def _extract_primary_tag(tags_val: object) -> Optional[str]:
    """
    Extracts the primary/class tag from a document's tags.
    Prefers tags matching '#class...' (case-insensitive), otherwise returns the first tag.
    """
    tags = _parse_document_tags(tags_val)
    if not tags:
        return None
    for tag in tags:
        if tag.lower().startswith("#class"):
            return tag
    return tags[0]


def _build_tag_color_map(tags_list: list[str]) -> dict[str, str]:
    """Maps a sorted list of unique tag names to discrete colors from the palette."""
    unique_tags = sorted(list(set(tags_list)))
    color_map = {}
    for i, tag in enumerate(unique_tags):
        color_map[tag] = DEFAULT_TAG_COLORS[i % len(DEFAULT_TAG_COLORS)]
    return color_map


def build_network_data(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
    theme_colors: Optional[dict] = None,
    highlighted_doc: Optional[str] = None,
    document_tags: Optional[dict] = None,
    doc_metadata: Optional[dict] = None,
) -> dict:
    """
    Processes similarity matrix data, constructs NetworkX graph layout, and formats node and edge traces.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold are connected.
        min_degree: Minimum degree threshold; nodes with degree < min_degree are filtered out.
        theme_colors: Optional dictionary containing theme colors.
        highlighted_doc: Optional document name to search/highlight with larger size and bright yellow color.

    Returns:
        Dictionary containing shapes, edge_hover_trace, node_trace, graph, pos coordinates,
        tag_color_map, and document_tags.
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
    # Dynamic iteration depth for responsive performance on large node counts
    num_nodes = len(G.nodes())
    layout_iterations = 10 if num_nodes >= 100 else 25
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.0 / np.sqrt(max(1, num_nodes)),
        iterations=layout_iterations,
    )

    # If document_tags is None, attempt to fetch from DB if available
    if document_tags is None:
        try:
            from src.db.corpus_db import get_document_tags

            fetched_tags = {}
            for name in doc_names:
                t = get_document_tags(name)
                if t:
                    fetched_tags[name] = t
            if fetched_tags:
                document_tags = fetched_tags
        except Exception:
            pass

    # Extract primary tags for each document node and build color map
    node_primary_tags = {}
    all_tags = []
    if document_tags and isinstance(document_tags, dict):
        for node in doc_names:
            raw_tags = document_tags.get(node)
            primary_tag = _extract_primary_tag(raw_tags)
            if primary_tag:
                node_primary_tags[node] = primary_tag
                all_tags.append(primary_tag)

    tag_color_map = _build_tag_color_map(all_tags) if all_tags else {}

    # ── Draw Edges ─────────────────────────────────────────────────────────────

    shapes = []
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_texts = []

    for doc_a, doc_b in G.edges():
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

        # Check if edge is connected to highlighted document
        is_highlighted_edge = (
            highlighted_doc is not None
            and (doc_a == highlighted_doc or doc_b == highlighted_doc)
        )

        if is_highlighted_edge:
            line_width = max(line_width * 1.8, 5.0)
            color = "#FFD700"
        elif score >= 0.90:
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

        # Midpoint coordinate for hover info
        edge_hover_x.append((x0 + x1) / 2.0)
        edge_hover_y.append((y0 + y1) / 2.0)
        edge_hover_texts.append(
            f"<b>Match:</b> {doc_a} ↔ {doc_b}<br>" f"<b>Similarity:</b> {score:.1%}"
        )

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
        base_size = 20 + deg * 6

        # Calculate top match from similarity matrix
        top_match_str = "N/A"
        max_score = 0.0
        if node in similarity_df.index and node in similarity_df.columns:
            sim_series = similarity_df.loc[node].drop(labels=[node], errors="ignore")
            if not sim_series.empty:
                max_score = float(sim_series.max())
                if max_score > 0:
                    top_doc = sim_series.idxmax()
                    top_match_str = f"{top_doc} ({max_score:.1%})"

        if highlighted_doc is not None and node == highlighted_doc:
            node_size.append(base_size + 15)
            node_color.append("#FFFF00")  # Bright yellow for highlighted node
        else:
            node_size.append(base_size)
            # Color based on maximum similarity score (plagiarism severity tier) instead of degree
            if max_score >= 0.90:
                node_color.append(
                    theme_colors.get("danger", "#c62828")
                    if theme_colors
                    else "#c62828"
                )
            elif max_score >= 0.75:
                node_color.append(
                    theme_colors.get("warning", "#f9a825")
                    if theme_colors
                    else "#f9a825"
                )
            else:
                node_color.append(
                    theme_colors.get("success", "#2e7d32")
                    if theme_colors
                    else "#2e7d32"
                )

        meta = doc_metadata.get(node, {}) if doc_metadata and node in doc_metadata else {}
        word_count = meta.get("word_count", "N/A")
        upload_date = meta.get("upload_date", meta.get("created_at", "N/A"))

        node_hover.append(
            f"<b>📄 Document Title:</b> {node}<br>"
            f"<b>🚨 Flagged connections:</b> {deg} / {max(1, len(doc_names) - 1)}<br>"
            f"<b>📝 Word Count:</b> {word_count}<br>"
            f"<b>📅 Upload Date:</b> {upload_date}<br>"
            f"<b>🔗 Top Match:</b> {top_match_str}"
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
        "tag_color_map": tag_color_map,
        "document_tags": document_tags,
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
    min_degree: int = 0,
    title: str = "Document Plagiarism Network",
    theme_colors: Optional[dict] = None,
    highlighted_doc: Optional[str] = None,
) -> go.Figure:
    """
    Builds a networkx graph from the similarity matrix and returns an interactive Plotly figure.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold are connected.
        min_degree: Minimum degree threshold; nodes with degree < min_degree are filtered out.
        title: Title of the graph.
        theme_colors: Optional dictionary containing theme colors.
        highlighted_doc: Optional document name to search/highlight with larger size and bright yellow color.

    Returns:
        Plotly Graph Objects Figure.
    """
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
        theme_colors=theme_colors,
        highlighted_doc=highlighted_doc,
    )
    return render_network_plotly(
        network_data=network_data,
        title=title,
        theme_colors=theme_colors,
    )


def export_graph_to_gexf(graph: nx.Graph) -> str:
    """
    Serialize a NetworkX graph to GEXF XML format string.

    GEXF (Graph Exchange XML Format) is the standard format supported by
    Gephi, Sigma.js, and other graph visualization tools.

    Args:
        graph: NetworkX Graph object.

    Returns:
        GEXF XML string.
    """
    return "".join(nx.generate_gexf(graph))


def export_network_to_gexf_bytes(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
) -> bytes:
    """
    Build a network from the similarity matrix and export as GEXF bytes.

    GEXF (Graph Exchange XML Format) is supported by Gephi, Sigma.js,
    and other graph visualization tools. Similarity scores are attached
    as edge attributes for visualization in Gephi.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold
            are connected.
        min_degree: Minimum degree threshold; nodes with degree below
            this value are excluded.

    Returns:
        GEXF XML as UTF-8 encoded bytes, ready for download.
    """
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
    )
    G = network_data["graph"]

    doc_names = list(similarity_df.columns)
    name_to_idx = {name: i for i, name in enumerate(doc_names)}

    for u, v in G.edges():
        i = name_to_idx[u]
        j = name_to_idx[v]
        G[u][v]["similarity"] = float(similarity_df.iloc[i, j])

    gexf_str = export_graph_to_gexf(G)
    return gexf_str.encode("utf-8")


