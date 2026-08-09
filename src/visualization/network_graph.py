"""
network_graph.py
----------------
Generates interactive document plagiarism network graphs using networkx and Plotly.
Documents are represented as nodes, and similarities above the threshold are edges.
"""

import csv
import io
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
NETWORK_GRAPH_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plagiarism_network",
        "scale": 2,
    },
}


def _parse_document_tags(tags_val: object) -> list[str]:
    """Extracts a list of normalized tag strings from string, list, set or tuple input."""
    if not tags_val:
        return []
    if isinstance(tags_val, str):
        raw_list = [
            t.strip() for t in tags_val.replace(" ", ",").split(",") if t.strip()
        ]
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
    node_scale: float = 1.0,
    theme_colors: Optional[dict] = None,
    selected_node: Optional[str] = None,
    document_tags: Optional[dict] = None,
    doc_metadata: Optional[dict] = None,
    show_isolated: bool = False,
    spring_k: float = 0.15,
    iterations: int = 50,
    repulsion: float = 1.0,
    max_label_len: int = 15,
) -> dict:
    """Processes similarity matrix data, constructs NetworkX graph layout with force-directed physics, and formats traces.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Edge threshold; pairs with similarity >= threshold are connected.
        min_degree: Minimum degree threshold; nodes with degree < min_degree are filtered out.
        theme_colors: Optional dictionary containing theme colors.
        selected_node: Optional document name to highlight.
        document_tags: Optional dictionary mapping document names to tags.
        doc_metadata: Optional dictionary mapping document names to metadata (word_count, upload_date, etc.).
        show_isolated: Whether to keep nodes with degree 0 (no similarity connections).
        spring_k: Optimal node spacing spring constant for force-directed layout (default 0.15).
        iterations: Number of force-directed spring layout simulation iterations (default 50).
        repulsion: Repulsion force multiplier factor for node positioning.
        max_label_len: Maximum length for node label text before truncation.

    Returns:
        Dictionary containing shapes, edge_hover_trace, node_trace, graph, pos coordinates,
        tag_color_map, and document_tags.
    """
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

    # Hide isolated nodes (degree 0) by default to declutter the graph
    if not show_isolated:
        G.remove_nodes_from(
            [node for node, degree in dict(G.degree()).items() if degree == 0]
        )

    # Compute force-directed layout coordinates with physics customization
    num_nodes = len(G.nodes())
    if spring_k is None or not isinstance(spring_k, (int, float)) or spring_k <= 0:
        k_val = 1.0 / np.sqrt(max(1, num_nodes))
    else:
        k_val = float(spring_k)

    try:
        iter_val = int(iterations)
        if iter_val <= 0:
            iter_val = 50
    except (TypeError, ValueError):
        iter_val = 50

    try:
        rep_val = float(repulsion)
        if rep_val <= 0:
            rep_val = 1.0
    except (TypeError, ValueError):
        rep_val = 1.0

    if rep_val != 1.0:
        k_val = k_val * rep_val

    pos = nx.spring_layout(
        G,
        seed=42,
        k=k_val,
        iterations=iter_val,
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
        is_highlighted_edge = selected_node is not None and (
            doc_a == selected_node or doc_b == selected_node
        )

        if is_highlighted_edge:
            line_width = max(line_width * 1.8, 5.0)
            color = "#FFD700"
        elif score >= 0.90:
            color = theme_colors.get("danger", "#ff4b4b") if theme_colors else "#ff4b4b"
        elif score >= 0.75:
            color = (
                theme_colors.get("warning", "#ffa500") if theme_colors else "#ffa500"
            )
        else:
            color = (
                theme_colors.get("success", "#21c55d") if theme_colors else "#21c55d"
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

    # ── Community Clustering ───────────────────────────────────────────────────
    community_map = {}
    if len(G.nodes()) > 0:
        try:
            from networkx.algorithms import community as nx_community

            if hasattr(nx_community, "louvain_communities"):
                communities = nx_community.louvain_communities(G, seed=42)
            else:
                communities = nx_community.greedy_modularity_communities(G)
        except Exception:
            communities = [set(G.nodes())]

        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i

    # ── Plagiarism Cluster Detection (Issue #1675) ───────────────────────────────
    # Use connected components to identify collusion rings
    connected_components = list(nx.connected_components(G))
    cluster_map = {}
    for cluster_id, component in enumerate(connected_components):
        for node in component:
            cluster_map[node] = cluster_id

    # ── Draw Nodes ─────────────────────────────────────────────────────────────

    node_x = []
    node_y = []
    node_labels = []
    node_hover = []
    node_color = []
    node_size = []
    node_document_ids = []

    for node in G.nodes():
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        # Truncate label text if it exceeds max_label_len
        base_label = node.split(".")[0]
        if len(base_label) > max_label_len:
            truncated = (
                base_label[: max_label_len - 3] + "..."
                if max_label_len > 3
                else base_label[:max_label_len]
            )
        else:
            truncated = base_label
        node_labels.append(truncated)

        node_document_ids.append(node)

        deg = G.degree(node)
        base_size = (20 + deg * 6) * node_scale

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

        if selected_node is not None and node == selected_node:
            node_size.append(base_size + 15)
            node_color.append("#FFFF00")  # Bright yellow for highlighted node
        else:
            node_size.append(base_size)
            comm_idx = community_map.get(node, 0)
            node_color.append(DEFAULT_TAG_COLORS[comm_idx % len(DEFAULT_TAG_COLORS)])

        # Determine cluster size for suspicion indicator
        cluster_id = cluster_map.get(node, -1)
        cluster_size = len([n for n, cid in cluster_map.items() if cid == cluster_id])
        suspicion_badge = "🚨 COLLUSION RISK" if cluster_size >= 3 else "✅ Normal"

        meta = (
            doc_metadata.get(node, {}) if doc_metadata and node in doc_metadata else {}
        )
        word_count = meta.get("word_count", "N/A")
        upload_date = meta.get("upload_date", meta.get("created_at", "N/A"))

        node_hover.append(
            f"<b>📄 Document Title:</b> {node}<br>"
            f"<b>🔗 Cluster ID:</b> {cluster_id} ({cluster_size} docs)<br>"
            f"<b>🚨 Status:</b> {suspicion_badge}<br>"
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
        customdata=node_document_ids,
        text=node_labels,
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
        "cluster_map": cluster_map,
    }


def render_network_plotly(
    network_data: dict,
    title: str = "Document Plagiarism Network",
    theme_colors: Optional[dict] = None,
    font_scale: float = 1.0,  # <-- Added parameter
) -> go.Figure:
    """
    Renders an interactive Plotly figure layout using preformatted graph data.

    Args:
        network_data: Dictionary containing shapes, edge_hover_trace, and node_trace.
        title: Title of the graph.
        theme_colors: Optional dictionary containing theme colors.
        font_scale: Global multiplier for all font sizes to support accessibility
                    scaling. Defaults to 1.0. Minimum enforced value is 0.5.

    Returns:
        Plotly Graph Objects Figure.
    """
    # Enforce minimum font scale to prevent unreadable text
    scale = max(0.5, float(font_scale))

    # Base font sizes before scaling
    base_title_size = 16
    base_node_text_size = 10
    base_hover_size = 12

    shapes = network_data.get("shapes", [])
    edge_hover_trace = network_data.get("edge_hover_trace")
    node_trace = network_data.get("node_trace")

    bg_color = theme_colors.get("background", "#FFFFFF") if theme_colors else "#FFFFFF"

    ink_color = theme_colors.get("ink", "#0F172A") if theme_colors else "#0F172A"

    traces = []
    if edge_hover_trace is not None:
        traces.append(edge_hover_trace)
    if node_trace is not None:
        # Apply font scaling to node text
        if hasattr(node_trace, "textfont") and node_trace.textfont:
            node_trace.textfont.size = int(base_node_text_size * scale)
        traces.append(node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(
                    size=int(base_title_size * scale),
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
                t=int(50 * scale),
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
                size=int(base_hover_size * scale),
            ),
        ),
    )

    return fig


def calculate_force_directed_layout(
    graph: nx.Graph,
    spring_k: float = 0.15,
    iterations: int = 50,
    repulsion: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Calculate 2D node coordinates using custom force-directed spring layout physics."""
    num_nodes = len(graph.nodes())
    if spring_k is None or not isinstance(spring_k, (int, float)) or spring_k <= 0:
        k_val = 1.0 / np.sqrt(max(1, num_nodes))
    else:
        k_val = float(spring_k)

    try:
        iter_val = int(iterations)
        if iter_val <= 0:
            iter_val = 50
    except (TypeError, ValueError):
        iter_val = 50

    try:
        rep_val = float(repulsion)
        if rep_val <= 0:
            rep_val = 1.0
    except (TypeError, ValueError):
        rep_val = 1.0

    if rep_val != 1.0:
        k_val = k_val * rep_val

    return nx.spring_layout(
        graph,
        seed=seed,
        k=k_val,
        iterations=iter_val,
    )


def plot_similarity_network(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
    title: str = "Document Plagiarism Network",
    node_scale: float = 1.0,
    theme_colors: Optional[dict] = None,
    selected_node: Optional[str] = None,
    show_isolated: bool = False,
    spring_k: float = 0.15,
    iterations: int = 50,
    repulsion: float = 1.0,
    max_label_len: int = 15,
    font_scale: float = 1.0,
) -> go.Figure:
    """Builds a NetworkX graph from the similarity matrix and returns an interactive Plotly figure."""
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
        node_scale=node_scale,
        theme_colors=theme_colors,
        selected_node=selected_node,
        document_tags=None,
        doc_metadata=None,
        show_isolated=show_isolated,
        spring_k=spring_k,
        iterations=iterations,
        repulsion=repulsion,
        max_label_len=max_label_len,
    )
    return render_network_plotly(
        network_data=network_data,
        title=title,
        theme_colors=theme_colors,
        font_scale=font_scale,
    )


def plot_plagiarism_network_graph(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
    title: str = "Document Plagiarism Network",
    node_scale: float = 1.0,
    theme_colors: Optional[dict] = None,
    selected_node: Optional[str] = None,
    show_isolated: bool = False,
    spring_k: float = 0.15,
    iterations: int = 50,
    repulsion: float = 1.0,
    max_label_len: int = 15,
) -> go.Figure:
    """Renders an interactive force-directed plagiarism network graph with custom physics controls and label truncation."""
    return plot_similarity_network(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
        title=title,
        node_scale=node_scale,
        theme_colors=theme_colors,
        selected_node=selected_node,
        show_isolated=show_isolated,
        spring_k=spring_k,
        iterations=iterations,
        repulsion=repulsion,
        max_label_len=max_label_len,
    )


def export_graph_to_gexf(graph: nx.Graph) -> str:
    """Serialize a NetworkX graph to GEXF XML format string."""
    return "".join(nx.generate_gexf(graph))


def export_network_to_gexf_bytes(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
) -> bytes:
    """Build a network from the similarity matrix and export as GEXF bytes."""
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
        document_tags=None,
        doc_metadata=None,
        show_isolated=True,
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


def export_graph_to_csv(
    graph: nx.Graph,
    similarity_df: Optional[pd.DataFrame] = None,
) -> str:
    """Serialize NetworkX graph edges into CSV format string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Source", "Target", "Similarity"])

    name_to_idx = {}
    if similarity_df is not None and not similarity_df.empty:
        doc_names = list(similarity_df.columns)
        name_to_idx = {name: i for i, name in enumerate(doc_names)}

    for u, v, data in graph.edges(data=True):
        if similarity_df is not None and u in name_to_idx and v in name_to_idx:
            i = name_to_idx[u]
            j = name_to_idx[v]
            score = float(similarity_df.iloc[i, j])
        elif "similarity" in data:
            score = float(data["similarity"])
        else:
            score = 0.0
        writer.writerow([u, v, score])

    return output.getvalue()


def export_network_adjacency_csv(graph: nx.Graph) -> str:
    """Export a NetworkX graph as an adjacency list CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Source", "Target", "Weight"])

    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        writer.writerow([source, target, weight])

    return output.getvalue()


def export_network_to_csv_bytes(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    min_degree: int = 0,
) -> bytes:
    """Build a network from the similarity matrix and export as CSV edge list bytes."""
    network_data = build_network_data(
        similarity_df=similarity_df,
        threshold=threshold,
        min_degree=min_degree,
        document_tags=None,
        doc_metadata=None,
        show_isolated=True,
    )
    G = network_data["graph"]
    csv_str = export_graph_to_csv(G, similarity_df=similarity_df)
    return csv_str.encode("utf-8")


def export_network_centrality_csv(graph: nx.Graph) -> str:
    """
    Calculate node degree centrality using NetworkX and export as a CSV string
    formatted with headers: Document_Name,Degree,Centrality_Score.
    """
    import csv
    import io

    degrees = dict(graph.degree())
    centralities = nx.degree_centrality(graph)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Document_Name", "Degree", "Centrality_Score"])

    for node in graph.nodes():
        deg = degrees.get(node, 0)
        score = centralities.get(node, 0.0)
        writer.writerow([node, deg, score])

    return output.getvalue()


import logging

import networkx as nx

logger = logging.getLogger(__name__)


def get_cluster_count(graph: nx.Graph) -> int:
    """Calculate the total number of isolated clusters (connected components) in a plagiarism network.

    In the context of plagiarism detection, a connected component represents
    a group of documents that are linked by similarity edges exceeding the
    configured threshold. A high number of isolated clusters might indicate
    multiple independent collusion rings or distinct source materials being
    shared among different student groups.

    This helper function provides a quick integer summary of the network's
    fragmentation, which is useful for dashboard metrics and automated alerts.

    Args:
        graph: A NetworkX Graph object representing the document similarity network.
               Nodes should represent documents and edges represent similarity links.

    Returns:
        The integer count of connected components in the graph.
        Returns 0 if the graph is None, invalid, or contains no nodes.

    Examples:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edges_from([("doc_A", "doc_B"), ("doc_C", "doc_D")])
        >>> get_cluster_count(G)
        2

        >>> empty_G = nx.Graph()
        >>> get_cluster_count(empty_G)
        0
    """
    # Validate input type to prevent runtime crashes from malformed pipeline data
    if graph is None or not isinstance(graph, nx.Graph):
        logger.warning(
            "get_cluster_count: Invalid or None graph provided. Expected nx.Graph."
        )
        return 0

    # An empty graph has 0 connected components
    if len(graph.nodes()) == 0:
        logger.debug("get_cluster_count: Graph contains no nodes.")
        return 0

    try:
        # nx.number_connected_components is highly optimized in C and runs in O(V+E)
        component_count = nx.number_connected_components(graph)

        logger.debug(
            "get_cluster_count: Found %d connected components in graph with %d nodes.",
            component_count,
            len(graph.nodes()),
        )

        return int(component_count)

    except Exception as exc:
        # Catch any unexpected NetworkX errors (e.g., memory issues on massive graphs)
        # and return 0 rather than crashing the dashboard rendering pipeline.
        logger.error(
            "get_cluster_count: Failed to compute connected components: %s",
            exc,
            exc_info=True,
        )
        return 0
