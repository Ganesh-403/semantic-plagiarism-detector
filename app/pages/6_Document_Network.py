"""
app/pages/6_Document_Network.py
-------------------------------
Streamlit multi-page app: Document Network Graph.

Visualizes plagiarism relationships between documents as an interactive
network graph with cluster analysis, relationship strength mapping, and
community detection for identifying plagiarism rings.
"""

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Document Network - Plagiarism Detector",
    page_icon="🕸️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Graph data generation
# ---------------------------------------------------------------------------

def _generate_network_data() -> dict[str, Any]:
    """Generate mock document network data with nodes and edges."""
    import random
    random.seed(77)

    authors = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eva Martinez",
        "Frank Lee", "Grace Kim", "Henry Wilson", "Iris Chen", "Jack Davis",
        "Karen Patel", "Leo Nguyen", "Mia Thompson", "Noah Garcia", "Olivia Davis",
    ]
    departments = [
        "Computer Science", "Electrical Engineering", "Physics", "Mathematics",
        "Biology", "Chemistry", "Data Science", "Mechanical Engineering",
    ]
    doc_types = ["Thesis", "Paper", "Report", "Assignment", "Essay", "Review", "Proposal", "Dissertation"]

    nodes = []
    for i in range(30):
        author = random.choice(authors)
        dept = random.choice(departments)
        doc_type = random.choice(doc_types)
        nodes.append({
            "id": f"DOC-{300 + i}",
            "label": f"{doc_type}_{author.split()[0]}_{random.randint(1,20)}",
            "author": author,
            "department": dept,
            "word_count": random.randint(800, 12000),
            "uploaded": (datetime.now() - timedelta(days=random.randint(0, 120))).strftime("%Y-%m-%d"),
            "similarity_score": round(random.uniform(10, 98), 1),
            "cluster": -1,  # will be assigned
            "x": random.uniform(50, 750),
            "y": random.uniform(50, 450),
        })

    # Assign clusters (community detection simulation)
    author_clusters = {}
    cluster_id = 0
    for n in nodes:
        if n["author"] not in author_clusters:
            if random.random() < 0.4:
                author_clusters[n["author"]] = cluster_id
                cluster_id += 1
            else:
                author_clusters[n["author"]] = random.randint(0, max(cluster_id - 1, 0))
        n["cluster"] = author_clusters[n["author"]]

    # Generate edges (plagiarism relationships)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            # Same author documents have higher chance of connection
            if a["author"] == b["author"]:
                sim = random.uniform(60, 95)
            elif a["department"] == b["department"]:
                sim = random.uniform(20, 70)
            else:
                sim = random.uniform(5, 40)

            if sim > 35 or (a["author"] == b["author"] and sim > 50):
                edges.append({
                    "source": a["id"],
                    "target": b["id"],
                    "similarity": round(sim, 1),
                    "type": "self-plagiarism" if a["author"] == b["author"] else "cross-document",
                    "matched_segments": random.randint(2, 25),
                    "shared_phrases": random.randint(1, 50),
                })

    # Position nodes by cluster using force-directed approximation
    cluster_positions = {}
    for n in nodes:
        c = n["cluster"]
        if c not in cluster_positions:
            cluster_positions[c] = {"x_sum": 0, "y_sum": 0, "count": 0}
        cluster_positions[c]["x_sum"] += n["x"]
        cluster_positions[c]["y_sum"] += n["y"]
        cluster_positions[c]["count"] += 1

    cluster_centers = {}
    for c, pos in cluster_positions.items():
        cluster_centers[c] = (pos["x_sum"] / pos["count"], pos["y_sum"] / pos["count"])

    # Reposition nodes toward cluster centers with some spread
    for n in nodes:
        cx, cy = cluster_centers[n["cluster"]]
        n["x"] = cx + random.uniform(-120, 120)
        n["y"] = cy + random.uniform(-80, 80)

    return {"nodes": nodes, "edges": edges, "num_clusters": len(cluster_centers)}


# ---------------------------------------------------------------------------
# SVG Network Renderer
# ---------------------------------------------------------------------------

def _render_network_svg(nodes: list[dict], edges: list[dict], selected_node: str | None = None,
                        min_sim: float = 0, show_labels: bool = True,
                        highlight_cluster: int | None = None) -> str:
    """Render the network graph as an SVG element."""
    width, height = 800, 500
    cluster_colors = [
        "#4a90d9", "#e83e8c", "#28a745", "#fd7e14", "#6f42c1",
        "#20c997", "#dc3545", "#ffc107", "#17a2b8", "#6610f2",
        "#e83e8c", "#343a40",
    ]

    svg_parts = [
        f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0d1117;border-radius:8px">',
    ]

    # Build node lookup
    node_map = {n["id"]: n for n in nodes}

    # Filter edges by similarity
    visible_edges = [e for e in edges if e["similarity"] >= min_sim]

    # Draw edges
    for e in visible_edges:
        src = node_map.get(e["source"])
        tgt = node_map.get(e["target"])
        if not src or not tgt:
            continue

        # Skip if highlight cluster is set and neither node is in it
        if highlight_cluster is not None:
            if src["cluster"] != highlight_cluster and tgt["cluster"] != highlight_cluster:
                continue

        sim = e["similarity"]
        opacity = 0.2 + (sim / 100) * 0.6
        stroke_w = 0.5 + (sim / 100) * 3
        color = "#ff4444" if sim > 80 else "#ff8844" if sim > 60 else "#888888"

        # Highlight selected node edges
        if selected_node and (e["source"] == selected_node or e["target"] == selected_node):
            opacity = 0.9
            stroke_w += 1
            color = "#ffffff"

        svg_parts.append(
            f'<line x1="{src["x"]:.0f}" y1="{src["y"]:.0f}" x2="{tgt["x"]:.0f}" y2="{tgt["y"]:.0f}" '
            f'stroke="{color}" stroke-width="{stroke_w:.1f}" stroke-opacity="{opacity:.2f}"/>'
        )

    # Draw nodes
    for n in nodes:
        if highlight_cluster is not None and n["cluster"] != highlight_cluster:
            opacity = 0.15
        else:
            opacity = 1.0

        color = cluster_colors[n["cluster"] % len(cluster_colors)]
        is_selected = n["id"] == selected_node
        r = 12 if is_selected else 8
        stroke = "white" if is_selected else "none"
        stroke_w = 3 if is_selected else 0

        svg_parts.append(
            f'<circle cx="{n["x"]:.0f}" cy="{n["y"]:.0f}" r="{r}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_w}" '
            f'opacity="{opacity:.2f}" style="cursor:pointer"/>'
        )

        if show_labels and opacity > 0.5:
            svg_parts.append(
                f'<text x="{n["x"]:.0f}" y="{n["y"] - 14:.0f}" text-anchor="middle" '
                f'fill="white" font-size="9" opacity="0.8">{n["label"][:18]}</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_network_overview(data: dict):
    """Render network overview KPIs."""
    st.subheader("🕸️ Network Overview")
    nodes = data["nodes"]
    edges = data["edges"]
    num_clusters = data["num_clusters"]

    total_words = sum(n["word_count"] for n in nodes)
    avg_sim = sum(e["similarity"] for e in edges) / len(edges) if edges else 0
    high_sim = sum(1 for e in edges if e["similarity"] > 80)
    self_plag = sum(1 for e in edges if e["type"] == "self-plagiarism")
    unique_authors = len(set(n["author"] for n in nodes))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Documents", len(nodes))
    c2.metric("Relationships", len(edges))
    c3.metric("Clusters", num_clusters)
    c4.metric("Unique Authors", unique_authors)
    c5.metric("Avg Similarity", f"{avg_sim:.1f}%")
    c6.metric("High Risk Links", high_sim)

    st.markdown(
        f"The document network contains **{len(nodes)} documents** across **{unique_authors} authors** "
        f"with **{len(edges)} plagiarism relationships** forming **{num_clusters} clusters**. "
        f"**{high_sim} links** exceed 80% similarity (high risk) and **{self_plag}** are self-plagiarism."
    )
    if self_plag > 0:
        st.warning(f"⚠️ **{self_plag} self-plagiarism links** detected — authors reusing their own work without attribution.")


def _render_cluster_analysis(data: dict):
    """Render cluster / community analysis."""
    st.subheader("🏘️ Cluster Analysis")
    nodes = data["nodes"]
    edges = data["edges"]

    # Group nodes by cluster
    clusters: dict[int, list[dict]] = {}
    for n in nodes:
        clusters.setdefault(n["cluster"], []).append(n)

    cluster_colors = [
        "#4a90d9", "#e83e8c", "#28a745", "#fd7e14", "#6f42c1",
        "#20c997", "#dc3545", "#ffc107", "#17a2b8", "#6610f2",
    ]

    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        color = cluster_colors[cid % len(cluster_colors)]
        authors = list(set(m["author"] for m in members))
        avg_sim = sum(m["similarity_score"] for m in members) / len(members) if members else 0

        # Count internal edges
        internal = sum(1 for e in edges if e["source"].startswith("DOC-") and e["target"].startswith("DOC-")
                       and any(n["id"] == e["source"] for n in members)
                       and any(n["id"] == e["target"] for n in members))

        with st.expander(f"**Cluster {cid}** — {len(members)} docs, {len(authors)} authors, {internal} links", expanded=False):
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:8px 12px;background:#f8f9fa;border-radius:4px">'
                f'<strong>Members:</strong> {", ".join(authors)}<br/>'
                f'<strong>Documents:</strong> {", ".join(m["label"][:20] for m in members)}<br/>'
                f'<strong>Avg Similarity:</strong> {avg_sim:.1f}% | <strong>Internal Links:</strong> {internal}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Member table
            member_data = [{"ID": m["id"], "Label": m["label"][:25], "Author": m["author"],
                            "Dept": m["department"], "Similarity": f"{m['similarity_score']}%",
                            "Words": m["word_count"]} for m in members]
            st.dataframe(pd.DataFrame(member_data), use_container_width=True, hide_index=True)


def _render_edge_analysis(data: dict, min_sim: float = 0):
    """Render relationship/edge analysis."""
    st.subheader("🔗 Relationship Analysis")
    edges = data["edges"]
    node_map = {n["id"]: n for n in data["nodes"]}

    filtered = [e for e in edges if e["similarity"] >= min_sim]

    if not filtered:
        st.info("No relationships match the current filters.")
        return

    # Edge type breakdown
    type_counts = {}
    for e in filtered:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Relationship Types:**")
        for t, count in type_counts.items():
            icon = "🔄" if t == "self-plagiarism" else "📄"
            st.markdown(f'{icon} **{t.replace("-", " ").title()}**: {count}')

    with c2:
        st.markdown("**Similarity Distribution:**")
        sim_buckets = {"90-100%": 0, "80-89%": 0, "70-79%": 0, "60-69%": 0, "50-59%": 0, "35-49%": 0}
        for e in filtered:
            s = e["similarity"]
            if s >= 90: sim_buckets["90-100%"] += 1
            elif s >= 80: sim_buckets["80-89%"] += 1
            elif s >= 70: sim_buckets["70-79%"] += 1
            elif s >= 60: sim_buckets["60-69%"] += 1
            elif s >= 50: sim_buckets["50-59%"] += 1
            else: sim_buckets["35-49%"] += 1

        for bucket, count in sim_buckets.items():
            pct = (count / len(filtered) * 100) if filtered else 0
            color = "#dc3545" if "90" in bucket or "80" in bucket else "#ffc107" if "70" in bucket or "60" in bucket else "#28a745"
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:3px 0">'
                f'<span style="width:80px;font-size:0.85em">{bucket}</span>'
                f'<div style="width:50%;background:#e0e0e0;border-radius:3px;height:12px">'
                f'<div style="width:{pct:.0f}%;background:{color};border-radius:3px;height:100%"></div></div>'
                f'<span style="margin-left:6px;font-size:0.82em">{count} ({pct:.0f}%)</span></div>',
                unsafe_allow_html=True,
            )

    # Top edges table
    st.markdown("**Strongest Relationships:**")
    top_edges = sorted(filtered, key=lambda e: e["similarity"], reverse=True)[:20]
    edge_rows = []
    for e in top_edges:
        src = node_map.get(e["source"], {})
        tgt = node_map.get(e["target"], {})
        edge_rows.append({
            "From": f"{e['source']} ({src.get('author', '?')})",
            "To": f"{e['target']} ({tgt.get('author', '?')})",
            "Similarity": f"{e['similarity']}%",
            "Segments": e["matched_segments"],
            "Phrases": e["shared_phrases"],
            "Type": e["type"],
        })
    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)


def _render_plagiarism_rings(data: dict):
    """Detect and render potential plagiarism rings."""
    st.subheader("🚨 Plagiarism Ring Detection")
    nodes = data["nodes"]
    edges = data["edges"]
    node_map = {n["id"]: n for n in nodes}

    # Build adjacency with high-similarity edges
    high_sim_edges = [e for e in edges if e["similarity"] > 70]
    adj: dict[str, list[tuple[str, float]]] = {}
    for e in high_sim_edges:
        adj.setdefault(e["source"], []).append((e["target"], e["similarity"]))
        adj.setdefault(e["target"], []).append((e["source"], e["similarity"]))

    # Find connected components (potential rings)
    visited = set()
    rings = []
    for node_id in adj:
        if node_id in visited:
            continue
        component = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor, _ in adj.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) >= 3:
            rings.append(component)

    if not rings:
        st.success("✅ No plagiarism rings detected (groups of 3+ documents with strong cross-links).")
    else:
        st.warning(f"⚠️ **{len(rings)} potential plagiarism ring(s)** detected!")
        for idx, ring in enumerate(rings):
            ring_nodes = [node_map[nid] for nid in ring if nid in node_map]
            authors = list(set(n["author"] for n in ring_nodes))
            avg_sim = sum(n["similarity_score"] for n in ring_nodes) / len(ring_nodes) if ring_nodes else 0

            with st.expander(f"🔴 Ring {idx + 1}: {len(ring)} documents, {len(authors)} authors, {avg_sim:.0f}% avg", expanded=True):
                st.markdown(
                    f'<div style="border:2px solid #dc3545;border-radius:8px;padding:12px;background:#fff5f5">'
                    f'<strong>⚠️ Plagiarism Ring</strong><br/>'
                    f'<strong>Documents:</strong> {", ".join(n["label"][:18] for n in ring_nodes)}<br/>'
                    f'<strong>Authors:</strong> {", ".join(authors)}<br/>'
                    f'<strong>Avg Similarity:</strong> {avg_sim:.1f}%<br/>'
                    f'<strong>Risk Level:</strong> {"🔴 Critical" if avg_sim > 80 else "🟠 High" if avg_sim > 60 else "🟡 Moderate"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                ring_edges = [e for e in high_sim_edges if e["source"] in ring and e["target"] in ring]
                for e in ring_edges:
                    st.markdown(f"  → {e['source']} ↔ {e['target']}: **{e['similarity']}%** ({e['matched_segments']} segments)")


def _render_author_centrality(data: dict):
    """Render author centrality and influence analysis."""
    st.subheader("📊 Author Centrality")
    nodes = data["nodes"]
    edges = data["edges"]
    node_map = {n["id"]: n for n in nodes}

    # Calculate degree centrality per author
    author_connections: dict[str, set[str]] = {}
    author_weights: dict[str, list[float]] = {}
    for e in edges:
        src = node_map.get(e["source"])
        tgt = node_map.get(e["target"])
        if not src or not tgt:
            continue
        author_connections.setdefault(src["author"], set()).add(tgt["author"])
        author_connections.setdefault(tgt["author"], set()).add(src["author"])
        author_weights.setdefault(src["author"], []).append(e["similarity"])
        author_weights.setdefault(tgt["author"], []).append(e["similarity"])

    centrality_data = []
    for author in author_connections:
        connections = len(author_connections[author])
        weights = author_weights.get(author, [])
        avg_weight = sum(weights) / len(weights) if weights else 0
        doc_count = sum(1 for n in nodes if n["author"] == author)
        centrality_data.append({
            "Author": author,
            "Documents": doc_count,
            "Connections": connections,
            "Avg Similarity": round(avg_weight, 1),
            "Centrality": round(connections / max(len(nodes) - 1, 1) * 100, 1),
        })

    centrality_df = pd.DataFrame(centrality_data).sort_values("Centrality", ascending=False)
    st.dataframe(centrality_df, use_container_width=True, hide_index=True)

    # Top authors by centrality
    st.markdown("**Most Connected Authors:**")
    for _, row in centrality_df.head(8).iterrows():
        c = row["Centrality"]
        color = "#dc3545" if c > 40 else "#fd7e14" if c > 25 else "#ffc107" if c > 15 else "#28a745"
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:3px 0">'
            f'<span style="width:130px;font-size:0.88em;font-weight:600">{row["Author"]}</span>'
            f'<div style="width:45%;background:#e0e0e0;border-radius:3px;height:14px">'
            f'<div style="width:{c}%;background:{color};border-radius:3px;height:100%"></div></div>'
            f'<span style="margin-left:8px;font-size:0.82em">{row["Connections"]} connections ({c}%)</span></div>',
            unsafe_allow_html=True,
        )


def _render_document_details(data: dict, selected_id: str | None):
    """Render detail panel for selected document."""
    st.subheader("📄 Document Details")
    if not selected_id:
        st.info("Click a node in the network graph to see details.")
        return

    node_map = {n["id"]: n for n in data["nodes"]}
    node = node_map.get(selected_id)
    if not node:
        st.warning("Document not found.")
        return

    st.markdown(
        f'<div style="border:2px solid #4a90d9;border-radius:8px;padding:16px;background:#f0f7ff">'
        f'<h3 style="margin:0 0 8px 0">📄 {node["label"]}</h3>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.9em">'
        f'<div><strong>ID:</strong> {node["id"]}</div>'
        f'<div><strong>Author:</strong> {node["author"]}</div>'
        f'<div><strong>Department:</strong> {node["department"]}</div>'
        f'<div><strong>Word Count:</strong> {node["word_count"]:,}</div>'
        f'<div><strong>Uploaded:</strong> {node["uploaded"]}</div>'
        f'<div><strong>Similarity:</strong> {node["similarity_score"]}%</div>'
        f'<div><strong>Cluster:</strong> {node["cluster"]}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Connected documents
    connected = []
    for e in data["edges"]:
        if e["source"] == selected_id:
            target = node_map.get(e["target"])
            if target:
                connected.append({"id": e["target"], "label": target["label"], "author": target["author"],
                                  "similarity": e["similarity"], "segments": e["matched_segments"],
                                  "phrases": e["shared_phrases"], "type": e["type"]})
        elif e["target"] == selected_id:
            source = node_map.get(e["source"])
            if source:
                connected.append({"id": e["source"], "label": source["label"], "author": source["author"],
                                  "similarity": e["similarity"], "segments": e["matched_segments"],
                                  "phrases": e["shared_phrases"], "type": e["type"]})

    if connected:
        connected.sort(key=lambda x: x["similarity"], reverse=True)
        st.markdown(f"**{len(connected)} connected documents:**")
        for c in connected:
            icon = "🔄" if c["type"] == "self-plagiarism" else "📄"
            color = "#dc3545" if c["similarity"] > 80 else "#fd7e14" if c["similarity"] > 60 else "#ffc107"
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:6px 10px;margin:4px 0;background:#f8f9fa;border-radius:3px">'
                f'{icon} <strong>{c["label"][:22]}</strong> ({c["author"]}) — '
                f'<span style="color:{color};font-weight:600">{c["similarity"]}%</span> | '
                f'{c["segments"]} segments, {c["phrases"]} phrases'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No plagiarism relationships found for this document.")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render_document_network():
    """Render the Document Network Graph page."""
    st.title("🕸️ Document Network Graph")
    st.markdown(
        "Visualize plagiarism relationships between documents, detect clusters, and identify plagiarism rings."
    )

    data = _generate_network_data()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Network Controls")

        min_sim = st.slider("Min Similarity (%)", 0, 95, 35)
        show_labels = st.checkbox("Show Node Labels", True)

        cluster_list = sorted(set(n["cluster"] for n in data["nodes"]))
        highlight_cluster = st.selectbox(
            "Highlight Cluster",
            [None] + cluster_list,
            format_func=lambda x: "All Clusters" if x is None else f"Cluster {x}",
        )

        st.markdown("---")
        st.subheader("📊 Sections")
        show_overview = st.checkbox("Network Overview", True)
        show_graph = st.checkbox("Interactive Graph", True)
        show_edges = st.checkbox("Relationship Analysis", True)
        show_clusters = st.checkbox("Cluster Analysis", True)
        show_rings = st.checkbox("Plagiarism Rings", True)
        show_centrality = st.checkbox("Author Centrality", True)
        show_details = st.checkbox("Document Details", True)

    if show_overview:
        _render_network_overview(data)

    # Network graph
    if show_graph:
        st.markdown("---")
        st.subheader("🕸️ Network Visualization")
        selected = st.session_state.get("selected_node")

        svg = _render_network_svg(
            data["nodes"], data["edges"], selected,
            min_sim=min_sim, show_labels=show_labels,
            highlight_cluster=highlight_cluster,
        )
        st.markdown(svg, unsafe_allow_html=True)

        # Node selector
        node_options = ["None"] + [f"{n['id']} — {n['label'][:25]}" for n in data["nodes"]]
        selected_str = st.selectbox("Select Document", node_options, key="net_node_select")
        if selected_str != "None":
            st.session_state["selected_node"] = selected_str.split(" — ")[0]
        else:
            st.session_state["selected_node"] = None

    if show_edges:
        st.markdown("---")
        _render_edge_analysis(data, min_sim=min_sim)

    if show_clusters:
        st.markdown("---")
        _render_cluster_analysis(data)

    if show_rings:
        st.markdown("---")
        _render_plagiarism_rings(data)

    if show_centrality:
        st.markdown("---")
        _render_author_centrality(data)

    if show_details:
        st.markdown("---")
        _render_document_details(data, st.session_state.get("selected_node"))

    # Stats footer
    filtered_edges = [e for e in data["edges"] if e["similarity"] >= min_sim]
    st.markdown("---")
    st.caption(
        f"Document Network | {len(data['nodes'])} nodes | {len(filtered_edges)} visible edges (≥{min_sim}%) | "
        f"{data['num_clusters']} clusters | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


# Entry point
if __name__ == "__main__" or True:
    render_document_network()
