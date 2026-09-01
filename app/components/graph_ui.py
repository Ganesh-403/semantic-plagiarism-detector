"""
Streamlit UI Visualization component for the Plagiarism Graph Engine.
Renders interactive graphs using Plotly Network Graphs for visual ring detection.
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Optional
from src.core.graph_engine.engine import PlagiarismGraphEngine

def render_interactive_graph(graph: nx.Graph, title: str = "Plagiarism Ecosystem Graph"):
    """
    Renders a NetworkX graph interactively using Plotly.
    """
    if graph.number_of_nodes() == 0:
        st.info("No data available to construct the graph.")
        return

    # Use spring layout for force-directed community clustering
    pos = nx.spring_layout(graph, seed=42)
    
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Differentiate between Students and Documents
        labels = data.get('labels', set())
        if "Student" in labels:
            node_color.append('blue')
            node_text.append(f"Student: {data.get('name', node)}")
        else:
            node_color.append('red')
            node_text.append(f"Doc: {data.get('title', node)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[text.split(':')[1][:15] for text in node_text], # Short text label
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
    st.plotly_chart(fig, use_container_width=True)


def render_graph_dashboard(engine: PlagiarismGraphEngine):
    """
    Renders the Streamlit dashboard for exploring Plagiarism Rings and Communities.
    """
    st.header("🕸️ Plagiarism Ring & Ecosystem Analysis")
    st.markdown("Analyze historical trends and detect complex collusion rings beyond 1-to-1 matching.")
    
    stats = engine.get_graph_stats()
    
    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", stats.node_count)
    col2.metric("Total Connections", stats.edge_count)
    col3.metric("Suspicious Clusters", stats.suspicious_clusters)
    col4.metric("Graph Density", f"{stats.density:.4f}")
    
    st.divider()
    
    # Filters
    st.subheader("Filter Ecosystem")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        min_sim = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.40, 0.05)
    with f_col2:
        # Extract unique years
        years = sorted(list(stats.historical_trends.keys()))
        year_filter = st.selectbox("Year", ["All Time"] + years)
        selected_year = None if year_filter == "All Time" else year_filter
        
    filtered_graph = engine.filter_graph(min_similarity=min_sim, year=selected_year)
    
    st.subheader(f"Ecosystem Network Map ({filtered_graph.number_of_nodes()} Nodes)")
    render_interactive_graph(filtered_graph)
    
    st.divider()
    
    # Community & Ring Detection
    st.subheader("🔍 Suspicious Rings & Communities")
    rings = engine.detect_suspicious_rings(min_similarity=min_sim)
    
    if not rings:
        st.success("No suspicious plagiarism rings detected at this threshold.")
    else:
        for ring in rings:
            if ring.is_suspicious:
                with st.expander(f"🚨 Suspicious Ring #{ring.community_id} ({ring.size} Documents)"):
                    st.write(f"**Average Similarity:** {ring.average_similarity:.1%}")
                    st.write("**Documents Involved:**")
                    st.write(", ".join(ring.nodes))
            else:
                with st.expander(f"ℹ️ Minor Cluster #{ring.community_id} ({ring.size} Documents)"):
                    st.write(f"**Average Similarity:** {ring.average_similarity:.1%}")
                    st.write(", ".join(ring.nodes))

    # Centrality/Hubs
    if stats.highly_connected_students:
        st.subheader("🔗 Central Hubs (Potential Sources)")
        for node_info in stats.highly_connected_students:
            st.write(f"- Node **{node_info['id']}** (Centrality Score: {node_info['score']:.4f})")
