"""
Heatmap & Network Graph View Component.

Renders Tab 4 containing similarity heatmap, network visualization,
and collusion ring cluster summaries.
"""

import streamlit as st

from app.state_manager import ui_exception_handler
from app.theme import get_chart_colors
from src.core.similarity import detect_plagiarism_clusters
from src.visualization.heatmap import plot_similarity_heatmap
from src.visualization.network_graph import plot_similarity_network


def render_heatmap_view(active_sim_df, threshold: float, doc_names: list):
    """Render Tab 4: Heatmap & Network Graph."""
    st.subheader("🗺️ Heatmap & Network")
    heatmap_fig = None
    load_heatmap = st.checkbox("Show similarity heatmap", key="load_similarity_heatmap")
    if load_heatmap and active_sim_df is not None:
        heatmap_fig = ui_exception_handler("Similarity Heatmap")(
            plot_similarity_heatmap
        )(active_sim_df, threshold=threshold, theme_colors=get_chart_colors())

    if heatmap_fig is not None:
        st.pyplot(heatmap_fig, use_container_width=True)

    doc_select_options = (
        ["None"] + list(active_sim_df.columns)
        if active_sim_df is not None
        else ["None"]
    )
    selected_highlight_doc = st.selectbox(
        "Highlight Document Node",
        options=doc_select_options,
        index=0,
        key="highlight_doc_node_selector",
    )
    highlighted_doc = (
        selected_highlight_doc if selected_highlight_doc != "None" else None
    )

    network_fig = None
    load_network = st.checkbox("Show plagiarism network", key="load_plagiarism_network")
    if load_network and active_sim_df is not None:
        network_fig = ui_exception_handler("Plagiarism Network")(
            plot_similarity_network
        )(
            similarity_df=active_sim_df,
            threshold=threshold,
            highlighted_doc=highlighted_doc,
            title="Interactive Document Plagiarism Network",
        )

    if network_fig is not None:
        st.plotly_chart(network_fig, use_container_width=True)

    if active_sim_df is not None and len(doc_names) >= 2:
        cluster_data = detect_plagiarism_clusters(active_sim_df, threshold=threshold)
        suspicious_groups = cluster_data.get("suspicious_groups", [])

        if suspicious_groups:
            with st.expander(
                f"🚨 Suspicious Collusion Rings Detected ({len(suspicious_groups)})",
                expanded=True,
            ):
                st.warning(
                    f"Found {len(suspicious_groups)} group(s) of 3+ highly similar documents. "
                    "These may indicate collusion or shared source material."
                )

                for group in suspicious_groups:
                    st.markdown(
                        f"**Cluster #{group['cluster_id']}** ({group['size']} documents):"
                    )
                    for doc in group["documents"]:
                        st.markdown(f"- 📄 `{doc}`")
                    st.divider()
