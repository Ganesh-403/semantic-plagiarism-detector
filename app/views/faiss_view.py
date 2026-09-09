"""
FAISS Vector Search View Component.

Renders Tab 2 for querying the FAISS vector index and visualizing match results.
"""

import streamlit as st

from app.components.faiss_results import render_faiss_results_ui
from src.core.embedding_model import embed_chunks
from src.core.faiss_index import search_similar_chunks


def render_faiss_view(
    faiss_index,
    registry,
    faiss_top_k: int,
    threshold: float,
    file_bytes_dict: dict = None,
):
    """Render Tab 2: FAISS Vector Search."""
    st.subheader("⚡ FAISS Vector Search")
    if faiss_index is not None:
        st.info(f"Index total: {faiss_index.ntotal} vectors.")
        faiss_query = st.text_input("Query FAISS Index:", key="faiss_query_input_tab2")
        if st.button("Run Search", key="run_search_tab2") and faiss_query.strip():
            q_vec = embed_chunks([faiss_query.strip()])[0]
            results = search_similar_chunks(
                q_vec, faiss_index, registry, top_k=faiss_top_k, threshold=threshold
            )
            st.session_state["faiss_search_result"] = (faiss_query.strip(), results)
        # Keep the result available when an inspector button reruns the page.
        saved = st.session_state.get("faiss_search_result")
        if saved is not None:
            render_faiss_results_ui(saved[1], saved[0], document_pdf_bytes=file_bytes_dict)
    else:
        st.session_state.pop("faiss_search_result", None)
        st.info("No FAISS index is currently loaded.")
