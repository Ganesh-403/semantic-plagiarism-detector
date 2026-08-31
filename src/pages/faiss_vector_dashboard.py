"""Streamlit Dashboard Page for FAISS Semantic Search Vector Engine Suite."""

import random

import streamlit as st

from src.components.faiss_vector_card import render_vector_match_card
from src.components.faiss_vector_timeline import render_vector_search_timeline
from src.services.faiss_vector_engine import FaissSemanticVectorEngine


def init_demo_vector_database(engine: FaissSemanticVectorEngine):
    """Populates FAISS index with initial synthetic vector embeddings."""
    demo_docs = [
        (
            "DOC-101",
            "Introduction to Machine Learning",
            "Supervised learning relies on labeled datasets to train models.",
        ),
        (
            "DOC-102",
            "Deep Neural Networks Architecture",
            "Convolutional neural networks extract hierarchical spatial features.",
        ),
        (
            "DOC-103",
            "Natural Language Processing Guide",
            "Transformer models leverage self-attention mechanisms for context.",
        ),
        (
            "DOC-104",
            "Vector Databases & FAISS Indexing",
            "FAISS provides fast dense vector similarity search across high dimensions.",
        ),
    ]

    for doc_id, title, text in demo_docs:
        # Generate 384-dimensional synthetic normalized embedding
        vector = [round(random.uniform(-1.0, 1.0), 4) for _ in range(384)]
        engine.add_document_chunk(
            document_id=doc_id,
            document_title=title,
            chunk_index=0,
            raw_text=text,
            embedding=vector,
        )


def render_faiss_vector_dashboard():
    """Main rendering function for Streamlit FAISS vector dashboard tab."""
    st.set_page_config(page_title="FAISS Semantic Vector Engine", layout="wide")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #064E3B 100%);
            padding: 32px;
            border-radius: 24px;
            border: 1px solid #334155;
            margin-bottom: 28px;
        ">
            <span style="
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
                color: #6EE7B7;
                font-size: 12px;
                font-weight: 800;
                padding: 4px 14px;
                border-radius: 9999px;
            ">
                High-Performance Dense Vector Indexing
            </span>
            <h1 style="color: white; font-weight: 900; font-size: 36px; margin-top: 12px; margin-bottom: 8px;">
                FAISS Semantic Search & Vector Database Suite
            </h1>
            <p style="color: #94A3B8; font-size: 16px; margin: 0;">
                Query sub-millisecond k-NN semantic similarity matches using dense vector inner product embeddings and L2 metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "faiss_engine" not in st.session_state:
        engine = FaissSemanticVectorEngine(dimensions=384)
        init_demo_vector_database(engine)
        st.session_state["faiss_engine"] = engine

    if "search_reports" not in st.session_state:
        st.session_state["search_reports"] = []

    engine = st.session_state["faiss_engine"]
    telemetry = engine.get_index_telemetry()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Indexed Vectors", telemetry.total_vectors_indexed)
    with col2:
        st.metric("Vector Dimensions", telemetry.vector_dimensions)
    with col3:
        st.metric("Index Type", telemetry.index_type)

    st.markdown("---")

    st.subheader("Execute Semantic Vector Similarity Query")
    query_text = st.text_input(
        "Semantic Search Input Query",
        value="How do self-attention transformer models process contextual embeddings?",
    )
    top_k = st.slider("Select k-NN Top Candidates", min_value=1, max_value=4, value=3)

    if st.button("Run FAISS k-NN Vector Search", use_container_width=True):
        # Generate query vector
        query_vector = [round(random.uniform(-1.0, 1.0), 4) for _ in range(384)]
        report = engine.search_similar_chunks(query_vector, query_text, top_k=top_k)

        st.session_state["search_reports"].append(report)
        st.success(
            f"Query Executed in {report.execution_time_ms} ms! Top Cosine Similarity Score: {int(report.highest_similarity_ratio * 100)}%"
        )

    if st.session_state["search_reports"]:
        latest_report = st.session_state["search_reports"][-1]
        st.markdown("### Top k-NN Vector Search Results")

        for match in latest_report.matches:
            st.markdown(
                render_vector_match_card(
                    match if isinstance(match, dict) else match.__dict__
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_vector_search_timeline(st.session_state["search_reports"]),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_faiss_vector_dashboard()
