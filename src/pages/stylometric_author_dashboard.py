"""Streamlit Dashboard Page for Stylometric Author Attribution Engine Suite."""

import streamlit as st

from src.components.stylometric_author_card import render_stylometric_card
from src.components.stylometric_author_timeline import render_stylometric_timeline
from src.services.stylometric_author_engine import StylometricAuthorEngine


def render_stylometric_author_dashboard():
    """Main rendering function for Streamlit Stylometric Author dashboard tab."""
    st.set_page_config(
        page_title="Stylometric Author Attribution Engine", layout="wide"
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #312E81 100%);
            padding: 32px;
            border-radius: 24px;
            border: 1px solid #334155;
            margin-bottom: 28px;
        ">
            <span style="
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.4);
                color: #A5B4FC;
                font-size: 12px;
                font-weight: 800;
                padding: 4px 14px;
                border-radius: 9999px;
            ">
                Linguistic Stylometry & Author Profiling
            </span>
            <h1 style="color: white; font-weight: 900; font-size: 36px; margin-top: 12px; margin-bottom: 8px;">
                Stylometric Author Attribution & Forensic Profiler
            </h1>
            <p style="color: #94A3B8; font-size: 16px; margin: 0;">
                Analyze sentence length variance, Type-Token Ratio (TTR), Hapax Legomena, and function word distributions for forensic author verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "stylometric_matches" not in st.session_state:
        st.session_state["stylometric_matches"] = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Query Document (Anonymous Author)")
        query_text = st.text_area(
            "Query Text Content",
            value="However, the empirical evidence demonstrates that deep neural networks require substantial amounts of labeled data. In addition, the computational overhead associated with backpropagation remains a critical bottleneck for real-time edge deployment.",
            height=160,
        )
        query_doc_id = st.text_input("Query Document ID", value="DOC-QUERY-999")

    with col2:
        st.subheader("Candidate Author Profile Document")
        candidate_text = st.text_area(
            "Candidate Author Text Corpus",
            value="Furthermore, empirical results indicate that deep learning architectures are highly data-dependent. Therefore, hardware acceleration is necessary to mitigate the computational latency observed during gradient optimization.",
            height=160,
        )
        candidate_alias = st.text_input(
            "Candidate Author Alias", value="Prof. Alex Mercer"
        )

    if st.button("Run Stylometric Authorship Verification", use_container_width=True):
        fp_query = StylometricAuthorEngine.extract_fingerprint(
            document_id=query_doc_id, author_alias="Anonymous", text_content=query_text
        )

        fp_candidate = StylometricAuthorEngine.extract_fingerprint(
            document_id="DOC-CANDIDATE-001",
            author_alias=candidate_alias,
            text_content=candidate_text,
        )

        match = StylometricAuthorEngine.compare_authorship(fp_query, fp_candidate)
        st.session_state["stylometric_matches"].append(match)

        if match.is_same_author:
            st.success(
                f"Same Author Verified! Confidence: {match.attribution_confidence_percentage}% (Distance: {match.stylometric_distance})"
            )
        else:
            st.info(
                f"Different Author / Writing Style Detected. Confidence: {match.attribution_confidence_percentage}% (Distance: {match.stylometric_distance})"
            )

    if st.session_state["stylometric_matches"]:
        st.markdown("### Attribution Results")
        for match in st.session_state["stylometric_matches"]:
            st.markdown(
                render_stylometric_card(
                    match if isinstance(match, dict) else match.__dict__
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_stylometric_timeline(st.session_state["stylometric_matches"]),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_stylometric_author_dashboard()
