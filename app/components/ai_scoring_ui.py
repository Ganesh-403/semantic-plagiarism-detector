# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
AI Plagiarism Scoring Dashboard Component.

Streamlit-based interface for AI-powered plagiarism scoring with
detailed metrics visualization and comparison reports.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.ai_scoring_engine import (
    AIScoringEngine,
    ContentFingerprinter,
    PlagiarismScore,
    ScoringConfig,
)


def render_ai_scoring_dashboard():
    """Render the main AI scoring dashboard."""
    st.title("🤖 AI-Powered Plagiarism Scoring")
    st.markdown(
        "Advanced plagiarism detection with **multi-metric scoring** and **content fingerprinting**."
    )

    tab_score, tab_fingerprint, tab_analytics = st.tabs(
        ["📊 AI Scoring", "🔍 Fingerprinting", "📈 Analytics"]
    )

    with tab_score:
        _render_scoring()

    with tab_fingerprint:
        _render_fingerprinting()

    with tab_analytics:
        _render_analytics()


def _render_scoring():
    """Render AI scoring interface."""
    st.subheader("Document Pair Scoring")

    with st.form("ai_score_form"):
        col1, col2 = st.columns(2)
        with col1:
            doc_a_name = st.text_input("Document A Name", value="Document A")
            text_a = st.text_area(
                "Document A Text",
                height=200,
                placeholder="Paste the first document text here...",
            )
        with col2:
            doc_b_name = st.text_input("Document B Name", value="Document B")
            text_b = st.text_area(
                "Document B Text",
                height=200,
                placeholder="Paste the second document text here...",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.35, 0.05)
        with col2:
            lexical_weight = st.slider("Lexical Weight", 0.0, 1.0, 0.25, 0.05)
        with col3:
            enable_fp = st.checkbox("Enable Fingerprinting", value=True)

        submitted = st.form_submit_button(
            "🚀 Run AI Scoring", type="primary", use_container_width=True
        )

    if submitted and text_a.strip() and text_b.strip():
        config = ScoringConfig(
            weights={
                "semantic": semantic_weight,
                "lexical": lexical_weight,
                "structural": 0.15,
                "statistical": 0.15,
                "fingerprint": 0.10,
            },
            enable_fingerprint=enable_fp,
        )
        engine = AIScoringEngine(config)
        score = engine.score_documents(text_a, text_b, doc_a_name, doc_b_name)
        st.session_state.ai_score = score
        _display_score_result(score)
    elif submitted:
        st.warning("Please enter text for both documents.")


def _display_score_result(score: PlagiarismScore):
    """Display scoring result with visualizations."""
    severity_colors = {
        "clean": "#22c55e",
        "low": "#84cc16",
        "moderate": "#eab308",
        "high": "#f97316",
        "critical": "#ef4444",
    }
    color = severity_colors.get(score.severity.value, "#64748b")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{score.overall_score:.1%}", delta=None)
    with col2:
        st.metric("Severity", score.severity.value.upper())
    with col3:
        st.metric("Fingerprint Match", "✅ Yes" if score.fingerprint_match else "❌ No")

    st.progress(min(score.overall_score, 1.0))

    st.subheader("📊 Component Breakdown")
    component_data = []
    for comp in score.components:
        if comp.method != "ensemble":
            component_data.append(
                {
                    "Method": comp.method.title(),
                    "Score": f"{comp.score:.1%}",
                    "Confidence": f"{comp.confidence:.1%}",
                    "Details": ", ".join(
                        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in comp.details.items()
                    ),
                }
            )
    if component_data:
        st.dataframe(pd.DataFrame(component_data), use_container_width=True)

    st.subheader("📈 Score Radar Chart")
    radar_data = [c for c in score.components if c.method != "ensemble"]
    if radar_data:
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=[c.score for c in radar_data],
                theta=[c.method.title() for c in radar_data],
                fill="toself",
                name="Score",
                line=dict(color=color),
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Details", expanded=False):
        st.json(score.to_dict())


def _render_fingerprinting():
    """Render content fingerprinting interface."""
    st.subheader("Content Fingerprinting")

    st.markdown(
        "Generate and compare content fingerprints for **near-duplicate detection**."
    )

    with st.form("fingerprint_form"):
        col1, col2 = st.columns(2)
        with col1:
            fp_text_a = st.text_area("Document A", height=150, key="fp_a")
        with col2:
            fp_text_b = st.text_area("Document B", height=150, key="fp_b")
        shingle_size = st.slider("Shingle Size", 3, 10, 5)
        if st.form_submit_button("🔍 Generate & Compare", type="primary"):
            if fp_text_a.strip() and fp_text_b.strip():
                config = ScoringConfig(shingle_size=shingle_size)
                fp_engine = ContentFingerprinter(config)
                fp_a = fp_engine.create_fingerprint(fp_text_a, "Doc A")
                fp_b = fp_engine.create_fingerprint(fp_text_b, "Doc B")
                similarity = fp_engine.compare_fingerprints(fp_a, fp_b)
                st.session_state.fp_result = {
                    "fp_a": fp_a,
                    "fp_b": fp_b,
                    "similarity": similarity,
                }

    fp_result = st.session_state.get("fp_result")
    if fp_result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Shingle Jaccard", f"{fp_result['similarity']:.1%}")
        col2.metric("Doc A Shingles", len(fp_result["fp_a"].shingles))
        col3.metric("Doc B Shingles", len(fp_result["fp_b"].shingles))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Doc A Fingerprint**")
            st.json(fp_result["fp_a"].to_dict())
        with col2:
            st.markdown("**Doc B Fingerprint**")
            st.json(fp_result["fp_b"].to_dict())


def _render_analytics():
    """Render scoring analytics."""
    st.subheader("Scoring Analytics")

    if "ai_score" in st.session_state:
        score = st.session_state.ai_score
        st.markdown(f"**Last Scored Pair:** {score.doc_a} ↔ {score.doc_b}")

        components = [c for c in score.components if c.method != "ensemble"]
        if components:
            fig = px.bar(
                x=[c.method.title() for c in components],
                y=[c.score for c in components],
                color=[c.score for c in components],
                color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
                title="Component Scores",
            )
            fig.update_layout(
                xaxis_title="Method", yaxis_title="Score", yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a scoring first to see analytics here.")

    st.subheader("📋 Scoring Methods Guide")
    methods = [
        {
            "Method": "Semantic",
            "Description": "Word overlap and cosine similarity",
            "Weight": "35%",
        },
        {
            "Method": "Lexical",
            "Description": "Bigram and trigram matching",
            "Weight": "25%",
        },
        {
            "Method": "Structural",
            "Description": "Paragraph and sentence structure",
            "Weight": "15%",
        },
        {
            "Method": "Statistical",
            "Description": "Type-token ratio and keyword overlap",
            "Weight": "15%",
        },
        {
            "Method": "Fingerprint",
            "Description": "Shingling and MinHash comparison",
            "Weight": "10%",
        },
    ]
    st.dataframe(pd.DataFrame(methods), use_container_width=True)
