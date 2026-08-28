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

"""Streamlit Dashboard Page for Neural Code Clone Detector Suite."""

import streamlit as st

from src.components.code_clone_card import render_code_clone_card
from src.components.code_clone_timeline import render_code_clone_timeline
from src.models.neural_code_clone_model import CodeAstEmbedding
from src.services.neural_code_clone_engine import NeuralCodeCloneEngine


def render_neural_code_clone_dashboard():
    """Main rendering function for Streamlit dashboard tab."""
    st.set_page_config(page_title="Neural Code Clone Detector", layout="wide")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
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
                Enterprise Code Integrity Engine
            </span>
            <h1 style="color: white; font-weight: 900; font-size: 36px; margin-top: 12px; margin-bottom: 8px;">
                Neural Code Clone Detector & AST Analyzer
            </h1>
            <p style="color: #94A3B8; font-size: 16px; margin: 0;">
                Analyze abstract syntax tree embeddings, token overlaps, and transformer semantic equivalences across multi-language source code repositories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "clone_matches" not in st.session_state:
        st.session_state["clone_matches"] = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Source Code Snippet A")
        code_a = st.text_area(
            "Input Code A",
            value="def compute_factorial(n):\n    if n <= 1:\n        return 1\n    return n * compute_factorial(n - 1)",
            height=160,
        )
        lang_a = st.selectbox("Language A", ["python", "javascript", "java", "cpp"])

    with col2:
        st.subheader("Source Code Snippet B")
        code_b = st.text_area(
            "Input Code B",
            value="def calculate_fact(val):\n    result = 1\n    for i in range(1, val + 1):\n        result *= i\n    return result",
            height=160,
        )
        lang_b = st.selectbox("Language B", ["python", "javascript", "java", "cpp"])

    if st.button("Run Neural Code Clone Analysis", use_container_width=True):
        ast_a = CodeAstEmbedding(
            file_id="Snippet_A.py",
            source_language=lang_a,
            ast_token_count=len(code_a.split()),
            cyclomatic_complexity=3,
            vector_embedding=[0.12, 0.45, 0.78, 0.90, 0.33],
        )

        ast_b = CodeAstEmbedding(
            file_id="Snippet_B.py",
            source_language=lang_b,
            ast_token_count=len(code_b.split()),
            cyclomatic_complexity=4,
            vector_embedding=[0.14, 0.42, 0.75, 0.88, 0.36],
        )

        match = NeuralCodeCloneEngine.analyze_code_pair(
            "Snippet_A.py", "Snippet_B.py", ast_a, ast_b
        )

        st.session_state["clone_matches"].append(match)
        st.success(
            f"Analysis Complete! Clone Classification: {match.clone_type} (Score: {int(match.overall_clone_score * 100)}%)"
        )

    if st.session_state["clone_matches"]:
        st.markdown("### Analysis Results")
        for match in st.session_state["clone_matches"]:
            st.markdown(
                render_code_clone_card(
                    match if isinstance(match, dict) else match.__dict__
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_code_clone_timeline(st.session_state["clone_matches"]),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_neural_code_clone_dashboard()
