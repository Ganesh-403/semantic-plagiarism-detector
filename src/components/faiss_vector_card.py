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
Streamlit Dashboard Component for FAISS Vector Embedding Search Telemetry
"""

from typing import Any, Dict, List

import streamlit as st


class FAISSVectorDashboardComponent:
    """
    Renders enterprise Streamlit UI widgets for FAISS dense vector search,
    L2 distance distribution, and nearest neighbor matches.
    """

    @staticmethod
    def render_vector_index_metrics(vector_count: int, dimension: int) -> None:
        """Renders vector index state metrics card."""
        st.subheader("⚡ FAISS Dense Vector Index Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vectors Indexed", vector_count)
        with col2:
            st.metric("Embedding Dimension", f"{dimension}D")
        with col3:
            st.metric("Index Search Metric", "L2 Euclidean")

    @staticmethod
    def render_nearest_neighbor_results(results: list[dict[str, Any]]) -> None:
        """Renders top-k nearest neighbor match candidates."""
        st.subheader("🎯 Nearest Neighbor Semantic Matches")
        if not results:
            st.info("No nearest neighbor document matches found.")
            return

        for idx, res in enumerate(results):
            st.write(
                f"**Match #{idx + 1} - Document:** `{res.get('matchedDocId')}` | "
                f"**L2 Distance:** {res.get('l2Distance')} | "
                f"**Similarity Score:** {res.get('semanticSimilarityScore') * 100}%"
            )


# ==============================================================================
# STREAMLIT UI COMPONENT SPECIFICATIONS — FAISS VECTOR VISUALIZATION
# ------------------------------------------------------------------------------
# Section 1: Visual Design Guidelines
# - High-density metrics cards for vector search telemetry.
# - Clean interactive list layout for nearest neighbor search results.
# ==============================================================================
