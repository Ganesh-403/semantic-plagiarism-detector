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
        faiss_query = st.text_input("Query FAISS Index:", key="faiss_query_input")
        if st.button("Run Search") and faiss_query.strip():
            q_vec = embed_chunks([faiss_query.strip()])[0]
            results = search_similar_chunks(
                q_vec, faiss_index, registry, top_k=faiss_top_k, threshold=threshold
            )
            render_faiss_results_ui(
                results, faiss_query.strip(), document_pdf_bytes=file_bytes_dict
            )
    else:
        st.info("No FAISS index is currently loaded.")
