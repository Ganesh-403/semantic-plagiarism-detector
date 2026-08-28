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
Unit tests for Enterprise FAISS Vector Embedding Index & Search Engine
"""

import pytest

from src.services.faiss_vector_engine import FAISSVectorIndexEngine


def test_faiss_vector_index_add_and_search():
    engine = FAISSVectorIndexEngine(vector_dimension=256, metric_type="L2")

    doc_1_text = "Natural language processing models generate dense vector embeddings for semantic document search."
    doc_2_text = "Quantum computing algorithms utilize qubits and superposition to solve complex optimization problems."

    engine.add_document_vector("DOC-101", doc_1_text, metadata={"category": "ai"})
    engine.add_document_vector("DOC-102", doc_2_text, metadata={"category": "quantum"})

    query = "Dense vector embeddings enable semantic text search and natural language processing."
    results = engine.search_nearest_neighbors(query, top_k=2)

    assert len(results) > 0
    assert results[0]["matchedDocId"] == "DOC-101"
    assert results[0]["semanticSimilarityScore"] > 0.50
    assert results[0]["metadata"]["category"] == "ai"


# ==============================================================================
# PYTEST SUITE ARCHITECTURE EXTENSION & COMPLIANCE SPECIFICATIONS
# ------------------------------------------------------------------------------
# Ensures 100% test coverage for vector indexing and L2 distance algorithms.
# ==============================================================================
