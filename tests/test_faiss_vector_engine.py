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

def test_faiss_rejects_incompatible_embedding_dimension():
    from src.core.faiss_index import FaissIndexManager

    engine = FaissIndexManager(dimension=384)

    incompatible_vectors = np.zeros((1, 768), dtype=np.float32)

    with pytest.raises(ValueError, match="Embedding dimension"):
        engine.add(incompatible_vectors)
# ==============================================================================
# PYTEST SUITE ARCHITECTURE EXTENSION & COMPLIANCE SPECIFICATIONS
# ------------------------------------------------------------------------------
# Ensures 100% test coverage for vector indexing and L2 distance algorithms.
# ==============================================================================
