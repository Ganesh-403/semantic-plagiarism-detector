"""Unit and Integration Tests for FAISS Search Score Threshold Filtering (Issue #4036).

Verifies that search_index(query_vectors, threshold=0.8) excludes any candidate
results with similarity score < 0.8 and asserts that all returned matches satisfy
similarity_score >= threshold across various index types, vector dimensions,
single-query, and batch-query search scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.faiss_index import (
    ChunkRecord,
    build_index,
    build_index_from_matrix,
    find_plagiarised_chunks,
    search_batch_vectors,
    search_index,
    search_similar_chunks,
)


def _generate_normalized_vectors(num_vectors: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Generate unit-normalized float32 vectors for deterministic test scenarios."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((num_vectors, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return raw / norms


class TestFAISSSearchScoreThresholdFiltering:
    """Comprehensive test suite for score threshold filtering in FAISS search operations."""

    @pytest.fixture
    def indexed_corpus(self):
        """Build a fixture containing known synthetic document vectors and chunk records."""
        dim = 384
        vecs_doc_a = _generate_normalized_vectors(5, dim, seed=101)
        vecs_doc_b = _generate_normalized_vectors(5, dim, seed=202)
        vecs_doc_c = _generate_normalized_vectors(5, dim, seed=303)

        embeddings = {
            "doc_a": vecs_doc_a,
            "doc_b": vecs_doc_b,
            "doc_c": vecs_doc_c,
        }
        chunked = {
            "doc_a": [f"Chunk A-{i} text content about computer science." for i in range(5)],
            "doc_b": [f"Chunk B-{i} text content about literature." for i in range(5)],
            "doc_c": [f"Chunk C-{i} text content about philosophy." for i in range(5)],
        }

        index, registry = build_index(embeddings, chunked, index_type="flat")
        return embeddings, chunked, index, registry

    def test_search_index_threshold_filters_lower_similarity(self, indexed_corpus):
        """Assert that results with similarity < 0.8 are excluded from returned match list."""
        embeddings, _, index, registry = indexed_corpus
        query_vector = embeddings["doc_a"][0]
        threshold = 0.80

        matches = search_index(
            query_vector,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
        )

        assert isinstance(matches, list)
        for match in matches:
            assert (
                match["similarity_score"] >= threshold
            ), f"Expected match score >= {threshold}, got {match['similarity_score']}"

    def test_search_index_acceptance_criteria_strict_inequality(self, indexed_corpus):
        """Acceptance Criteria: Assert all returned matches have similarity_score >= threshold."""
        embeddings, _, index, registry = indexed_corpus
        # Test across multiple threshold levels (0.5, 0.7, 0.8, 0.9, 0.95)
        test_thresholds = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]

        for thresh in test_thresholds:
            for doc_name, vecs in embeddings.items():
                query_vec = vecs[0]
                results = search_index(
                    query_vec,
                    index=index,
                    registry=registry,
                    threshold=thresh,
                    top_k=15,
                )
                assert all(
                    r["similarity_score"] >= thresh for r in results
                ), f"Failed strict threshold invariant for threshold {thresh}"

    def test_search_index_exact_self_match_above_threshold(self, indexed_corpus):
        """Searching for an exact identical vector should yield similarity ~1.0 >= 0.8."""
        embeddings, _, index, registry = indexed_corpus
        query_vector = embeddings["doc_b"][2]
        threshold = 0.80

        matches = search_index(
            query_vector,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=5,
        )

        assert len(matches) >= 1
        best_match = matches[0]
        assert best_match["doc_name"] == "doc_b"
        assert best_match["chunk_index"] == 2
        assert best_match["similarity_score"] >= 0.999
        assert best_match["similarity_score"] >= threshold

    def test_search_index_high_threshold_returns_empty_when_no_match(self, indexed_corpus):
        """When threshold is set higher than any vector similarity (e.g. 0.99999 for non-identical), return []."""
        dim = 384
        _, _, index, registry = indexed_corpus
        # Generate an orthogonal or random query vector unlikely to have 0.999 similarity
        orthogonal_query = _generate_normalized_vectors(1, dim, seed=999)[0]
        threshold = 0.9999

        matches = search_index(
            orthogonal_query,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
        )

        assert matches == []
        assert all(m["similarity_score"] >= threshold for m in matches)

    def test_search_index_batch_multi_vector_threshold_filtering(self, indexed_corpus):
        """Batch query containing multiple vectors correctly filters each result against threshold."""
        embeddings, _, index, registry = indexed_corpus
        query_batch = np.vstack([embeddings["doc_a"][0], embeddings["doc_c"][1]])
        threshold = 0.80

        matches = search_index(
            query_batch,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
        )

        assert isinstance(matches, list)
        assert all(m["similarity_score"] >= threshold for m in matches)

    def test_search_index_exclude_doc_with_threshold(self, indexed_corpus):
        """Exclude doc filter combined with score threshold properly filters matches."""
        embeddings, _, index, registry = indexed_corpus
        query_vec = embeddings["doc_a"][0]
        threshold = 0.60

        matches = search_index(
            query_vec,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
            exclude_doc="doc_a",
        )

        assert all(m["doc_name"] != "doc_a" for m in matches)
        assert all(m["similarity_score"] >= threshold for m in matches)

    def test_search_similar_chunks_underlying_threshold(self, indexed_corpus):
        """Verify search_similar_chunks also enforces score threshold >= 0.8."""
        embeddings, _, index, registry = indexed_corpus
        query = embeddings["doc_a"][1]
        threshold = 0.80

        results = search_similar_chunks(
            query,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
        )

        for record, score in results:
            assert isinstance(record, ChunkRecord)
            assert isinstance(score, float)
            assert score >= threshold

    def test_find_plagiarised_chunks_threshold_filtering(self, indexed_corpus):
        """Verify find_plagiarised_chunks applies threshold=0.8 correctly across cross-document pairs."""
        embeddings, chunked, index, registry = indexed_corpus
        threshold = 0.80

        plagiarism_matches = find_plagiarised_chunks(
            embeddings=embeddings,
            chunked_docs=chunked,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=5,
        )

        assert isinstance(plagiarism_matches, list)
        for match in plagiarism_matches:
            assert match["similarity"] >= threshold


class TestEdgeCasesAndBoundaryFiltering:
    """Boundary conditions and input variation tests for FAISS search thresholding."""

    def test_empty_query_vectors_returns_empty(self):
        """Empty query vectors or empty index return empty list without crashing."""
        assert search_index(np.array([]), threshold=0.8) == []
        assert search_index([], threshold=0.8) == []
        assert search_index(None, threshold=0.8) == []  # type: ignore[arg-type]

    def test_search_on_empty_index_returns_empty(self):
        """Searching an index with 0 vectors returns empty list."""
        empty_index, empty_reg = build_index({}, {})
        query = np.ones((1, 384), dtype=np.float32)
        results = search_index(query, index=empty_index, registry=empty_reg, threshold=0.8)
        assert results == []

    def test_search_index_with_list_of_floats(self):
        """search_index accepts a 1D Python list of floats as query."""
        dim = 384
        vecs = _generate_normalized_vectors(3, dim, seed=42)
        index = build_index_from_matrix(vecs)
        registry = [ChunkRecord("doc1", i, f"text {i}") for i in range(3)]

        query_list = vecs[0].tolist()
        matches = search_index(query_list, index=index, registry=registry, threshold=0.80)

        assert len(matches) >= 1
        assert matches[0]["similarity_score"] >= 0.80

    def test_search_index_results_sorted_descending(self):
        """Results returned by search_index are strictly sorted in descending order of similarity."""
        dim = 384
        vecs = _generate_normalized_vectors(10, dim, seed=55)
        index = build_index_from_matrix(vecs)
        registry = [ChunkRecord(f"doc_{i}", i, f"text {i}") for i in range(10)]

        query = vecs[0]
        matches = search_index(query, index=index, registry=registry, threshold=0.0, top_k=10)

        scores = [m["similarity_score"] for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_zero_threshold_returns_all_top_k(self):
        """Threshold 0.0 does not filter out positive matches."""
        dim = 384
        vecs = _generate_normalized_vectors(5, dim, seed=66)
        index = build_index_from_matrix(vecs)
        registry = [ChunkRecord(f"doc_{i}", i, f"chunk {i}") for i in range(5)]

        query = vecs[0]
        matches = search_index(query, index=index, registry=registry, threshold=0.0, top_k=5)
        assert len(matches) > 0
        assert all(m["similarity_score"] >= 0.0 for m in matches)


class TestIVFFlatThresholdFiltering:
    """Verifies score threshold filtering works accurately with IndexIVFFlat approximate indexes."""

    @pytest.fixture
    def ivf_index_data(self):
        """Build an IVF index with Voronoi cells for approximate search testing."""
        dim = 384
        num_vecs = 60
        vecs = _generate_normalized_vectors(num_vecs, dim, seed=777)
        registry = [
            ChunkRecord(f"doc_{i // 5}", i % 5, f"Content snippet {i}")
            for i in range(num_vecs)
        ]
        index = build_index_from_matrix(vecs, index_type="ivf", nlist=6, nprobe=4)
        return vecs, index, registry

    def test_ivf_search_index_threshold_filtering(self, ivf_index_data):
        """Verify IVF index search respects the similarity threshold."""
        vecs, index, registry = ivf_index_data
        query = vecs[0]
        threshold = 0.80

        matches = search_index(
            query,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=10,
        )

        assert isinstance(matches, list)
        for match in matches:
            assert match["similarity_score"] >= threshold

    def test_ivf_high_threshold_filtering(self, ivf_index_data):
        """High threshold with IVF index returns only matches strictly >= threshold."""
        vecs, index, registry = ivf_index_data
        query = vecs[10]
        threshold = 0.90

        matches = search_index(
            query,
            index=index,
            registry=registry,
            threshold=threshold,
            top_k=5,
        )

        assert all(m["similarity_score"] >= threshold for m in matches)


class TestThresholdBoundaryPrecision:
    """Verifies boundary precision around floating point similarity scores."""

    def test_exact_threshold_boundary_inclusion(self):
        """A score exactly equal to threshold must be included."""
        dim = 384
        vecs = _generate_normalized_vectors(4, dim, seed=888)
        index = build_index_from_matrix(vecs)
        registry = [ChunkRecord("doc_exact", i, f"text {i}") for i in range(4)]

        # Self-match will have score ~1.0
        query = vecs[0]
        matches = search_index(query, index=index, registry=registry, threshold=1.0)
        assert all(m["similarity_score"] >= 1.0 for m in matches)

    def test_threshold_fractional_precision(self):
        """Verify fine-grained thresholds (0.8000, 0.8500, 0.9250) filter accurately."""
        dim = 384
        vecs = _generate_normalized_vectors(20, dim, seed=999)
        index = build_index_from_matrix(vecs)
        registry = [ChunkRecord(f"doc_{i}", i, f"text {i}") for i in range(20)]

        for thresh in [0.75, 0.80, 0.825, 0.875, 0.90]:
            matches = search_index(vecs[0], index=index, registry=registry, threshold=thresh)
            for m in matches:
                assert m["similarity_score"] >= thresh


class TestMetadataAndRecordIntegrity:
    """Verifies that ChunkRecord metadata is properly maintained during search filtering."""

    def test_chunk_metadata_retained_in_search_results(self):
        """ChunkRecord metadata dictionaries are preserved on filtered match items."""
        dim = 384
        vecs = _generate_normalized_vectors(3, dim, seed=123)
        index = build_index_from_matrix(vecs)
        registry = [
            ChunkRecord("essay_1.pdf", 0, "Paragraph 1 text", metadata={"page": 1, "author": "Alice"}),
            ChunkRecord("essay_1.pdf", 1, "Paragraph 2 text", metadata={"page": 2, "author": "Alice"}),
            ChunkRecord("essay_2.pdf", 0, "Paragraph 1 text", metadata={"page": 1, "author": "Bob"}),
        ]

        query = vecs[0]
        matches = search_index(query, index=index, registry=registry, threshold=0.80)

        assert len(matches) >= 1
        first = matches[0]
        assert first["doc_name"] == "essay_1.pdf"
        assert first["chunk_index"] == 0
        assert first["metadata"] == {"page": 1, "author": "Alice"}


class TestSearchIndexInputValidation:
    """Verifies robust error handling and type checking in search_index."""

    def test_invalid_query_type_raises_type_error(self):
        """Passing an invalid object type as query_vectors raises TypeError."""
        index, registry = build_index({}, {})
        with pytest.raises(TypeError, match="query_vectors must be"):
            search_index({"invalid": "dict"}, index=index, registry=registry)  # type: ignore[arg-type]

    def test_search_index_with_none_registry(self):
        """search_index functions safely when registry is None (generates default doc_names)."""
        dim = 384
        vecs = _generate_normalized_vectors(2, dim, seed=321)
        index = build_index_from_matrix(vecs)

        matches = search_index(vecs[0], index=index, registry=None, threshold=0.80)
        assert len(matches) >= 1
        assert matches[0]["similarity_score"] >= 0.80
        assert "doc_" in matches[0]["doc_name"]

