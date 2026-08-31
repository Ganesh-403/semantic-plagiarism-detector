"""
tests/core/test_semantic_alignment.py
-------------------------------------
Unit tests for the semantic-aware sequence alignment algorithm.
"""

import numpy as np
import pytest

from src.core.semantic_alignment import (
    MAX_SEQUENCE_LENGTH,
    _cosine_similarity_matrix,
    align_semantic_sequences,
)


class TestCosineSimilarityMatrix:
    """Tests for the internal cosine similarity matrix computation."""

    def test_identical_embeddings_return_ones(self):
        """Identical normalized vectors should produce 1.0 on the diagonal."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _cosine_similarity_matrix(emb, emb)
        np.testing.assert_array_almost_equal(np.diag(sim), [1.0, 1.0])

    def test_orthogonal_embeddings_return_zero(self):
        """Orthogonal vectors should produce 0.0 similarity."""
        emb_a = np.array([[1.0, 0.0]])
        emb_b = np.array([[0.0, 1.0]])
        sim = _cosine_similarity_matrix(emb_a, emb_b)
        assert sim[0, 0] == pytest.approx(0.0)

    def test_empty_arrays_return_empty_matrix(self):
        """Empty input arrays should return an empty matrix."""
        sim = _cosine_similarity_matrix(np.array([]), np.array([[1.0]]))
        assert sim.size == 0

    def test_handles_zero_vectors(self):
        """Zero vectors should not cause division by zero errors."""
        emb_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        emb_b = np.array([[1.0, 0.0]])
        sim = _cosine_similarity_matrix(emb_a, emb_b)
        assert sim[0, 0] == 0.0


class TestAlignSemanticSequences:
    """Tests for the banded DP alignment algorithm and edge cases (Issue #2029)."""

    def test_exact_match_alignment(self):
        """Identical sequences should align perfectly with 'match' type."""
        chunks = ["Sentence one.", "Sentence two."]
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])

        alignment = align_semantic_sequences(
            chunks, chunks, emb, emb, match_threshold=0.5
        )

        assert len(alignment) == 2
        assert all(op["type"] == "match" for op in alignment)
        assert all(op["score"] == pytest.approx(1.0) for op in alignment)

    def test_insertions_and_deletions(self):
        """Sequences with different lengths should produce gap operations."""
        chunks_a = ["A1", "A2", "A3"]
        chunks_b = ["B1", "B3"]

        emb_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        emb_b = np.array([[1.0, 0.0], [0.5, 0.5]])

        alignment = align_semantic_sequences(
            chunks_a, chunks_b, emb_a, emb_b, match_threshold=0.8, gap_penalty=-1.0
        )

        types = [op["type"] for op in alignment]
        assert "insert_a" in types or "insert_b" in types

    def test_both_inputs_empty(self):
        """Empty inputs ([], []) should return an empty alignment."""
        alignment = align_semantic_sequences([], [], np.array([]), np.array([]))
        assert alignment == []

    def test_only_a_empty(self):
        """Aligning ([], ['chunk']) should produce 'insert_b' operations for sequence B."""
        chunks_b = ["chunk"]
        emb_b = np.array([[1.0, 0.0]])

        alignment = align_semantic_sequences([], chunks_b, np.array([]), emb_b)

        assert len(alignment) == 1
        assert alignment[0]["type"] == "insert_b"
        assert alignment[0]["chunk_b"] == "chunk"

    def test_only_b_empty(self):
        """Aligning (['chunk'], []) should produce 'insert_a' operations for sequence A."""
        chunks_a = ["chunk"]
        emb_a = np.array([[1.0, 0.0]])

        alignment = align_semantic_sequences(chunks_a, [], emb_a, np.array([]))

        assert len(alignment) == 1
        assert alignment[0]["type"] == "insert_a"
        assert alignment[0]["chunk_a"] == "chunk"

    def test_single_chunk_inputs_match(self):
        """Aligning single identical chunks should produce a 'match' operation."""
        chunks_a = ["a"]
        chunks_b = ["a"]
        emb_a = np.array([[1.0, 0.0]])
        emb_b = np.array([[1.0, 0.0]])

        alignment = align_semantic_sequences(
            chunks_a, chunks_b, emb_a, emb_b, match_threshold=0.5
        )

        assert len(alignment) == 1
        assert alignment[0]["type"] == "match"
        assert alignment[0]["chunk_a"] == "a"
        assert alignment[0]["chunk_b"] == "a"

    def test_single_chunk_inputs_mismatch(self):
        """Aligning single distinct chunks (['a'], ['b']) should handle mismatch gracefully."""
        chunks_a = ["a"]
        chunks_b = ["b"]
        emb_a = np.array([[1.0, 0.0]])
        emb_b = np.array([[0.0, 1.0]])

        alignment = align_semantic_sequences(
            chunks_a, chunks_b, emb_a, emb_b, match_threshold=0.5
        )

        assert len(alignment) > 0
        valid_types = {"match", "mismatch", "insert_a", "insert_b"}
        assert all(op["type"] in valid_types for op in alignment)


class TestMemoryAllocationGuard:
    """Test suite for the N > 1000 memory allocation guard (Issue #2001)."""

    def test_raises_value_error_when_n_exceeds_limit(self):
        """Verify ValueError is raised when len(chunks_a) > 1000."""
        n = MAX_SEQUENCE_LENGTH + 1
        m = 10

        chunks_a = [f"chunk_{i}" for i in range(n)]
        chunks_b = [f"chunk_{i}" for i in range(m)]

        emb_a = np.random.rand(n, 384).astype(np.float32)
        emb_b = np.random.rand(m, 384).astype(np.float32)

        with pytest.raises(
            ValueError, match="Sequence alignment matrix size limit exceeded"
        ):
            align_semantic_sequences(chunks_a, chunks_b, emb_a, emb_b)

    def test_raises_value_error_when_m_exceeds_limit(self):
        """Verify ValueError is raised when len(chunks_b) > 1000."""
        n = 10
        m = MAX_SEQUENCE_LENGTH + 1

        chunks_a = [f"chunk_{i}" for i in range(n)]
        chunks_b = [f"chunk_{i}" for i in range(m)]

        emb_a = np.random.rand(n, 384).astype(np.float32)
        emb_b = np.random.rand(m, 384).astype(np.float32)

        with pytest.raises(
            ValueError, match="Sequence alignment matrix size limit exceeded"
        ):
            align_semantic_sequences(chunks_a, chunks_b, emb_a, emb_b)

    def test_raises_value_error_when_both_exceed_limit(self):
        """Verify ValueError is raised when both N and M > 1000."""
        n = MAX_SEQUENCE_LENGTH + 50
        m = MAX_SEQUENCE_LENGTH + 50

        chunks_a = [f"chunk_{i}" for i in range(n)]
        chunks_b = [f"chunk_{i}" for i in range(m)]

        emb_a = np.random.rand(n, 384).astype(np.float32)
        emb_b = np.random.rand(m, 384).astype(np.float32)

        with pytest.raises(ValueError, match="Maximum allowed is 1000x1000"):
            align_semantic_sequences(chunks_a, chunks_b, emb_a, emb_b)

    def test_succeeds_at_exact_limit(self):
        """Verify alignment succeeds when N and M are exactly at the limit."""
        n = MAX_SEQUENCE_LENGTH
        m = MAX_SEQUENCE_LENGTH

        chunks_a = [f"chunk_{i}" for i in range(n)]
        chunks_b = [f"chunk_{i}" for i in range(m)]

        emb_a = np.random.rand(n, 384).astype(np.float32)
        emb_b = np.random.rand(m, 384).astype(np.float32)

        # Should not raise
        alignment = align_semantic_sequences(chunks_a, chunks_b, emb_a, emb_b)
        assert isinstance(alignment, list)

    def test_succeeds_below_limit(self):
        """Verify alignment succeeds for normal document sizes."""
        n = 50
        m = 60

        chunks_a = [f"chunk_{i}" for i in range(n)]
        chunks_b = [f"chunk_{i}" for i in range(m)]

        emb_a = np.random.rand(n, 384).astype(np.float32)
        emb_b = np.random.rand(m, 384).astype(np.float32)

        alignment = align_semantic_sequences(chunks_a, chunks_b, emb_a, emb_b)
        assert len(alignment) > 0
