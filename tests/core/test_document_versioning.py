"""
tests/core/test_document_versioning.py
--------------------------------------
Unit tests for the document versioning and diff engine.
"""

import pytest
from src.core.document_versioning import (
    tokenize_by_words,
    compute_myers_diff,
    generate_diff_blocks,
    calculate_retention_score,
    DiffOp,
)


class TestTokenization:
    """Test suite for text tokenization."""

    def test_tokenize_simple_sentence(self):
        """Verify simple sentences are tokenized correctly."""
        tokens = tokenize_by_words("Hello world.")
        assert tokens == ["Hello", " ", "world", "."]

    def test_tokenize_preserves_whitespace(self):
        """Verify whitespace is preserved as distinct tokens."""
        tokens = tokenize_by_words("A  B")
        assert tokens == ["A", "  ", "B"]


class TestMyersDiff:
    """Test suite for the Myers diff algorithm."""

    def test_identical_texts(self):
        """Verify identical texts produce only EQUAL operations."""
        t1 = ["A", " ", "B"]
        t2 = ["A", " ", "B"]
        edits = compute_myers_diff(t1, t2)

        assert all(op == DiffOp.EQUAL for op, _, _ in edits)

    def test_simple_insertion(self):
        """Verify a simple insertion is detected."""
        t1 = ["A", " ", "B"]
        t2 = ["A", " ", "new", " ", "B"]
        edits = compute_myers_diff(t1, t2)

        ops = [op for op, _, _ in edits]
        assert DiffOp.INSERT in ops

    def test_simple_deletion(self):
        """Verify a simple deletion is detected."""
        t1 = ["A", " ", "old", " ", "B"]
        t2 = ["A", " ", "B"]
        edits = compute_myers_diff(t1, t2)

        ops = [op for op, _, _ in edits]
        assert DiffOp.DELETE in ops

    def test_empty_to_text(self):
        """Verify diffing empty text to text produces all INSERTs."""
        edits = compute_myers_diff([], ["A", "B"])
        assert all(op == DiffOp.INSERT for op, _, _ in edits)


class TestDiffBlocks:
    """Test suite for high-level diff block generation."""

    def test_generate_blocks_groups_operations(self):
        """Verify consecutive identical operations are grouped."""
        text_v1 = "The quick brown fox."
        text_v2 = "The slow brown fox."

        blocks = generate_diff_blocks(text_v1, text_v2)

        # Should have EQUAL, REPLACE/DELETE+INSERT, EQUAL
        assert len(blocks) >= 3

    def test_retention_score_identical(self):
        """Verify identical texts have a retention score of 1.0."""
        blocks = generate_diff_blocks("Hello world", "Hello world")
        score = calculate_retention_score(blocks)
        assert score == 1.0

    def test_retention_score_half_deleted(self):
        """Verify deleting half the text results in ~0.5 retention."""
        blocks = generate_diff_blocks("A B C D", "A B")
        score = calculate_retention_score(blocks)
        assert 0.4 <= score <= 0.6
