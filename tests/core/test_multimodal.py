"""
tests/core/test_multimodal.py
-----------------------------
Unit tests for Multimodal Plagiarism Detection (Images and Equations).
"""

import pytest
import numpy as np
from src.core.image_phash_engine import compute_phash, compute_hamming_distance
from src.core.equation_ast_parser import (
    tokenize_latex,
    normalize_equation_ast,
    compute_tree_edit_distance,
)


class TestImagePHash:
    """Test suite for perceptual hashing."""

    def test_phash_identical_images(self):
        """Verify identical images produce identical hashes."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert h1 == h2

    def test_hamming_distance_identical(self):
        """Verify Hamming distance is 0 for identical hashes."""
        assert compute_hamming_distance("abcdef1234567890", "abcdef1234567890") == 0


class TestEquationAST:
    """Test suite for equation AST parsing and comparison."""

    def test_tokenize_latex(self):
        """Verify LaTeX tokenization."""
        tokens = tokenize_latex("x^2 + y^2 = z^2")
        assert "x" in tokens
        assert "^" in tokens
        assert "2" in tokens

    def test_normalize_equation_ast(self):
        """Verify variable renaming normalization."""
        tokens = ["x", "+", "y", "=", "z"]
        norm = normalize_equation_ast(tokens)
        # Variables should be replaced with VAR_1, VAR_2, etc.
        assert "VAR_1" in norm
        assert "VAR_2" in norm
        assert "+" in norm

    def test_tree_edit_distance_identical(self):
        """Verify edit distance is 0 for identical sequences."""
        seq = ["VAR_1", "+", "VAR_2"]
        assert compute_tree_edit_distance(seq, seq) == 0
