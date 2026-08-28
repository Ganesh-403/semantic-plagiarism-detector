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
tests/core/test_multimodal.py
-----------------------------
Unit tests for Multimodal Plagiarism Detection (Images and Equations).
"""

import numpy as np
import pytest

from src.core.equation_ast_parser import (
    compute_tree_edit_distance,
    normalize_equation_ast,
    tokenize_latex,
)
from src.core.image_phash_engine import compute_hamming_distance, compute_phash


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
