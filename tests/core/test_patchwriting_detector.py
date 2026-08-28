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
tests/core/test_patchwriting_detector.py
----------------------------------------
Unit tests for the Mosaic Plagiarism (Patchwriting) Detection engine.
"""

import pytest

from src.core.patchwriting_detector import (
    compute_ngram_overlap,
    compute_syntactic_jaccard,
    detect_patchwriting,
)
from src.core.pos_normalizer import compute_pos_ngrams, extract_pos_sequence


class TestPOSNormalizer:
    """Test suite for POS extraction and normalization."""

    def test_extract_pos_sequence_basic(self):
        """Verify basic POS sequence extraction."""
        text = "The quick brown fox jumps."
        seq = extract_pos_sequence(text, use_nltk=False)
        # Heuristic: The(DET) quick(ADJ/NN) brown(ADJ/NN) fox(NN) jumps(VERB)
        assert len(seq) > 0
        assert "VERB" in seq or "NOUN" in seq

    def test_empty_text_returns_empty(self):
        """Verify empty text returns an empty sequence."""
        assert extract_pos_sequence("") == []

    def test_compute_ngrams(self):
        """Verify n-gram generation from POS sequences."""
        seq = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]
        trigrams = compute_pos_ngrams(seq, n=3)
        assert len(trigrams) == 4
        assert trigrams[0] == ("DET", "NOUN", "VERB")


class TestPatchwritingDetector:
    """Test suite for syntactic similarity scoring."""

    def test_identical_structure_high_overlap(self):
        """Verify identical syntactic structures produce high overlap."""
        text_a = "The cat sat on the mat."
        text_b = "A dog rested on the rug."  # Same DET-NOUN-VERB-ADP-DET-NOUN structure
        result = detect_patchwriting(text_a, text_b, n=3)
        assert result["ngram_overlap"] > 0.5

    def test_different_structure_low_overlap(self):
        """Verify structurally different texts produce low overlap."""
        text_a = "The cat sat on the mat."
        text_b = "Running quickly, the dog chased the ball."
        result = detect_patchwriting(text_a, text_b, n=3)
        assert result["ngram_overlap"] < 0.5

    def test_syntactic_jaccard(self):
        """Verify Jaccard similarity calculation."""
        seq_a = ["DET", "NOUN", "VERB"]
        seq_b = ["DET", "NOUN", "VERB"]
        assert compute_syntactic_jaccard(seq_a, seq_b) == 1.0

    def test_empty_texts(self):
        """Verify empty texts return 0.0 similarity."""
        result = detect_patchwriting("", "")
        assert result["ngram_overlap"] == 0.0
        assert result["is_patchwriting"] is False
