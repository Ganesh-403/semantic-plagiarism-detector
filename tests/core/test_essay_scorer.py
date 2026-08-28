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
tests/core/test_essay_scorer.py
-------------------------------
Unit tests for Automated Essay Scoring and Analytic Trait Analysis.
"""

import pytest

from src.core.essay_scorer import DEFAULT_RUBRIC, score_essay
from src.core.trait_analyzer import (
    compute_argumentation_structure,
    compute_coherence_score,
    compute_lexical_complexity,
    extract_analytic_traits,
)


class TestTraitAnalyzer:
    """Test suite for analytic trait extraction."""

    def test_coherence_score_high_overlap(self):
        """Verify high coherence for sentences with high lexical overlap."""
        text = (
            "The cat sat on the mat. The cat sat on the rug. The cat sat on the floor."
        )
        score = compute_coherence_score(text)
        assert score > 0.5

    def test_coherence_score_low_overlap(self):
        """Verify low coherence for completely disjoint sentences."""
        text = "Apple banana cherry. Dog cat mouse. Quantum physics relativity."
        score = compute_coherence_score(text)
        assert score < 0.2

    def test_lexical_complexity_ttr(self):
        """Verify Type-Token Ratio calculation."""
        text = "the cat the cat the cat"
        complexity = compute_lexical_complexity(text)
        # 2 unique words, 6 total words -> TTR = 2/6 = 0.333
        assert complexity["ttr"] == pytest.approx(0.333, rel=1e-2)

    def test_lexical_complexity_academic_density(self):
        """Verify academic vocabulary density calculation."""
        text = "The hypothesis requires empirical evaluation and theoretical framework."
        complexity = compute_lexical_complexity(text)
        # "hypothesis", "empirical", "evaluation", "theoretical", "framework" = 5 words
        # Total words = 9
        assert complexity["academic_density"] > 0.5

    def test_argumentation_structure_markers(self):
        """Verify argumentation marker counting."""
        text = "However, the data shows X. Furthermore, Y is true. Therefore, Z."
        structure = compute_argumentation_structure(text)
        assert structure["contrast"] >= 1
        assert structure["addition"] >= 1
        assert structure["consequence"] >= 1


class TestEssayScorer:
    """Test suite for holistic essay scoring."""

    def test_score_essay_basic(self):
        """Verify basic essay scoring pipeline."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        result = score_essay(text)

        assert "final_grade" in result
        assert 0.0 <= result["final_grade"] <= 100.0
        assert "traits" in result
        assert "criterion_scores" in result

    def test_score_essay_empty_text(self):
        """Verify empty text results in a zero grade."""
        result = score_essay("")
        assert result["final_grade"] == 0.0

    def test_rubric_weight_normalization(self):
        """Verify rubric weights are normalized to sum to 1.0."""
        DEFAULT_RUBRIC.normalize_weights()
        total_weight = sum(c.weight for c in DEFAULT_RUBRIC.criteria)
        assert total_weight == pytest.approx(1.0, rel=1e-4)
