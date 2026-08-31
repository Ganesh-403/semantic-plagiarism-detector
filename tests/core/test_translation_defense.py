"""
tests/core/test_translation_defense.py
--------------------------------------
Unit tests for Cross-Lingual Back-Translation Attack Simulation and Defense.
"""

import pytest
from src.core.back_translation_simulator import (
    simulate_back_translation,
    generate_adversarial_batch,
)
from src.core.translation_invariance_scorer import (
    compute_lexical_drift,
    compute_structural_variance,
    score_translation_invariance,
)


class TestBackTranslationSimulator:
    """Test suite for back-translation simulation."""

    def test_simulate_back_translation_drift(self):
        """Verify back-translation introduces semantic drift."""
        text = "The important use of this method is good."
        drifted = simulate_back_translation(text, drift_probability=1.0)
        assert drifted != text
        assert "important" not in drifted.lower() or "crucial" in drifted.lower()

    def test_simulate_back_translation_no_drift(self):
        """Verify 0.0 drift probability returns original text."""
        text = "The quick brown fox."
        drifted = simulate_back_translation(text, drift_probability=0.0)
        assert drifted == text

    def test_generate_adversarial_batch(self):
        """Verify batch generation creates correct number of variants."""
        texts = ["Text one.", "Text two."]
        batch = generate_adversarial_batch(texts, num_variants=3, drift_probability=0.5)
        assert len(batch) == 2
        assert len(batch["Text one."]) == 3


class TestTranslationInvarianceScorer:
    """Test suite for translation invariance scoring."""

    def test_lexical_drift_identical(self):
        """Verify lexical drift is 0.0 for identical texts."""
        text = "The cat sat on the mat."
        drift = compute_lexical_drift(text, text)
        assert drift == 0.0

    def test_lexical_drift_different(self):
        """Verify lexical drift is high for completely different texts."""
        text_a = "The cat sat on the mat."
        text_b = "Dogs run in the park."
        drift = compute_lexical_drift(text_a, text_b)
        assert drift > 0.8

    def test_structural_variance_identical(self):
        """Verify structural variance is 0.0 for identical texts."""
        text = "Sentence one. Sentence two."
        var = compute_structural_variance(text, text)
        assert var == 0.0

    def test_score_translation_invariance_obfuscated(self):
        """Verify heavily drifted text is flagged as obfuscated."""
        text_a = "The important method is good."
        text_b = "Crucial technique is excellent."
        result = score_translation_invariance(text_a, text_b)
        assert result["is_obfuscated"] is True
        assert result["invariance_score"] < 0.6
