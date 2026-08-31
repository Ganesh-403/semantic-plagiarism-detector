"""
tests/core/test_drift_detector.py
---------------------------------
Unit tests for the Intra-Document Style Drift and Change-Point Detection engine.
"""

import pytest
from src.core.style_drift_detector import (
    extract_sliding_window_features,
    compute_type_token_ratio,
    compute_yules_k,
)
from src.core.changepoint_analysis import detect_cusum_changepoints


class TestSlidingWindowFeatures:
    """Test suite for sliding window feature extraction."""

    def test_extract_features_basic(self):
        """Verify features are extracted for a basic text."""
        text = "This is a test sentence. " * 100
        features = extract_sliding_window_features(text, window_size=50, step_size=25)
        assert len(features) > 0
        assert "ttr" in features[0]
        assert "yules_k" in features[0]

    def test_short_text_single_window(self):
        """Verify short text results in a single window."""
        text = "Short text."
        features = extract_sliding_window_features(text, window_size=50, step_size=25)
        assert len(features) == 1

    def test_ttr_calculation(self):
        """Verify TTR calculation math."""
        words = ["the", "cat", "sat", "on", "the", "mat"]
        ttr = compute_type_token_ratio(words)
        assert ttr == 5 / 6  # 5 unique, 6 total


class TestChangepointDetection:
    """Test suite for CUSUM change-point detection."""

    def test_detect_changepoints_uniform(self):
        """Verify no change-points are detected in uniform text."""
        text = "The quick brown fox jumps over the lazy dog. " * 200
        features = extract_sliding_window_features(text, window_size=50, step_size=25)
        changepoints = detect_cusum_changepoints(features)
        # Uniform text should have very few or no significant change-points
        assert len(changepoints) < 3

    def test_detect_changepoints_spliced(self):
        """Verify change-points are detected in spliced multi-author text."""
        # Author 1: Simple, repetitive
        text_a = "The cat sat on the mat. The dog ran in the park. " * 100
        # Author 2: Complex, high variance
        text_b = (
            "Notwithstanding the aforementioned complexities, the epistemological framework necessitates a paradigm shift. "
            * 100
        )

        spliced_text = text_a + text_b
        features = extract_sliding_window_features(
            spliced_text, window_size=50, step_size=25
        )
        changepoints = detect_cusum_changepoints(features)

        # Should detect at least one major shift around the splice point
        assert len(changepoints) > 0
