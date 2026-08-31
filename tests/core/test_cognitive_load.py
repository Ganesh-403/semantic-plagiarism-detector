"""
tests/core/test_cognitive_load.py
---------------------------------
Unit tests for Cognitive Load and Readability Fingerprinting.
"""

import pytest
from src.core.readability_analyzer import (
    count_syllables,
    compute_flesch_kincaid,
    extract_readability_timeseries,
)
from src.core.cognitive_load_fingerprinter import (
    compute_cognitive_load_variance,
    analyze_cognitive_load,
)


class TestReadabilityAnalyzer:
    def test_count_syllables(self):
        assert count_syllables("hello") == 2
        assert count_syllables("world") == 1
        assert count_syllables("beautiful") == 3

    def test_compute_flesch_kincaid(self):
        text = "The cat sat on the mat. The dog ran in the park."
        fk = compute_flesch_kincaid(text)
        assert fk >= 0.0

    def test_extract_readability_timeseries(self):
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
        ts = extract_readability_timeseries(text, window_size=20)
        assert len(ts) > 0
        assert "fk_grade" in ts[0]


class TestCognitiveLoadFingerprinter:
    def test_compute_variance_uniform(self):
        # Uniform complexity (AI-like)
        ts = [{"fk_grade": 8.0, "cli": 10.0} for _ in range(10)]
        var = compute_cognitive_load_variance(ts)
        assert var["fk_variance"] == 0.0
        assert var["is_synthetic"] is True

    def test_compute_variance_variable(self):
        # Variable complexity (Human-like)
        ts = [{"fk_grade": i * 2.0, "cli": i * 3.0} for i in range(10)]
        var = compute_cognitive_load_variance(ts)
        assert var["fk_variance"] > 0.0
        assert var["is_synthetic"] is False

    def test_analyze_cognitive_load(self):
        ts = [{"fk_grade": 8.0, "cli": 10.0} for _ in range(10)]
        result = analyze_cognitive_load(ts)
        assert result["is_ai_generated"] is True
