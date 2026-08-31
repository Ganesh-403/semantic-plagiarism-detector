"""
tests/core/test_stylometry.py
-----------------------------
Comprehensive unit tests for the Stylometric Authorship Attribution engine.
"""

import pytest
import math
from src.core.stylometry_engine import (
    extract_stylometric_profile,
    compute_type_token_ratio,
    compute_yules_k,
    compute_sentence_stats,
    StylometricProfile,
)
from src.db.stylometry_profiles_db import (
    initialize_stylometry_db,
    save_profile,
    get_user_baseline,
)


class TestStylometryMath:
    """Test suite for stylometric feature extraction mathematics."""

    def test_type_token_ratio_unique_words(self):
        """Verify TTR is 1.0 when all words are unique."""
        words = ["the", "quick", "brown", "fox"]
        assert compute_type_token_ratio(words) == 1.0

    def test_type_token_ratio_repeated_words(self):
        """Verify TTR decreases with repeated words."""
        words = ["the", "the", "the", "the"]
        assert compute_type_token_ratio(words) == 0.25

    def test_yules_k_calculation(self):
        """Verify Yule's K is computed without errors."""
        words = ["the", "cat", "sat", "on", "the", "mat"]
        k = compute_yules_k(words)
        assert k >= 0.0

    def test_sentence_stats_variance(self):
        """Verify sentence length variance is calculated correctly."""
        sentences = ["One two.", "One two three four.", "One."]
        mean, var = compute_sentence_stats(sentences)
        assert mean == 2.0  # (2 + 4 + 1) / 3
        assert var > 0.0

    def test_extract_profile_empty_text(self):
        """Verify empty text returns a zeroed profile."""
        profile = extract_stylometric_profile("")
        assert profile.type_token_ratio == 0.0
        assert profile.yules_k == 0.0

    def test_extract_profile_valid_text(self):
        """Verify a valid text produces a populated profile."""
        text = "This is a test. It has multiple sentences. The vocabulary is somewhat diverse."
        profile = extract_stylometric_profile(text)
        assert profile.type_token_ratio > 0.0
        assert profile.avg_sentence_length > 0.0


class TestStylometryDeviation:
    """Test suite for baseline deviation scoring."""

    def test_deviation_score_identical(self):
        """Verify deviation score is 0.0 for identical profiles."""
        p1 = StylometricProfile(0.5, 10.0, 5.0, 4.5, 12.0, 100.0)
        p2 = StylometricProfile(0.5, 10.0, 5.0, 4.5, 12.0, 100.0)
        assert p1.compute_deviation_score(p2) == 0.0

    def test_deviation_score_different(self):
        """Verify deviation score is > 0.0 for different profiles."""
        p1 = StylometricProfile(0.5, 10.0, 5.0, 4.5, 12.0, 100.0)
        p2 = StylometricProfile(0.8, 20.0, 15.0, 6.5, 5.0, 50.0)
        assert p1.compute_deviation_score(p2) > 0.0


class TestStylometryDB:
    """Test suite for the stylometry profiles database."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "test_stylo.db"
        initialize_stylometry_db(db_path)
        return db_path

    def test_save_and_get_baseline(self, temp_db):
        """Verify profiles can be saved and averaged into a baseline."""
        p1 = StylometricProfile(0.4, 10.0, 5.0, 4.0, 10.0, 80.0)
        p2 = StylometricProfile(0.6, 20.0, 15.0, 6.0, 20.0, 120.0)

        save_profile("user_1", "doc_1", p1, db_path=temp_db)
        save_profile("user_1", "doc_2", p2, db_path=temp_db)

        baseline = get_user_baseline("user_1", db_path=temp_db)
        assert baseline is not None
        assert baseline.type_token_ratio == 0.5  # Average of 0.4 and 0.6
        assert baseline.avg_sentence_length == 15.0  # Average of 10 and 20

    def test_get_baseline_no_history(self, temp_db):
        """Verify baseline returns None for a user with no history."""
        baseline = get_user_baseline("unknown_user", db_path=temp_db)
        assert baseline is None
