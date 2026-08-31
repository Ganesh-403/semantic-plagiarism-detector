# tests/core/test_similarity_mutation.py
import pytest
import numpy as np
from hypothesis import given, strategies as st
from src.core.similarity import (
    compute_cosine_similarity,
    flag_plagiarism,
    classify_severity,
    normalize_similarity_matrix
)
from src.core.config import PLAGIARISM_THRESHOLD, MEDIUM_THRESHOLD, HIGH_THRESHOLD

class TestSimilarityScoring:
    """Tests designed to kill mutants in similarity scoring."""
    
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=100),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_threshold_boundary_flags_correctly(self, scores, threshold):
        """Property: Scores >= threshold are flagged, < threshold are not.
        
        This test is designed to kill relational operator mutants.
        If `>=` is mutated to `>`, scores exactly at threshold will be missed.
        If `>=` is mutated to `<=`, scores below threshold will be incorrectly flagged.
        """
        for score in scores:
            result = flag_plagiarism(score, threshold)
            if score >= threshold:
                assert result is True, f"Score {score} >= {threshold} should be flagged"
            else:
                assert result is False, f"Score {score} < {threshold} should NOT be flagged"
    
    def test_severity_classification_exact_boundaries(self):
        """Test all severity boundaries.
        
        Kills boundary mutants: < → <=, > → >= mutations.
        """
        test_cases = [
            # (score, expected_severity)
            (0.59, "Medium"),      # Exactly at threshold
            (0.75, "Medium"),      # Exactly at medium boundary
            (0.89, "Medium"),      # Below high boundary
            (0.90, "High"),        # Exactly at high boundary
            (0.50, "Low"),         # Below threshold
            (0.95, "High"),        # Above high boundary
            (0.99, "High"),        # Near max
            (1.00, "High"),        # Max
        ]
        
        for score, expected in test_cases:
            result = classify_severity(score)
            assert result == expected, f"Score {score} expected {expected}, got {result}"
    
    def test_normalization_clamps_values(self):
        """Test clamping of out-of-range values.
        
        Kills min/max and clamp mutants.
        """
        from src.core.similarity import normalize_similarity
        
        assert normalize_similarity(1.5) == 1.0, "Should clamp >1.0 to 1.0"
        assert normalize_similarity(-0.5) == 0.0, "Should clamp <0.0 to 0.0"
        assert normalize_similarity(0.75) == 0.75, "Should preserve in-range values"
        assert normalize_similarity(np.nan) == 0.0, "Should handle NaN"
    
    def test_cosine_similarity_identity(self):
        """Test cosine similarity with identity vectors.
        
        Kills arithmetic operator mutants (+, -, *, / in similarity computation).
        """
        vec = np.array([1.0, 2.0, 3.0])
        sim = compute_cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-6), "Identity should produce 1.0"
        
        # Orthogonal vectors should produce 0.0
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert compute_cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)
    
    def test_cosine_similarity_negative_values(self):
        """Test cosine similarity with negative values.
        
        Kills sign-related mutants.
        """
        v1 = np.array([1.0, 1.0])
        v2 = np.array([-1.0, -1.0])
        sim = compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(-1.0, abs=1e-6), "Opposite vectors should be -1.0"
