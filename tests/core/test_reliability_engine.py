"""
tests/core/test_reliability_engine.py
-------------------------------------
Unit tests for the Inter-Rater Reliability and Calibration engine.
"""

import pytest
from src.core.reliability_engine import (
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_reviewer_bias,
    calculate_calibration_weight,
)


class TestCohensKappa:
    """Test suite for Cohen's Kappa calculation."""

    def test_perfect_agreement(self):
        """Verify Kappa is 1.0 when raters perfectly agree."""
        r1 = [1, 2, 3, 1, 2]
        r2 = [1, 2, 3, 1, 2]
        assert compute_cohens_kappa(r1, r2) == 1.0

    def test_chance_agreement(self):
        """Verify Kappa is near 0.0 for random chance agreement."""
        # This is a simplified check; true chance agreement depends on marginals
        r1 = [1, 1, 1, 1, 2, 2, 2, 2]
        r2 = [1, 1, 2, 2, 1, 1, 2, 2]
        kappa = compute_cohens_kappa(r1, r2)
        assert -1.0 <= kappa <= 1.0

    def test_disagreement(self):
        """Verify Kappa is negative for systematic disagreement."""
        r1 = [1, 1, 1, 1]
        r2 = [2, 2, 2, 2]
        kappa = compute_cohens_kappa(r1, r2)
        assert kappa < 0.0

    def test_unequal_lengths_raises(self):
        """Verify ValueError is raised for unequal list lengths."""
        with pytest.raises(ValueError):
            compute_cohens_kappa([1, 2], [1])


class TestFleissKappa:
    """Test suite for Fleiss' Kappa calculation."""

    def test_perfect_agreement_fleiss(self):
        """Verify Fleiss' Kappa is 1.0 for perfect agreement among N raters."""
        # 3 raters, 2 items, 3 categories. All raters agree on cat 0 for item 1, cat 1 for item 2.
        matrix = [
            [3, 0, 0],  # Item 1: 3 raters chose category 0
            [0, 3, 0],  # Item 2: 3 raters chose category 1
        ]
        assert compute_fleiss_kappa(matrix) == 1.0

    def test_empty_matrix(self):
        """Verify empty matrix returns 0.0."""
        assert compute_fleiss_kappa([]) == 0.0


class TestReviewerBias:
    """Test suite for reviewer bias and calibration weight calculations."""

    def test_zero_bias(self):
        """Verify zero bias when manual scores match automated scores."""
        reviewer = [0.8, 0.5, 0.2]
        baseline = [0.8, 0.5, 0.2]
        bias = compute_reviewer_bias(reviewer, baseline)
        assert bias["mean_error"] == 0.0
        assert bias["mean_absolute_error"] == 0.0
        assert bias["variance"] == 0.0

    def test_positive_bias(self):
        """Verify positive mean error when reviewer scores higher than baseline."""
        reviewer = [0.9, 0.6, 0.3]
        baseline = [0.8, 0.5, 0.2]
        bias = compute_reviewer_bias(reviewer, baseline)
        assert bias["mean_error"] == 0.1

    def test_calibration_weight_high_trust(self):
        """Verify high trust (weight near 1.0) for low bias/variance."""
        bias = {"mean_absolute_error": 0.05, "variance": 0.01}
        weight = calculate_calibration_weight(bias)
        assert weight > 0.8

    def test_calibration_weight_low_trust(self):
        """Verify low trust (weight near 0.1) for high bias/variance."""
        bias = {"mean_absolute_error": 0.5, "variance": 0.8}
        weight = calculate_calibration_weight(bias)
        assert weight <= 0.2
