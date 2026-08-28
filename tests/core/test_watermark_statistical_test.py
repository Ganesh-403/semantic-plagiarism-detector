"""
tests/core/test_watermark_statistical_test.py
---------------------------------------------
Unit tests for Watermark Hypothesis Testing, Z-score, p-value calculations,
confidence intervals, and False-Positive Rate (FPR) verification.
"""

import math
import numpy as np
import pytest

from src.core.watermark_statistical_test import (
    ConfidenceInterval,
    WatermarkStatisticalResult,
    WatermarkStatisticalTester,
    compute_watermark_statistics,
)


class TestWatermarkStatisticalTest:
    """Tests for statistical hypothesis testing of AI watermarks."""

    def test_z_score_calculation_accuracy(self):
        # N = 100, S_g = 60, gamma = 0.5
        # Expected = 50, Var = 100 * 0.5 * 0.5 = 25, StdDev = 5
        # Z = (60 - 50) / 5 = 2.0
        z = WatermarkStatisticalTester.compute_z_score(green_count=60, total_count=100, gamma=0.5)
        assert pytest.approx(z, 0.0001) == 2.0

        # N = 400, S_g = 250, gamma = 0.5
        # Expected = 200, StdDev = sqrt(400 * 0.25) = 10
        # Z = (250 - 200) / 10 = 5.0
        z2 = WatermarkStatisticalTester.compute_z_score(green_count=250, total_count=400, gamma=0.5)
        assert pytest.approx(z2, 0.0001) == 5.0

        # Gamma = 0.25, N = 192, S_g = 72
        # Expected = 48, Var = 192 * 0.25 * 0.75 = 36, StdDev = 6
        # Z = (72 - 48) / 6 = 4.0
        z3 = WatermarkStatisticalTester.compute_z_score(green_count=72, total_count=192, gamma=0.25)
        assert pytest.approx(z3, 0.0001) == 4.0

    def test_z_score_edge_cases(self):
        # Empty text / 0 tokens
        assert WatermarkStatisticalTester.compute_z_score(0, 0, 0.5) == 0.0
        # Negative count guard
        assert WatermarkStatisticalTester.compute_z_score(10, -5, 0.5) == 0.0
        # Invalid gamma
        assert WatermarkStatisticalTester.compute_z_score(10, 20, 0.0) == 0.0
        assert WatermarkStatisticalTester.compute_z_score(10, 20, 1.0) == 0.0

    def test_exact_binomial_p_value_calculation(self):
        # N = 10, S_g = 9, gamma = 0.5
        # P(X >= 9) = (10 + 1) / 1024 = 11 / 1024 = 0.0107421875
        p_val = WatermarkStatisticalTester.compute_exact_binomial_p_value(9, 10, gamma=0.5)
        assert pytest.approx(p_val, 0.00001) == 11 / 1024

        # N = 10, S_g = 10, gamma = 0.5 -> 1/1024
        p_val_10 = WatermarkStatisticalTester.compute_exact_binomial_p_value(10, 10, gamma=0.5)
        assert pytest.approx(p_val_10, 0.00001) == 1 / 1024

        # S_g = 0 -> p = 1.0
        assert WatermarkStatisticalTester.compute_exact_binomial_p_value(0, 10, 0.5) == 1.0

        # S_g > N -> 0.0
        assert WatermarkStatisticalTester.compute_exact_binomial_p_value(15, 10, 0.5) == 0.0

    def test_asymptotic_p_value_normal_distribution(self):
        # z = 0 -> p = 0.5
        assert pytest.approx(WatermarkStatisticalTester.compute_asymptotic_p_value(0.0), 0.0001) == 0.5
        # z = 1.95996 -> p approx 0.025
        assert pytest.approx(WatermarkStatisticalTester.compute_asymptotic_p_value(1.95996), 0.001) == 0.025
        # z = 4.0 -> p approx 3.167e-5
        assert pytest.approx(WatermarkStatisticalTester.compute_asymptotic_p_value(4.0), 1e-4) == 0.00003167

    def test_wilson_confidence_interval_bounds(self):
        # 95% CI for 60 successes out of 100
        ci = WatermarkStatisticalTester.compute_wilson_confidence_interval(60, 100, confidence_level=0.95)
        assert isinstance(ci, ConfidenceInterval)
        assert ci.point_estimate == 0.60
        assert 0.0 <= ci.lower_bound < ci.point_estimate < ci.upper_bound <= 1.0

        # For 95% CI on 60/100, Wilson interval is roughly [0.502, 0.690]
        assert pytest.approx(ci.lower_bound, 0.02) == 0.50
        assert pytest.approx(ci.upper_bound, 0.02) == 0.69

        # Higher confidence (99%) should yield a wider interval
        ci_99 = WatermarkStatisticalTester.compute_wilson_confidence_interval(60, 100, confidence_level=0.99)
        assert ci_99.lower_bound < ci.lower_bound
        assert ci_99.upper_bound > ci.upper_bound

    def test_watermarked_text_decision_and_confidence(self):
        tester = WatermarkStatisticalTester(gamma=0.5, z_threshold=4.0, significance_alpha=0.01)

        # Heavily biased watermark: 80 green tokens out of 100 (z = (80-50)/5 = 6.0)
        result = tester.test(green_tokens=80, total_tokens=100)

        assert isinstance(result, WatermarkStatisticalResult)
        assert result.z_score == 6.0
        assert result.p_value < 1e-5
        assert result.is_watermarked is True
        assert result.confidence_score > 99.9
        assert result.effect_size == pytest.approx(0.30, 0.001)

    def test_unwatermarked_text_decision(self):
        tester = WatermarkStatisticalTester(gamma=0.5, z_threshold=4.0, significance_alpha=0.01)

        # Natural unwatermarked text: 52 green tokens out of 100 (z = (52-50)/5 = 0.4)
        result = tester.test(green_tokens=52, total_tokens=100)

        assert result.z_score == 0.4
        assert result.is_watermarked is False
        assert result.p_value > 0.30

    def test_false_positive_rate_simulation(self):
        """Simulate unwatermarked null hypothesis data to verify empirical false-positive rate."""
        np.random.seed(42)
        n_trials = 2000
        n_tokens = 200
        gamma = 0.5
        alpha = 0.05
        z_threshold = 1.64485  # one-tailed critical value for alpha=0.05

        tester = WatermarkStatisticalTester(
            gamma=gamma, z_threshold=z_threshold, significance_alpha=alpha
        )

        # Generate binomial samples under H0: B(n_tokens, gamma)
        green_counts = np.random.binomial(n=n_tokens, p=gamma, size=n_trials)

        false_positives = 0
        for g_count in green_counts:
            res = tester.test(int(g_count), n_tokens)
            if res.is_watermarked:
                false_positives += 1

        empirical_fpr = false_positives / n_trials
        # Empirical FPR should be close to alpha (e.g. within [0.03, 0.07] for alpha=0.05 with 2000 samples)
        assert empirical_fpr <= 0.075
        assert empirical_fpr >= 0.025

    def test_theoretical_fpr_estimation(self):
        fpr_4 = WatermarkStatisticalTester.estimate_theoretical_fpr(4.0)
        assert pytest.approx(fpr_4, 1e-4) == 0.00003167

        fpr_0 = WatermarkStatisticalTester.estimate_theoretical_fpr(0.0)
        assert fpr_0 == 1.0

    def test_zero_total_tokens_edge_case(self):
        tester = WatermarkStatisticalTester(gamma=0.5)
        res = tester.test(0, 0)
        assert res.total_tokens == 0
        assert res.z_score == 0.0
        assert res.is_watermarked is False
        assert res.confidence_score == 0.0

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError, match="Gamma must be between"):
            WatermarkStatisticalTester(gamma=0.0)

        with pytest.raises(ValueError, match="z_threshold must be non-negative"):
            WatermarkStatisticalTester(z_threshold=-1.0)

        with pytest.raises(ValueError, match="significance_alpha must be in"):
            WatermarkStatisticalTester(significance_alpha=1.5)

        with pytest.raises(ValueError, match="confidence_level must be in"):
            WatermarkStatisticalTester(confidence_level=0.0)

    def test_convenience_function_compute_watermark_statistics(self):
        res = compute_watermark_statistics(green_tokens=70, total_tokens=100, gamma=0.5)
        assert res.z_score == 4.0
        assert res.is_watermarked is True
