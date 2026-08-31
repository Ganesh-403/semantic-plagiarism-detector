"""
src/core/watermark_statistical_test.py
--------------------------------------
Statistical Hypothesis Testing Engine for AI Text Watermark Verification.

Computes Kirchenbauer z-scores, exact one-sided Binomial p-values, asymptotic
normal p-values, Wilson score confidence intervals, and hypothesis decisions for
statistical AI text watermarks (e.g., Maryland watermarking scheme).
"""

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


@dataclass
class ConfidenceInterval:
    """Confidence interval for the observed green-token proportion."""

    confidence_level: float  # e.g., 0.95 for 95% CI
    lower_bound: float
    upper_bound: float
    point_estimate: float


@dataclass
class WatermarkStatisticalResult:
    """Complete statistical test result for watermark presence verification."""

    total_tokens: int
    green_tokens: int
    red_tokens: int
    observed_green_ratio: float
    expected_green_ratio: float
    z_score: float
    p_value: float
    exact_p_value: float
    asymptotic_p_value: float
    confidence_interval: ConfidenceInterval
    is_watermarked: bool
    confidence_score: float  # Scale 0.0 to 100.0%
    significance_alpha: float
    z_threshold: float
    effect_size: float  # observed_ratio - expected_ratio


class WatermarkStatisticalTester:
    """Statistical hypothesis tester for AI watermark verification."""

    def __init__(
        self,
        gamma: float = 0.5,
        z_threshold: float = 4.0,
        significance_alpha: float = 0.01,
        confidence_level: float = 0.95,
    ):
        """Initialize the statistical tester.

        Args:
            gamma: Expected green list fraction under H0 (0 < gamma < 1).
            z_threshold: Minimum z-score to declare watermark presence (default: 4.0).
            significance_alpha: Significance level threshold for p-value (default: 0.01).
            confidence_level: Confidence level for proportion interval estimation (default: 0.95).
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"Gamma must be between 0.0 and 1.0 (exclusive), got {gamma}")
        if z_threshold < 0.0:
            raise ValueError(f"z_threshold must be non-negative, got {z_threshold}")
        if not (0.0 < significance_alpha < 1.0):
            raise ValueError(f"significance_alpha must be in (0, 1), got {significance_alpha}")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

        self.gamma = gamma
        self.z_threshold = z_threshold
        self.significance_alpha = significance_alpha
        self.confidence_level = confidence_level

    @staticmethod
    def compute_z_score(green_count: int, total_count: int, gamma: float = 0.5) -> float:
        """Compute the Kirchenbauer z-score test statistic.

        Formula: z = (S_g - gamma * N) / sqrt(gamma * (1 - gamma) * N)

        Args:
            green_count: Number of green list tokens observed (S_g).
            total_count: Total number of scored tokens (N).
            gamma: Expected proportion under null hypothesis.

        Returns:
            Calculated z-score (0.0 if total_count == 0).
        """
        if total_count <= 0 or not (0.0 < gamma < 1.0):
            return 0.0

        expected_green = gamma * total_count
        variance = gamma * (1.0 - gamma) * total_count
        std_dev = math.sqrt(variance)

        if std_dev == 0.0:
            return 0.0

        z = (green_count - expected_green) / std_dev
        return round(z, 6)

    @staticmethod
    def compute_asymptotic_p_value(z_score: float) -> float:
        """Compute asymptotic one-sided p-value from z-score using standard normal survival function.

        p = 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
        """
        if z_score <= -38.0:
            return 1.0
        if z_score >= 38.0:
            return 0.0

        p = 0.5 * math.erfc(z_score / math.sqrt(2.0))
        return float(max(0.0, min(1.0, p)))

    @staticmethod
    def compute_exact_binomial_p_value(
        green_count: int, total_count: int, gamma: float = 0.5
    ) -> float:
        """Compute exact one-sided Binomial test p-value: P(X >= green_count | N, gamma).

        Uses scipy.stats.binom.sf or exact formula with log-combinations.
        """
        if total_count <= 0:
            return 1.0
        if green_count <= 0:
            return 1.0
        if green_count > total_count:
            return 0.0

        if scipy_stats is not None:
            # Survival function sf(k) = P(X > k) = P(X >= k + 1), so sf(green_count - 1) gives P(X >= green_count)
            return float(scipy_stats.binom.sf(green_count - 1, total_count, gamma))

        # Fallback exact binomial summation using log factorials
        def log_comb(n: int, k: int) -> float:
            return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

        total_prob = 0.0
        for k in range(green_count, total_count + 1):
            log_prob = (
                log_comb(total_count, k)
                + (k * math.log(gamma))
                + ((total_count - k) * math.log(1.0 - gamma))
            )
            total_prob += math.exp(log_prob)

        return float(max(0.0, min(1.0, total_prob)))

    @classmethod
    def compute_wilson_confidence_interval(
        cls,
        successes: int,
        total: int,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """Compute Wilson score interval for binomial proportion.

        Args:
            successes: Number of green tokens (k).
            total: Total scored tokens (n).
            confidence_level: Desired confidence level (e.g., 0.95).

        Returns:
            ConfidenceInterval with lower_bound, upper_bound, and point_estimate.
        """
        if total <= 0:
            return ConfidenceInterval(
                confidence_level=confidence_level,
                lower_bound=0.0,
                upper_bound=1.0,
                point_estimate=0.0,
            )

        p_hat = successes / total

        # Critical value z_crit for standard normal distribution
        alpha = 1.0 - confidence_level
        if scipy_stats is not None:
            z_crit = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
        else:
            # High-precision approximation for standard normal quantile
            if abs(confidence_level - 0.95) < 0.005:
                z_crit = 1.959963984540054
            elif abs(confidence_level - 0.99) < 0.005:
                z_crit = 2.5758293035489004
            elif abs(confidence_level - 0.90) < 0.005:
                z_crit = 1.6448536269514722
            else:
                z_crit = 1.95996

        z_sq = z_crit * z_crit
        denominator = 1.0 + (z_sq / total)
        center = (p_hat + (z_sq / (2.0 * total))) / denominator
        margin = (
            z_crit
            * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z_sq / (4.0 * total * total)))
        ) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return ConfidenceInterval(
            confidence_level=confidence_level,
            lower_bound=round(lower, 6),
            upper_bound=round(upper, 6),
            point_estimate=round(p_hat, 6),
        )

    @staticmethod
    def estimate_theoretical_fpr(z_threshold: float) -> float:
        """Estimate theoretical false-positive rate under the null hypothesis."""
        if z_threshold <= 0:
            return 1.0
        return WatermarkStatisticalTester.compute_asymptotic_p_value(z_threshold)

    def test(
        self,
        green_tokens: int,
        total_tokens: int,
        gamma: Optional[float] = None,
        z_threshold: Optional[float] = None,
        significance_alpha: Optional[float] = None,
        confidence_level: Optional[float] = None,
    ) -> WatermarkStatisticalResult:
        """Perform full statistical hypothesis test on watermark green token counts.

        Args:
            green_tokens: Observed count of green tokens.
            total_tokens: Total count of evaluated tokens.
            gamma: Optional override for expected green ratio.
            z_threshold: Optional override for z-score decision threshold.
            significance_alpha: Optional override for alpha threshold.
            confidence_level: Optional override for CI calculation.

        Returns:
            WatermarkStatisticalResult containing z-score, p-values, CI, and detection decision.
        """
        g = self.gamma if gamma is None else gamma
        z_thresh = self.z_threshold if z_threshold is None else z_threshold
        alpha = self.significance_alpha if significance_alpha is None else significance_alpha
        conf_lvl = self.confidence_level if confidence_level is None else confidence_level

        if total_tokens <= 0:
            ci = ConfidenceInterval(
                confidence_level=conf_lvl,
                lower_bound=0.0,
                upper_bound=1.0,
                point_estimate=0.0,
            )
            return WatermarkStatisticalResult(
                total_tokens=0,
                green_tokens=0,
                red_tokens=0,
                observed_green_ratio=0.0,
                expected_green_ratio=g,
                z_score=0.0,
                p_value=1.0,
                exact_p_value=1.0,
                asymptotic_p_value=1.0,
                confidence_interval=ci,
                is_watermarked=False,
                confidence_score=0.0,
                significance_alpha=alpha,
                z_threshold=z_thresh,
                effect_size=0.0,
            )

        red_tokens = total_tokens - green_tokens
        observed_ratio = green_tokens / total_tokens

        z_score = self.compute_z_score(green_tokens, total_tokens, g)
        asymptotic_p = self.compute_asymptotic_p_value(z_score)
        exact_p = self.compute_exact_binomial_p_value(green_tokens, total_tokens, g)

        # Primary p-value used: exact binomial for N <= 2000, asymptotic for very large N
        primary_p = exact_p if total_tokens <= 2000 else asymptotic_p

        ci = self.compute_wilson_confidence_interval(
            green_tokens, total_tokens, confidence_level=conf_lvl
        )

        # Detection decision: both z >= z_thresh and p_value <= alpha
        is_watermarked = (z_score >= z_thresh) and (primary_p <= alpha)

        # Confidence percentage (0.0 to 100.0%) based on 1 - p_value
        confidence_score = round(max(0.0, min(100.0, (1.0 - primary_p) * 100.0)), 4)

        effect_size = round(observed_ratio - g, 6)

        return WatermarkStatisticalResult(
            total_tokens=total_tokens,
            green_tokens=green_tokens,
            red_tokens=red_tokens,
            observed_green_ratio=round(observed_ratio, 6),
            expected_green_ratio=g,
            z_score=z_score,
            p_value=primary_p,
            exact_p_value=exact_p,
            asymptotic_p_value=asymptotic_p,
            confidence_interval=ci,
            is_watermarked=is_watermarked,
            confidence_score=confidence_score,
            significance_alpha=alpha,
            z_threshold=z_thresh,
            effect_size=effect_size,
        )


def compute_watermark_statistics(
    green_tokens: int,
    total_tokens: int,
    gamma: float = 0.5,
    z_threshold: float = 4.0,
    significance_alpha: float = 0.01,
    confidence_level: float = 0.95,
) -> WatermarkStatisticalResult:
    """Convenience helper to compute watermark statistical verification test."""
    tester = WatermarkStatisticalTester(
        gamma=gamma,
        z_threshold=z_threshold,
        significance_alpha=significance_alpha,
        confidence_level=confidence_level,
    )
    return tester.test(green_tokens, total_tokens)


def compute_z_score(
    observed_ratio: float, expected_ratio: float, sample_size: int
) -> float:
    """Compute the z-score for the observed green list ratio.

    Uses the normal approximation to the binomial distribution.

    Args:
        observed_ratio: The observed proportion of green list tokens.
        expected_ratio: The expected proportion under the null hypothesis.
        sample_size: Total number of tokens in the text.

    Returns:
        The computed z-score.
    """
    if sample_size <= 0 or expected_ratio <= 0 or expected_ratio >= 1:
        return 0.0

    # Standard deviation of the binomial distribution
    std_dev = math.sqrt((expected_ratio * (1 - expected_ratio)) / sample_size)

    if std_dev == 0:
        return 0.0

    z_score = (observed_ratio - expected_ratio) / std_dev
    return round(z_score, 4)


def compute_p_value(z_score: float) -> float:
    """Compute the one-tailed p-value from a z-score.

    Args:
        z_score: The computed z-score.

    Returns:
        The one-tailed p-value.
    """
    # P(Z > z) = 0.5 * (1 - erf(z / sqrt(2)))
    p_value = 0.5 * (1 - math.erf(z_score / math.sqrt(2)))
    return round(p_value, 6)


def verify_watermark_presence(
    token_metrics: Dict[str, Any],
    expected_green_ratio: float = 0.5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Verify the presence of an AI watermark using statistical testing.

    Args:
        token_metrics: Metrics from ai_watermark_extractor.
        expected_green_ratio: Expected ratio of green list tokens under H0.
        significance_level: Alpha level for hypothesis testing.

    Returns:
        Dictionary containing z-score, p-value, and verification result.
    """
    observed_ratio = token_metrics.get("green_list_ratio", 0.0)
    sample_size = token_metrics.get("total_tokens", 0)

    z_score = compute_z_score(observed_ratio, expected_green_ratio, sample_size)
    p_value = compute_p_value(z_score)

    # Reject null hypothesis if p-value < alpha
    is_watermarked = p_value < significance_level

    return {
        "z_score": z_score,
        "p_value": p_value,
        "is_watermarked": is_watermarked,
        "confidence_level": 1.0 - significance_level,
        "observed_ratio": observed_ratio,
    }
