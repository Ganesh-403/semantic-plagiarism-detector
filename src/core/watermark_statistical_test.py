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
src/core/watermark_statistical_test.py
--------------------------------------
Statistical Hypothesis Testing for AI Watermark Verification.

Computes p-values and z-scores to determine if the proportion of "green list"
tokens in a text is statistically significantly higher than expected by chance,
indicating the presence of an AI watermark.
"""

import logging
import math
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
    if sample_size == 0 or expected_ratio <= 0 or expected_ratio >= 1:
        return 0.0

    # Standard deviation of the binomial distribution
    std_dev = math.sqrt((expected_ratio * (1 - expected_ratio)) / sample_size)

    if std_dev == 0:
        return 0.0

    z_score = (observed_ratio - expected_ratio) / std_dev
    return round(z_score, 4)


def compute_p_value(z_score: float) -> float:
    """Compute the one-tailed p-value from a z-score.

    Uses the error function (erf) to approximate the cumulative distribution
    function (CDF) of the standard normal distribution, avoiding the need
    for external libraries like scipy.

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
