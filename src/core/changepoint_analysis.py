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
src/core/changepoint_analysis.py
--------------------------------
Bayesian and CUSUM Change-Point Detection for Stylometric Time-Series.

Analyzes the sequence of sliding-window stylometric features to identify
abrupt shifts in authorship. Uses Cumulative Sum (CUSUM) control charts
to detect statistically significant deviations from the document's baseline.
"""

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_baseline(
    features_list: List[Dict[str, float]], feature_key: str
) -> Tuple[float, float]:
    """Compute the mean and standard deviation for a specific feature across all windows."""
    values = [f[feature_key] for f in features_list if feature_key in f]
    if not values:
        return 0.0, 1.0
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance) if variance > 0 else 1.0
    return mean_val, std_dev


def detect_cusum_changepoints(
    features_list: List[Dict[str, float]],
    feature_keys: List[str] = None,
    threshold_multiplier: float = 2.0,
) -> List[Dict[str, Any]]:
    """Detect change-points using the CUSUM (Cumulative Sum) algorithm.

    The CUSUM algorithm tracks the cumulative sum of deviations from the mean.
    When the cumulative sum exceeds a threshold (typically k * std_dev),
    a change-point is flagged.

    Args:
        features_list: List of feature dictionaries from sliding windows.
        feature_keys: List of feature keys to monitor (e.g., 'ttr', 'yules_k').
        threshold_multiplier: Number of standard deviations for the threshold.

    Returns:
        List of dictionaries containing detected change-points and confidence scores.
    """
    if not features_list:
        return []

    if feature_keys is None:
        feature_keys = ["ttr", "yules_k", "sent_len_var"]

    changepoints = []

    for key in feature_keys:
        mean_val, std_dev = compute_baseline(features_list, key)
        if std_dev == 0:
            continue

        threshold = threshold_multiplier * std_dev
        cusum_pos = 0.0
        cusum_neg = 0.0

        for i, features in enumerate(features_list):
            val = features.get(key, mean_val)
            deviation = val - mean_val

            cusum_pos = max(0, cusum_pos + deviation - (std_dev * 0.5))
            cusum_neg = max(0, cusum_neg - deviation - (std_dev * 0.5))

            if cusum_pos > threshold:
                changepoints.append(
                    {
                        "window_index": i,
                        "start_word": features.get("start_word", 0),
                        "feature": key,
                        "direction": "increase",
                        "cusum_score": cusum_pos,
                        "confidence": min(1.0, cusum_pos / (threshold * 2)),
                    }
                )
                cusum_pos = 0.0  # Reset after detection

            if cusum_neg > threshold:
                changepoints.append(
                    {
                        "window_index": i,
                        "start_word": features.get("start_word", 0),
                        "feature": key,
                        "direction": "decrease",
                        "cusum_score": cusum_neg,
                        "confidence": min(1.0, cusum_neg / (threshold * 2)),
                    }
                )
                cusum_neg = 0.0

    # Sort by word position
    changepoints.sort(key=lambda x: x["start_word"])
    logger.info(
        "Detected %d change-points across %d features.",
        len(changepoints),
        len(feature_keys),
    )
    return changepoints
