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
src/core/reliability_engine.py
------------------------------
Statistical engine for calculating Inter-Rater Reliability (IRR) metrics.

Computes Fleiss' Kappa, Cohen's Kappa, and Krippendorff's Alpha to measure
agreement among multiple instructors or TAs reviewing plagiarism flags.
These metrics are used to calibrate reviewer bias and weight manual
overrides against automated detection scores.
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's Kappa for two raters.

    Measures the agreement between two raters evaluating a set of items,
    correcting for agreement expected by chance.

    Args:
        rater1: List of categorical ratings from rater 1.
        rater2: List of categorical ratings from rater 2.

    Returns:
        Cohen's Kappa score between -1.0 and 1.0.

    Raises:
        ValueError: If the input lists are of different lengths or empty.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must be of equal length.")
    if not rater1:
        raise ValueError("Rater lists cannot be empty.")

    n = len(rater1)
    categories = list(set(rater1 + rater2))

    # Build confusion matrix
    matrix = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for i in range(n):
        matrix[rater1[i]][rater2[i]] += 1

    # Calculate observed agreement (Po)
    po = sum(matrix[c][c] for c in categories) / n

    # Calculate expected agreement by chance (Pe)
    pe = 0.0
    for c in categories:
        p1 = sum(matrix[c][c2] for c2 in categories) / n
        p2 = sum(matrix[c1][c] for c1 in categories) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def compute_fleiss_kappa(ratings_matrix: list[list[int]]) -> float:
    """Compute Fleiss' Kappa for multiple raters (N > 2).

    Measures the reliability of agreement between a fixed number of raters
    when assigning categorical ratings to a set of items.

    Args:
        ratings_matrix: A list of N items, where each item is a list of
                        counts representing how many raters assigned each
                        category. E.g., [[2, 0, 1], [1, 2, 0]] means for
                        item 1, 2 raters chose cat 0, 0 chose cat 1, 1 chose cat 2.

    Returns:
        Fleiss' Kappa score between -1.0 and 1.0.
    """
    if not ratings_matrix:
        return 0.0

    n_items = len(ratings_matrix)
    n_raters = sum(ratings_matrix[0])
    n_categories = len(ratings_matrix[0])

    if n_raters <= 1:
        return 1.0  # Perfect agreement if only 1 rater

    # Calculate P_j (proportion of all assignments to category j)
    p_j = []
    for j in range(n_categories):
        sum_j = sum(ratings_matrix[i][j] for i in range(n_items))
        p_j.append(sum_j / (n_items * n_raters))

    # Calculate P_i (extent of agreement for each item i)
    p_i = []
    for i in range(n_items):
        sum_squares = sum(x**2 for x in ratings_matrix[i])
        p_i_val = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
        p_i.append(p_i_val)

    # Mean of P_i (P_bar)
    p_bar = sum(p_i) / n_items

    # Expected agreement by chance (P_e)
    p_e = sum(p**2 for p in p_j)

    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0

    kappa = (p_bar - p_e) / (1.0 - p_e)
    return round(kappa, 4)


def compute_reviewer_bias(
    reviewer_scores: list[float], baseline_scores: list[float]
) -> dict[str, float]:
    """Compute the bias and variance of a reviewer against an automated baseline.

    Args:
        reviewer_scores: List of manual override scores from the reviewer.
        baseline_scores: List of automated similarity scores for the same items.

    Returns:
        Dictionary containing 'mean_error', 'mean_absolute_error', and 'variance'.
    """
    if len(reviewer_scores) != len(baseline_scores) or not reviewer_scores:
        return {"mean_error": 0.0, "mean_absolute_error": 0.0, "variance": 0.0}

    errors = [r - b for r, b in zip(reviewer_scores, baseline_scores)]
    abs_errors = [abs(e) for e in errors]

    mean_error = sum(errors) / len(errors)
    mae = sum(abs_errors) / len(abs_errors)
    variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)

    return {
        "mean_error": round(mean_error, 4),
        "mean_absolute_error": round(mae, 4),
        "variance": round(variance, 4),
    }


def calculate_calibration_weight(bias_metrics: dict[str, float]) -> float:
    """Calculate a calibration weight (0.0 to 1.0) based on reviewer bias.

    A reviewer with low variance and low mean absolute error gets a weight
    close to 1.0. High bias or high variance reduces their weight, meaning
    their manual overrides will be blended more heavily with the automated score.
    """
    mae = bias_metrics.get("mean_absolute_error", 1.0)
    variance = bias_metrics.get("variance", 1.0)

    # Simple inverse weighting: higher error/variance = lower weight
    # Scaled using a sigmoid-like function to keep it between 0.1 and 1.0
    penalty = (mae * 2.0) + (variance * 5.0)
    weight = 1.0 / (1.0 + penalty)

    # Clamp between 0.1 (minimum trust) and 1.0 (full trust)
    return round(max(0.1, min(1.0, weight)), 4)


# semantic-plagiarism-detector/src/core/reliability_engine.py

from typing import Any, Dict, List

import numpy as np


class ReliabilityEngine:
    """
    Statistical engine for calculating Inter-Rater Reliability (IRR) metrics
    such as Fleiss' Kappa, Cohen's Kappa, and reviewer calibration weights.
    """

    @staticmethod
    def compute_cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
        """Computes Cohen's Kappa for agreement between two reviewers."""
        if len(rater1) != len(rater2) or not rater1:
            return 0.0

        r1 = np.array(rater1)
        r2 = np.array(rater2)
        n = len(r1)

        # Observed agreement
        obs_agreement = np.sum(r1 == r2) / n

        # Expected agreement by chance
        classes = np.unique(np.concatenate([r1, r2]))
        p_e = 0.0
        for c in classes:
            p1 = np.sum(r1 == c) / n
            p2 = np.sum(r2 == c) / n
            p_e += p1 * p2

        if p_e == 1.0:
            return 1.0

        kappa = (obs_agreement - p_e) / (1.0 - p_e)
        return float(kappa)

    @staticmethod
    def compute_fleiss_kappa(ratings_matrix: list[list[int]]) -> float:
        """
        Computes Fleiss' Kappa for multi-rater agreement across review committees.
        Ratings matrix shape: N subjects x K categories, where each cell is the number of raters who assigned that category.
        """
        mat = np.array(ratings_matrix)
        if mat.size == 0:
            return 0.0

        n_subjects, n_categories = mat.shape
        # Total number of raters per subject (assumed constant)
        N = np.sum(mat[0])
        if N <= 1:
            return 0.0

        # Proportion of all assignments assigned to the j-th category
        p_j = np.sum(mat, axis=0) / (n_subjects * N)

        # Extent to which raters agree for the i-th subject
        P_i = (np.sum(mat**2, axis=1) - N) / (N * (N - 1))

        # Mean of P_i across all subjects
        P_bar = np.mean(P_i)

        # Sum of squares of proportions
        P_e_bar = np.sum(p_j**2)

        if P_e_bar == 1.0:
            return 1.0

        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
        return float(kappa)

    @staticmethod
    def compute_reviewer_bias_weights(
        historical_overrides: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Computes calibration bias weights for reviewers based on historical deviation from committee consensus.
        """
        weights = {}
        for record in historical_overrides:
            reviewer_id = record.get("reviewer_id")
            deviation = record.get("consensus_deviation", 0.0)
            # Higher deviation from consensus reduces override confidence weight
            weight = max(0.1, 1.0 - abs(float(deviation)))
            weights[reviewer_id] = round(weight, 3)

        return weights
