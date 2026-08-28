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
test_softmax_lexical_normalization.py
---------------------------------------
Exhaustive unit test suite for soft-max / sigmoidal normalization of lexical similarity scores (#924).
Verifies mathematical boundary conditions (0.0, 0.5, 1.0), non-linear score transformations,
strict [0.0, 1.0] output boundedness, edge cases, matrix vectorization, and property-based invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.lexical_similarity import (
    scale_lexical_matrix,
    scale_lexical_score,
    softmax_normalize_scores,
)


def test_scale_lexical_score_boundary_zero():
    """Verify input 0.0 maps strictly to 0.0."""
    res = scale_lexical_score(0.0)
    assert res == 0.0
    assert isinstance(res, float)


def test_scale_lexical_score_boundary_half():
    """Verify input 0.5 maps strictly to 0.5."""
    res = scale_lexical_score(0.5)
    assert res == pytest.approx(0.5, abs=1e-6)
    assert isinstance(res, float)


def test_scale_lexical_score_boundary_one():
    """Verify input 1.0 maps strictly to 1.0."""
    res = scale_lexical_score(1.0)
    assert res == 1.0
    assert isinstance(res, float)


@pytest.mark.parametrize(
    "score", [0.0, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0]
)
def test_scale_lexical_score_output_bounded_in_unit_interval(score):
    """Verify output remains strictly bounded between 0.0 and 1.0 for valid range."""
    res = scale_lexical_score(score)
    assert 0.0 <= res <= 1.0


@pytest.mark.parametrize(
    "invalid_input,expected",
    [
        (-0.01, 0.0),
        (-0.5, 0.0),
        (-100.0, 0.0),
        (1.01, 1.0),
        (1.5, 1.0),
        (100.0, 1.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        ("invalid_string", 0.0),
        (None, 0.0),
        ([], 0.0),
        ({}, 0.0),
    ],
)
def test_scale_lexical_score_edge_cases(invalid_input, expected):
    """Verify non-standard, out-of-bounds, and invalid inputs return expected boundary values."""
    res = scale_lexical_score(invalid_input)
    assert res == expected


def test_scale_lexical_score_suppresses_low_similarity_scores():
    """Verify low raw similarity scores (e.g. 0.2) are non-linearly suppressed below linear value."""
    raw_score = 0.2
    scaled = scale_lexical_score(raw_score, steepness=6.0, midpoint=0.5)
    # Sigmoidal curve suppresses low range values to avoid over-reporting mild overlap
    assert scaled < raw_score


def test_scale_lexical_score_enhances_high_similarity_scores():
    """Verify high raw similarity scores (e.g. 0.8) are non-linearly enhanced above linear value."""
    raw_score = 0.8
    scaled = scale_lexical_score(raw_score, steepness=6.0, midpoint=0.5)
    # Sigmoidal curve enhances high range values to emphasize significant similarity
    assert scaled > raw_score


def test_scale_lexical_score_monotonicity():
    """Verify that scale_lexical_score is strictly monotonic increasing over [0.0, 1.0]."""
    scores = np.linspace(0.0, 1.0, 100)
    scaled_scores = [scale_lexical_score(s) for s in scores]

    for i in range(len(scaled_scores) - 1):
        assert scaled_scores[i] <= scaled_scores[i + 1]


def test_scale_lexical_score_steepness_parameter_effect():
    """Verify increasing steepness increases suppression below 0.5 and enhancement above 0.5."""
    score_low = 0.3
    scaled_gentle = scale_lexical_score(score_low, steepness=4.0)
    scaled_steep = scale_lexical_score(score_low, steepness=10.0)
    assert scaled_steep < scaled_gentle

    score_high = 0.7
    scaled_gentle_high = scale_lexical_score(score_high, steepness=4.0)
    scaled_steep_high = scale_lexical_score(score_high, steepness=10.0)
    assert scaled_steep_high > scaled_gentle_high


def test_scale_lexical_score_custom_midpoint():
    """Verify custom midpoint inflection shift behavior."""
    res = scale_lexical_score(0.7, midpoint=0.7)
    assert 0.0 <= res <= 1.0


def test_softmax_normalize_scores_vector():
    """Verify vectorization over NumPy array of similarity scores."""
    scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scaled = softmax_normalize_scores(scores)

    assert isinstance(scaled, np.ndarray)
    assert len(scaled) == 5
    assert scaled[0] == 0.0
    assert scaled[2] == pytest.approx(0.5, abs=1e-6)
    assert scaled[4] == 1.0
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))


def test_softmax_normalize_scores_python_list():
    """Verify vectorization accepts python list input."""
    scores = [0.0, 0.5, 1.0]
    scaled = softmax_normalize_scores(scores)

    assert isinstance(scaled, np.ndarray)
    np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0], atol=1e-6)


def test_softmax_normalize_scores_empty():
    """Verify vectorization over empty container returns empty array."""
    scaled = softmax_normalize_scores([])
    assert isinstance(scaled, np.ndarray)
    assert len(scaled) == 0


def test_scale_lexical_matrix_numpy():
    """Verify matrix normalization on 2D NumPy array."""
    mat = np.array(
        [
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.5],
            [0.8, 0.5, 1.0],
        ]
    )
    scaled = scale_lexical_matrix(mat)

    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == (3, 3)
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))
    # Diagonals (1.0) must remain 1.0
    np.testing.assert_allclose(np.diag(scaled), [1.0, 1.0, 1.0])


def test_scale_lexical_matrix_pandas_dataframe():
    """Verify matrix normalization on pandas DataFrame preserves index and columns."""
    df = pd.DataFrame(
        [
            [1.0, 0.3, 0.7],
            [0.3, 1.0, 0.5],
            [0.7, 0.5, 1.0],
        ],
        index=["docA", "docB", "docC"],
        columns=["docA", "docB", "docC"],
    )
    scaled_df = scale_lexical_matrix(df)

    assert isinstance(scaled_df, pd.DataFrame)
    assert list(scaled_df.index) == ["docA", "docB", "docC"]
    assert list(scaled_df.columns) == ["docA", "docB", "docC"]
    assert scaled_df.loc["docA", "docA"] == 1.0
    assert scaled_df.loc["docB", "docB"] == 1.0
    assert scaled_df.loc["docA", "docB"] < 0.3
    assert scaled_df.loc["docA", "docC"] > 0.7


def test_scale_lexical_matrix_empty_dataframe():
    """Verify empty DataFrame scaling returns empty DataFrame."""
    empty_df = pd.DataFrame()
    scaled = scale_lexical_matrix(empty_df)
    assert isinstance(scaled, pd.DataFrame)
    assert scaled.empty


def test_scale_lexical_score_symmetry_properties():
    """Verify symmetry of sigmoid transformation around midpoint 0.5."""
    delta = 0.15
    low_val = 0.5 - delta
    high_val = 0.5 + delta

    scaled_low = scale_lexical_score(low_val)
    scaled_high = scale_lexical_score(high_val)

    # 0.5 - scaled_low should equal scaled_high - 0.5
    assert (0.5 - scaled_low) == pytest.approx(scaled_high - 0.5, abs=1e-5)


def test_scale_lexical_score_float32_input_compatibility():
    """Verify compatibility with numpy float32 input scalars."""
    val = np.float32(0.5)
    res = scale_lexical_score(val)
    assert res == pytest.approx(0.5, abs=1e-5)
    assert isinstance(res, float)


def test_scale_lexical_score_zero_steepness_fallback():
    """Verify handling when steepness is set to 0.0."""
    res = scale_lexical_score(0.4, steepness=0.0)
    assert 0.0 <= res <= 1.0


def test_scale_lexical_score_large_steepness_step_function():
    """Verify high steepness creates steeper transition around midpoint."""
    scaled_below = scale_lexical_score(0.40, steepness=20.0)
    scaled_above = scale_lexical_score(0.60, steepness=20.0)

    assert scaled_below < 0.15
    assert scaled_above > 0.85
