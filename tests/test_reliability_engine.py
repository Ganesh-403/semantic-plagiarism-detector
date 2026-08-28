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

# semantic-plagiarism-detector/tests/test_reliability_engine.py

import pytest

from src.core.reliability_engine import ReliabilityEngine


def test_cohens_kappa_perfect_agreement():
    r1 = [1, 0, 1, 1, 0]
    r2 = [1, 0, 1, 1, 0]
    assert ReliabilityEngine.compute_cohens_kappa(r1, r2) == 1.0


def test_fleiss_kappa_calculation():
    # Example ratings matrix: 3 subjects, 3 categories, 5 raters each
    matrix = [[0, 0, 5], [3, 1, 1], [1, 2, 2]]
    kappa = ReliabilityEngine.compute_fleiss_kappa(matrix)
    assert isinstance(kappa, float)
    assert -1.0 <= kappa <= 1.0


def test_reviewer_bias_weighting():
    history = [
        {"reviewer_id": "rev_01", "consensus_deviation": 0.1},
        {"reviewer_id": "rev_02", "consensus_deviation": 0.8},
    ]
    weights = ReliabilityEngine.compute_reviewer_bias_weights(history)
    assert weights["rev_01"] == 0.9
    assert weights["rev_02"] == 0.2
