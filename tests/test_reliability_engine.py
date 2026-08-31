# semantic-plagiarism-detector/tests/test_reliability_engine.py

import pytest
from src.core.reliability_engine import ReliabilityEngine

def test_cohens_kappa_perfect_agreement():
    r1 = [1, 0, 1, 1, 0]
    r2 = [1, 0, 1, 1, 0]
    assert ReliabilityEngine.compute_cohens_kappa(r1, r2) == 1.0

def test_fleiss_kappa_calculation():
    # Example ratings matrix: 3 subjects, 3 categories, 5 raters each
    matrix = [
        [0, 0, 5],
        [3, 1, 1],
        [1, 2, 2]
    ]
    kappa = ReliabilityEngine.compute_fleiss_kappa(matrix)
    assert isinstance(kappa, float)
    assert -1.0 <= kappa <= 1.0

def test_reviewer_bias_weighting():
    history = [
        {"reviewer_id": "rev_01", "consensus_deviation": 0.1},
        {"reviewer_id": "rev_02", "consensus_deviation": 0.8}
    ]
    weights = ReliabilityEngine.compute_reviewer_bias_weights(history)
    assert weights["rev_01"] == 0.9
    assert weights["rev_02"] == 0.2
