import math

import pytest

from src.core.config import (
    DEFAULT_THRESHOLDS,
    PLAGIARISM_THRESHOLD,
    SimilarityThresholds,
    is_plagiarism,
    normalize_score,
    normalize_severity_label,
    severity_from_score,
    severity_key,
    severity_rank,
    validate_thresholds,
)


def test_defaults_are_single_source_of_truth():
    assert DEFAULT_THRESHOLDS == SimilarityThresholds(
        plagiarism=0.59,
        medium=0.75,
        high=0.90,
    )
    assert PLAGIARISM_THRESHOLD == DEFAULT_THRESHOLDS.plagiarism


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plagiarism": -0.01},
        {"high": 1.01},
        {"plagiarism": 0.80, "medium": 0.75},
        {"medium": 0.95, "high": 0.90},
        {"plagiarism": math.nan},
        {"medium": math.inf},
    ],
)
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SimilarityThresholds(**kwargs)


def test_default_order_is_valid():
    assert validate_thresholds(DEFAULT_THRESHOLDS) is DEFAULT_THRESHOLDS


def test_threshold_values_are_normalized_to_floats():
    thresholds = SimilarityThresholds(
        plagiarism=0,
        medium=0.75,
        high=1,
    )

    assert thresholds.plagiarism == 0.0
    assert thresholds.medium == 0.75
    assert thresholds.high == 1.0

    assert isinstance(thresholds.plagiarism, float)
    assert isinstance(thresholds.medium, float)
    assert isinstance(thresholds.high, float)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-0.20, 0.0),
        (0.0, 0.0),
        (0.59, 0.59),
        (1.0, 1.0),
        (1.20, 1.0),
    ],
)
def test_score_is_clamped(score, expected):
    assert normalize_score(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "Low"),
        (0.749999, "Low"),
        (0.75, "Medium"),
        (0.899999, "Medium"),
        (0.90, "High"),
        (1.0, "High"),
    ],
)
def test_exact_severity_boundaries(score, expected):
    assert severity_from_score(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.589999, False),
        (0.59, True),
        (1.2, True),
        (-0.2, False),
    ],
)
def test_plagiarism_boundary(score, expected):
    assert is_plagiarism(score) is expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("High", "High"),
        ("🔴 High", "High"),
        ("ðŸ”´ High", "High"),
        ("Medium", "Medium"),
        ("🟡 Medium", "Medium"),
        ("warning", "Medium"),
        ("Low", "Low"),
        ("🟢 Low", "Low"),
    ],
)
def test_legacy_labels_are_normalized(label, expected):
    assert normalize_severity_label(label) == expected


@pytest.mark.parametrize(
    "label",
    [
        "very high risk",
        "highly suspicious",
        "medium priority",
        "low confidence",
        "warning message",
        "unrelated warning text",
        "below average",
    ],
)
def test_unrelated_strings_with_severity_words_are_rejected(label):
    with pytest.raises(ValueError, match="Unknown severity label"):
        normalize_severity_label(label)


def test_severity_keys_and_ranks():
    assert severity_key(0.1) == "low"
    assert severity_key(0.8) == "medium"
    assert severity_key(0.95) == "high"
    assert severity_rank("Low") < severity_rank("Medium")

    assert severity_rank("Medium") < severity_rank("High")

    assert severity_rank("Medium") < severity_rank("High")


def test_severity_respects_plagiarism_boundary():
    thresholds = SimilarityThresholds(
        plagiarism=0.59,
        medium=0.75,
        high=0.90,
    )

    # Below the plagiarism threshold: not flagged, but remains
    # Low to preserve the existing three-level severity contract.
    assert severity_from_score(0.58, thresholds) == "Low"
    assert is_plagiarism(0.58, thresholds.plagiarism) is False

    # At the plagiarism threshold: flagged with Low severity.
    assert severity_from_score(0.59, thresholds) == "Low"
    assert is_plagiarism(0.59, thresholds.plagiarism) is True

    # Higher severity boundaries remain unchanged.
    assert severity_from_score(0.75, thresholds) == "Medium"
    assert severity_from_score(0.90, thresholds) == "High"
