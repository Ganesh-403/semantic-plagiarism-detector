"""Central plagiarism threshold and severity configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from math import isfinite
from numbers import Real
from typing import Final, Mapping

LOW_SEVERITY: Final[str] = "Low"
MEDIUM_SEVERITY: Final[str] = "Medium"
HIGH_SEVERITY: Final[str] = "High"
CRITICAL_SEVERITY: Final[str] = "Critical"

SEVERITY_ORDER: Final[tuple[str, ...]] = (
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    HIGH_SEVERITY,
    CRITICAL_SEVERITY,
)
SEVERITY_RANK: Final[Mapping[str, int]] = {
    label: rank for rank, label in enumerate(SEVERITY_ORDER)
}

# Embedding batch size configuration (default: 32)
EMBEDDING_BATCH_SIZE: Final[int] = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


@dataclass(frozen=True)
class SimilarityThresholds:
    """Validated plagiarism and severity boundaries."""

    plagiarism: float = 0.59
    medium: float = 0.75
    high: float = 0.90

    def __post_init__(self) -> None:
        plagiarism = _validate_boundary("plagiarism", self.plagiarism)
        medium = _validate_boundary("medium", self.medium)
        high = _validate_boundary("high", self.high)

        if not plagiarism <= medium <= high:
            raise ValueError(
                "Thresholds must satisfy "
                "0.0 <= plagiarism <= medium <= high <= 1.0."
            )

        object.__setattr__(self, "plagiarism", plagiarism)
        object.__setattr__(self, "medium", medium)
        object.__setattr__(self, "high", high)


def _validate_boundary(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} threshold must be a real number.")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} threshold must be finite.")
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} threshold must be between 0.0 and 1.0.")
    return numeric


def validate_thresholds(
    thresholds: SimilarityThresholds,
) -> SimilarityThresholds:
    """Validate threshold type and ordering."""
    if not isinstance(thresholds, SimilarityThresholds):
        raise TypeError("thresholds must be a SimilarityThresholds instance.")

    if not thresholds.plagiarism <= thresholds.medium <= thresholds.high:
        raise ValueError(
            "Thresholds must satisfy "
            "0.0 <= plagiarism <= medium <= high <= 1.0."
        )

    return thresholds


DEFAULT_THRESHOLDS: Final[SimilarityThresholds] = SimilarityThresholds()
PLAGIARISM_THRESHOLD: Final[float] = DEFAULT_THRESHOLDS.plagiarism


def normalize_score(score: Real) -> float:
    """Return a finite score clamped to the inclusive [0, 1] range."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("Similarity score must be a real number.")

    value = float(score)
    if not isfinite(value):
        raise ValueError("Similarity score must be finite.")
    return min(1.0, max(0.0, value))


def is_plagiarism(
    score: Real,
    threshold: Real = DEFAULT_THRESHOLDS.plagiarism,
) -> bool:
    """Return whether a score reaches a validated flagging threshold."""
    normalized_score = normalize_score(score)
    normalized_threshold = _validate_boundary("plagiarism", threshold)
    return normalized_score >= normalized_threshold


def severity_from_score(
    score: Real,
    thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return the canonical Low, Medium, or High severity label."""
    validate_thresholds(thresholds)
    normalized = normalize_score(score)

    if normalized < thresholds.plagiarism:
        return LOW_SEVERITY
    if normalized >= thresholds.high:
        return HIGH_SEVERITY
    if normalized >= thresholds.medium:
        return MEDIUM_SEVERITY
    return LOW_SEVERITY


def severity_key(
    score: Real,
    thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return a lowercase key for colors and presentation helpers."""
    return severity_from_score(score, thresholds).lower()


def normalize_severity_label(label: str) -> str:
    """Normalize supported canonical and legacy severity labels."""
    clean = str(label or "").strip().lower()

    severity_aliases = {
        "low": LOW_SEVERITY,
        "medium": MEDIUM_SEVERITY,
        "high": HIGH_SEVERITY,
        "critical": CRITICAL_SEVERITY,
        "🟢 low": LOW_SEVERITY,
        "🟡 medium": MEDIUM_SEVERITY,
        "🔴 high": HIGH_SEVERITY,
        "ðÿ”´ high": HIGH_SEVERITY,
        "warning": MEDIUM_SEVERITY,
    }

    try:
        return severity_aliases[clean]
    except KeyError:
        raise ValueError(f"Unknown severity label: {label!r}") from None


def severity_rank(label: str) -> int:
    """Return a stable sort rank for a severity label."""
    return SEVERITY_RANK[normalize_severity_label(label)]
