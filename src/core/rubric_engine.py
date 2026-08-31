"""
src/core/rubric_engine.py
-------------------------
Automated Rubric-Based Feedback and Grading Engine.

Maps document analysis results (similarity score, AI probability, citation
quality, stylometric deviation) to a customizable grading rubric. This engine
evaluates submissions against weighted criteria and computes a suggested grade.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class CriterionType(str, Enum):
    """Enumeration of supported rubric criterion types."""

    SIMILARITY_SCORE = "similarity_score"
    AI_PROBABILITY = "ai_probability"
    CITATION_QUALITY = "citation_quality"
    STYLOMETRIC_DEVIATION = "stylometric_deviation"
    WORD_COUNT = "word_count"


@dataclass
class RubricCriterion:
    """Represents a single grading criterion within a rubric."""

    name: str
    type: CriterionType
    weight: float  # Weight from 0.0 to 1.0
    max_points: float
    # Thresholds for scoring (e.g., if similarity < 0.2, full points)
    thresholds: dict[str, float] = field(default_factory=dict)

    def evaluate(self, value: float) -> float:
        """Evaluate a specific metric value against this criterion.

        Returns the points awarded based on the criterion's thresholds.
        """
        if self.type == CriterionType.SIMILARITY_SCORE:
            # Lower similarity is better (less plagiarism)
            if value <= self.thresholds.get("excellent", 0.15):
                return self.max_points
            elif value <= self.thresholds.get("good", 0.30):
                return self.max_points * 0.8
            elif value <= self.thresholds.get("poor", 0.50):
                return self.max_points * 0.5
            else:
                return 0.0

        elif self.type == CriterionType.AI_PROBABILITY:
            # Lower AI probability is better
            if value <= self.thresholds.get("excellent", 0.20):
                return self.max_points
            elif value <= self.thresholds.get("good", 0.40):
                return self.max_points * 0.7
            else:
                return 0.0

        elif self.type == CriterionType.STYLOMETRIC_DEVIATION:
            # Lower deviation is better (matches student's baseline)
            if value <= self.thresholds.get("excellent", 5.0):
                return self.max_points
            elif value <= self.thresholds.get("good", 10.0):
                return self.max_points * 0.8
            else:
                return self.max_points * 0.4

        elif self.type == CriterionType.WORD_COUNT:
            # Check if within acceptable range
            min_wc = self.thresholds.get("min", 0)
            max_wc = self.thresholds.get("max", float("inf"))
            if min_wc <= value <= max_wc:
                return self.max_points
            else:
                return self.max_points * 0.5

        # Default: linear scaling based on value (for things like citation quality 0-1)
        return self.max_points * min(1.0, max(0.0, value))


@dataclass
class Rubric:
    """Represents a complete grading rubric."""

    name: str
    criteria: list[RubricCriterion]

    def get_total_max_points(self) -> float:
        """Compute the total maximum points for the rubric."""
        return sum(c.max_points for c in self.criteria)

    def normalize_weights(self) -> None:
        """Ensure all criterion weights sum to 1.0."""
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for c in self.criteria:
                c.weight /= total_weight


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a submission against a rubric."""

    rubric_name: str
    total_score: float
    max_points: float
    percentage: float
    criterion_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_submission(rubric: Rubric, metrics: dict[str, float]) -> EvaluationResult:
    """Evaluate a document's analysis metrics against a rubric.

    Args:
        rubric: The grading rubric to apply.
        metrics: Dictionary mapping CriterionType values to their computed scores.
                 e.g., {'similarity_score': 0.25, 'ai_probability': 0.10}

    Returns:
        A populated EvaluationResult object.
    """
    rubric.normalize_weights()
    total_score = 0.0
    criterion_results = []

    for criterion in rubric.criteria:
        metric_value = metrics.get(criterion.type.value, 0.0)
        points_awarded = criterion.evaluate(metric_value)

        # Apply weight to the points
        weighted_points = points_awarded * criterion.weight

        total_score += weighted_points
        criterion_results.append(
            {
                "criterion_name": criterion.name,
                "metric_value": metric_value,
                "points_awarded": round(points_awarded, 2),
                "weighted_points": round(weighted_points, 2),
                "max_points": criterion.max_points,
            }
        )

    max_points = rubric.get_total_max_points()
    percentage = (total_score / max_points * 100.0) if max_points > 0 else 0.0

    return EvaluationResult(
        rubric_name=rubric.name,
        total_score=round(total_score, 2),
        max_points=round(max_points, 2),
        percentage=round(percentage, 2),
        criterion_results=criterion_results,
    )
