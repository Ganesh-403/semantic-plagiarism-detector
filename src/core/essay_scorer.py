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
src/core/essay_scorer.py
------------------------
Automated Essay Scoring and Holistic Rubric Engine.

Orchestrates the holistic scoring pipeline by mapping extracted analytic
traits to a customizable grading rubric. Computes a final holistic grade
based on weighted trait scores.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.core.trait_analyzer import extract_analytic_traits

logger = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """Represents a single criterion in the scoring rubric."""

    name: str
    weight: float  # Weight from 0.0 to 1.0
    trait_mapping: (
        str  # Which trait to map to (e.g., 'coherence', 'lexical_complexity.ttr')
    )
    max_points: float = 10.0


@dataclass
class ScoringRubric:
    """Represents a complete grading rubric."""

    name: str
    criteria: List[RubricCriterion] = field(default_factory=list)

    def normalize_weights(self) -> None:
        """Ensure all criterion weights sum to 1.0."""
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for c in self.criteria:
                c.weight /= total_weight


# Default holistic rubric
DEFAULT_RUBRIC = ScoringRubric(
    name="Standard Holistic Rubric",
    criteria=[
        RubricCriterion(
            name="Coherence & Flow",
            weight=0.30,
            trait_mapping="coherence",
            max_points=10.0,
        ),
        RubricCriterion(
            name="Lexical Complexity",
            weight=0.30,
            trait_mapping="lexical_complexity.ttr",
            max_points=10.0,
        ),
        RubricCriterion(
            name="Academic Vocabulary",
            weight=0.20,
            trait_mapping="lexical_complexity.academic_density",
            max_points=10.0,
        ),
        RubricCriterion(
            name="Argumentation Structure",
            weight=0.20,
            trait_mapping="argumentation_total",
            max_points=10.0,
        ),
    ],
)


def _get_trait_value(traits: Dict[str, Any], mapping: str) -> float:
    """Safely extract a trait value using dot notation mapping."""
    keys = mapping.split(".")
    val = traits
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return 0.0

    # Special handling for argumentation_total
    if mapping == "argumentation_total":
        if isinstance(val, dict):
            # Sum all marker counts and normalize (e.g., max 20 markers = 10 points)
            total_markers = sum(val.values())
            return min(1.0, total_markers / 20.0) * 10.0
        return 0.0

    # For density/TTR (0-1 scale), scale to max_points
    if isinstance(val, (int, float)):
        if mapping in [
            "coherence",
            "lexical_complexity.ttr",
            "lexical_complexity.academic_density",
        ]:
            return val * 10.0  # Scale 0-1 to 0-10
        return float(val)

    return 0.0


def score_essay(text: str, rubric: Optional[ScoringRubric] = None) -> Dict[str, Any]:
    """Score an essay based on analytic traits and a rubric.

    Args:
        text: The essay text.
        rubric: Optional custom rubric. Uses DEFAULT_RUBRIC if None.

    Returns:
        Dictionary containing trait scores, criterion scores, and final grade.
    """
    if rubric is None:
        rubric = DEFAULT_RUBRIC

    rubric.normalize_weights()
    traits = extract_analytic_traits(text)

    criterion_scores = []
    total_weighted_score = 0.0
    max_possible = 0.0

    for criterion in rubric.criteria:
        raw_value = _get_trait_value(traits, criterion.trait_mapping)
        # Cap at max_points
        score = min(raw_value, criterion.max_points)
        weighted = score * criterion.weight

        criterion_scores.append(
            {
                "name": criterion.name,
                "raw_value": round(raw_value, 2),
                "score": round(score, 2),
                "weighted_score": round(weighted, 2),
                "max_points": criterion.max_points,
                "weight": criterion.weight,
            }
        )

        total_weighted_score += weighted
        max_possible += criterion.max_points * criterion.weight

    # Final grade out of 100
    final_grade = (
        (total_weighted_score / max_possible * 100.0) if max_possible > 0 else 0.0
    )

    return {
        "traits": traits,
        "criterion_scores": criterion_scores,
        "final_grade": round(final_grade, 2),
        "rubric_name": rubric.name,
    }
