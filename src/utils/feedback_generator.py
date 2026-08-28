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
src/utils/feedback_generator.py
-------------------------------
Natural Language Feedback Generator.

Synthesizes structured, human-readable feedback paragraphs based on
rubric evaluation results. This helps instructors quickly communicate
grading decisions to students.
"""

import logging
from typing import Any, Dict, List

from src.core.rubric_engine import EvaluationResult

logger = logging.getLogger(__name__)


def generate_feedback_paragraph(result: EvaluationResult) -> str:
    """Generate a natural language feedback summary from an evaluation result.

    Args:
        result: The EvaluationResult from the rubric engine.

    Returns:
        A formatted string containing the feedback summary.
    """
    lines = []
    lines.append(f"## Grading Summary: {result.rubric_name}")
    lines.append(
        f"**Overall Score:** {result.total_score} / {result.max_points} ({result.percentage:.1f}%)"
    )
    lines.append("")

    # Determine overall sentiment based on percentage
    if result.percentage >= 90:
        lines.append(
            "Excellent work! Your submission demonstrates a strong understanding of the material and adheres closely to academic integrity standards."
        )
    elif result.percentage >= 75:
        lines.append(
            "Good effort. Your submission meets most requirements, but there are areas for improvement."
        )
    elif result.percentage >= 60:
        lines.append(
            "Satisfactory work, but significant improvements are needed in several areas."
        )
    else:
        lines.append(
            "This submission requires major revision. Please review the feedback below carefully."
        )

    lines.append("")
    lines.append("### Detailed Breakdown:")

    for cr in result.criterion_results:
        name = cr["criterion_name"]
        awarded = cr["weighted_points"]
        max_pts = cr["max_points"] * (
            awarded / cr["points_awarded"] if cr["points_awarded"] > 0 else 1
        )  # Rough max

        # Generate specific feedback based on criterion name and score
        if "Similarity" in name or "Plagiarism" in name:
            if cr["points_awarded"] == cr["max_points"]:
                lines.append(
                    f"- **{name}:** Excellent. No significant plagiarism detected."
                )
            elif cr["points_awarded"] > 0:
                lines.append(
                    f"- **{name}:** Some overlapping text was detected. Please ensure all sources are properly cited."
                )
            else:
                lines.append(
                    f"- **{name}:** High similarity detected. This requires immediate attention and proper citation."
                )

        elif "AI" in name:
            if cr["points_awarded"] == cr["max_points"]:
                lines.append(
                    f"- **{name}:** The writing style appears authentic and human-generated."
                )
            else:
                lines.append(
                    f"- **{name}:** The text exhibits patterns consistent with AI generation. Please ensure the work is your own."
                )

        elif "Stylometric" in name or "Authorship" in name:
            if cr["points_awarded"] == cr["max_points"]:
                lines.append(
                    f"- **{name}:** The writing style is consistent with your previous submissions."
                )
            else:
                lines.append(
                    f"- **{name}:** The writing style deviates significantly from your historical baseline."
                )

        else:
            lines.append(f"- **{name}:** Awarded {cr['points_awarded']} points.")

    return "\n".join(lines)


def generate_json_feedback(result: EvaluationResult) -> dict[str, Any]:
    """Generate a structured JSON feedback object for API responses.

    Args:
        result: The EvaluationResult from the rubric engine.

    Returns:
        A dictionary containing the structured feedback.
    """
    return {
        "summary": {
            "rubric_name": result.rubric_name,
            "total_score": result.total_score,
            "max_points": result.max_points,
            "percentage": result.percentage,
        },
        "criteria": result.criterion_results,
        "text_feedback": generate_feedback_paragraph(result),
    }
