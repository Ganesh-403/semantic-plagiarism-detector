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
tests/core/test_rubric_engine.py
--------------------------------
Comprehensive unit tests for the Rubric-Based Feedback and Grading Engine.
"""

import pytest

from src.core.rubric_engine import (
    CriterionType,
    Rubric,
    RubricCriterion,
    evaluate_submission,
)
from src.db.rubrics_db import get_rubric, initialize_rubrics_db, save_rubric
from src.utils.feedback_generator import (
    generate_feedback_paragraph,
    generate_json_feedback,
)


class TestRubricEvaluation:
    """Test suite for rubric scoring logic."""

    def test_evaluate_perfect_submission(self):
        """Verify a perfect submission gets full points."""
        rubric = Rubric(
            name="Test Rubric",
            criteria=[
                RubricCriterion(
                    name="Plagiarism Check",
                    type=CriterionType.SIMILARITY_SCORE,
                    weight=1.0,
                    max_points=100.0,
                    thresholds={"excellent": 0.15},
                )
            ],
        )

        metrics = {"similarity_score": 0.10}  # Excellent
        result = evaluate_submission(rubric, metrics)

        assert result.total_score == 100.0
        assert result.percentage == 100.0

    def test_evaluate_poor_submission(self):
        """Verify a poor submission gets low points."""
        rubric = Rubric(
            name="Test Rubric",
            criteria=[
                RubricCriterion(
                    name="Plagiarism Check",
                    type=CriterionType.SIMILARITY_SCORE,
                    weight=1.0,
                    max_points=100.0,
                    thresholds={"excellent": 0.15, "good": 0.30, "poor": 0.50},
                )
            ],
        )

        metrics = {"similarity_score": 0.80}  # Very poor
        result = evaluate_submission(rubric, metrics)

        assert result.total_score == 0.0

    def test_weight_normalization(self):
        """Verify criterion weights are normalized to sum to 1.0."""
        rubric = Rubric(
            name="Test Rubric",
            criteria=[
                RubricCriterion(
                    name="A", type=CriterionType.WORD_COUNT, weight=2.0, max_points=50.0
                ),
                RubricCriterion(
                    name="B", type=CriterionType.WORD_COUNT, weight=2.0, max_points=50.0
                ),
            ],
        )

        rubric.normalize_weights()
        assert rubric.criteria[0].weight == 0.5
        assert rubric.criteria[1].weight == 0.5


class TestFeedbackGenerator:
    """Test suite for natural language feedback generation."""

    def test_generate_paragraph_excellent(self):
        """Verify excellent scores generate positive feedback."""
        result = evaluate_submission(
            Rubric(
                name="Final Essay",
                criteria=[
                    RubricCriterion(
                        name="Originality",
                        type=CriterionType.SIMILARITY_SCORE,
                        weight=1.0,
                        max_points=100.0,
                        thresholds={"excellent": 0.20},
                    )
                ],
            ),
            {"similarity_score": 0.05},
        )

        feedback = generate_feedback_paragraph(result)
        assert "Excellent work!" in feedback
        assert "100.0%" in feedback

    def test_generate_json_feedback_structure(self):
        """Verify JSON feedback contains required keys."""
        result = evaluate_submission(
            Rubric(
                name="Test",
                criteria=[
                    RubricCriterion(
                        name="AI Check",
                        type=CriterionType.AI_PROBABILITY,
                        weight=1.0,
                        max_points=10.0,
                        thresholds={"excellent": 0.20},
                    )
                ],
            ),
            {"ai_probability": 0.10},
        )

        json_fb = generate_json_feedback(result)
        assert "summary" in json_fb
        assert "criteria" in json_fb
        assert "text_feedback" in json_fb


class TestRubricsDB:
    """Test suite for the rubrics database."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "test_rubrics.db"
        initialize_rubrics_db(db_path)
        return db_path

    def test_save_and_get_rubric(self, temp_db):
        """Verify a rubric can be saved and retrieved."""
        rubric = Rubric(
            name="Midterm Rubric",
            criteria=[
                RubricCriterion(
                    name="Citations",
                    type=CriterionType.CITATION_QUALITY,
                    weight=1.0,
                    max_points=20.0,
                    thresholds={},
                )
            ],
        )

        assert save_rubric(rubric, db_path=temp_db) is True

        retrieved = get_rubric("Midterm Rubric", db_path=temp_db)
        assert retrieved is not None
        assert retrieved.name == "Midterm Rubric"
        assert len(retrieved.criteria) == 1
        assert retrieved.criteria[0].type == CriterionType.CITATION_QUALITY

    def test_get_nonexistent_rubric(self, temp_db):
        """Verify retrieving a nonexistent rubric returns None."""
        assert get_rubric("Nonexistent", db_path=temp_db) is None
