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

"""Tests for the pattern recognition and prediction system (issue #2840).

Covers: PatternDetectionEngine, PatternRepository, RecommendationEngine,
risk scoring, technique evolution, and detection accuracy evaluation.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, _PROJECT_ROOT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_incident(
    doc_a: str,
    doc_b: str,
    similarity: float,
    date_flagged: str | None = None,
    owner_a: str = "",
    owner_b: str = "",
    assignment_title: str = "",
) -> dict:
    return {
        "document_a": doc_a,
        "document_b": doc_b,
        "similarity_score": similarity,
        "date_flagged": date_flagged or _utc_now(),
        "last_seen": date_flagged or _utc_now(),
        "severity_rank": "High" if similarity >= 0.7 else "Medium",
        "review_status": "Pending",
        "owner_a": owner_a,
        "owner_b": owner_b,
        "assignment_title": assignment_title,
        "assignment_title_a": assignment_title,
        "assignment_title_b": assignment_title,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PatternDetectionEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternDetectionEngine:
    """Test the core pattern detection engine."""

    def test_detect_recurring_pairs(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        incidents = [
            _make_incident("doc_a.pdf", "doc_b.pdf", 0.92, "2026-01-01"),
            _make_incident("doc_a.pdf", "doc_b.pdf", 0.88, "2026-02-01"),
            _make_incident("doc_a.pdf", "doc_b.pdf", 0.91, "2026-03-01"),
        ]
        patterns = engine.detect_recurring_patterns(incidents, min_occurrence=2)

        assert len(patterns) >= 1
        pair_pattern = next(p for p in patterns if "doc_a.pdf" in p["document_group"])
        assert pair_pattern["occurrence_count"] == 3
        assert pair_pattern["avg_similarity"] >= 0.88
        assert pair_pattern["confidence_score"] > 0.0

    def test_detect_author_clusters(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        incidents = [
            _make_incident(
                "s1_essay1.pdf",
                "s1_essay2.pdf",
                0.75,
                owner_a="student1",
                owner_b="student2",
            ),
            _make_incident(
                "s1_essay3.pdf",
                "s1_essay4.pdf",
                0.80,
                owner_a="student1",
                owner_b="student3",
            ),
            _make_incident(
                "s1_essay5.pdf",
                "s1_essay6.pdf",
                0.72,
                owner_a="student1",
                owner_b="student4",
            ),
        ]
        patterns = engine.detect_recurring_patterns(incidents, min_occurrence=2)

        author_patterns = [p for p in patterns if p["pattern_type"] == "collaborative"]
        assert len(author_patterns) >= 1
        student1_pattern = next(
            p for p in author_patterns if "student1" in (p["author_group"] or [])
        )
        assert student1_pattern["occurrence_count"] == 3

    def test_detect_assignment_hotspots(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        incidents = [
            _make_incident("d1.pdf", "d2.pdf", 0.85, assignment_title="Essay 1"),
            _make_incident("d3.pdf", "d4.pdf", 0.78, assignment_title="Essay 1"),
            _make_incident("d5.pdf", "d6.pdf", 0.82, assignment_title="Essay 1"),
        ]
        patterns = engine.detect_recurring_patterns(incidents, min_occurrence=2)

        assignment_patterns = [
            p for p in patterns if p.get("assignment_title") == "Essay 1"
        ]
        assert len(assignment_patterns) >= 1

    def test_no_patterns_below_threshold(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        incidents = [
            _make_incident("a.pdf", "b.pdf", 0.5),
        ]
        patterns = engine.detect_recurring_patterns(incidents, min_occurrence=2)
        assert len(patterns) == 0

    def test_empty_incidents(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        patterns = engine.detect_recurring_patterns([], min_occurrence=2)
        assert patterns == []

    def test_technique_evolution(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        now = datetime.now(timezone.utc)
        recent = now.isoformat()
        old = now.replace(day=max(1, now.day - 45)).isoformat()

        incidents = [
            _make_incident("a.pdf", "b.pdf", 0.95, date_flagged=recent),
            _make_incident("c.pdf", "d.pdf", 0.92, date_flagged=recent),
            _make_incident("e.pdf", "f.pdf", 0.60, date_flagged=old),
        ]
        result = engine.detect_technique_evolution(incidents, window_days=30)

        assert "drift_score" in result
        assert "current_distribution" in result
        assert "emerging_techniques" in result

    def test_risk_scoring_heuristic(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        result = engine.score_document_risk(
            "test.pdf",
            {
                "max_similarity": 0.90,
                "avg_similarity": 0.75,
                "author_incident_count": 5,
                "author_avg_similarity": 0.80,
                "assignment_incident_rate": 0.3,
                "document_length": 5000,
                "submission_hour": 23,
                "days_until_deadline": 1,
                "class_section_risk": 0.4,
            },
        )

        assert result["risk_score"] >= 0.0
        assert result["risk_score"] <= 1.0
        assert result["risk_level"] in (
            "Critical",
            "High",
            "Medium",
            "Low",
            "Negligible",
        )
        assert len(result["contributing_factors"]) > 0

    def test_evaluate_detection_accuracy(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine()
        predicted = [
            {"pattern_id": "PAT-AAA"},
            {"pattern_id": "PAT-BBB"},
            {"pattern_id": "PAT-CCC"},
        ]
        ground_truth = [
            {"pattern_id": "PAT-AAA", "label": "true"},
            {"pattern_id": "PAT-BBB", "label": "true"},
            {"pattern_id": "PAT-DDD", "label": "true"},
        ]
        metrics = engine.evaluate_detection_accuracy(predicted, ground_truth)

        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_severity_classification(self):
        from src.core.pattern_recognition import PatternDetectionEngine

        assert PatternDetectionEngine._severity_from_score(0.95) == "Critical"
        assert PatternDetectionEngine._severity_from_score(0.75) == "High"
        assert PatternDetectionEngine._severity_from_score(0.50) == "Medium"
        assert PatternDetectionEngine._severity_from_score(0.20) == "Low"


# ═══════════════════════════════════════════════════════════════════════════
#  PatternRepository
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternRepository:
    """Test the pattern repository data access layer."""

    @pytest.fixture
    def repo(self, tmp_path):
        db_path = str(tmp_path / "test_patterns.db")
        from src.db.pattern_repository import PatternRepository

        return PatternRepository(db_path)

    def test_upsert_and_get_pattern(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-TEST01",
            pattern_type="copy_paste",
            document_group=["a.pdf", "b.pdf"],
            avg_similarity=0.92,
            occurrence_count=3,
            confidence_score=0.85,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-03-01T00:00:00+00:00",
            description="Test pattern",
        )
        pattern = repo.get_pattern_by_id("PAT-TEST01")
        assert pattern is not None
        assert pattern["pattern_type"] == "copy_paste"
        assert pattern["avg_similarity"] == 0.92

    def test_upsert_update(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-TEST02",
            pattern_type="paraphrase",
            document_group=["x.pdf"],
            avg_similarity=0.70,
            occurrence_count=2,
            confidence_score=0.60,
            severity="Medium",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
        )
        repo.upsert_pattern(
            pattern_id="PAT-TEST02",
            pattern_type="paraphrase",
            document_group=["x.pdf", "y.pdf"],
            avg_similarity=0.80,
            occurrence_count=4,
            confidence_score=0.75,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-03-01T00:00:00+00:00",
        )
        pattern = repo.get_pattern_by_id("PAT-TEST02")
        assert pattern["occurrence_count"] == 4
        assert pattern["avg_similarity"] == 0.80

    def test_get_patterns_with_filters(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-F01",
            pattern_type="copy_paste",
            document_group=["a.pdf"],
            avg_similarity=0.9,
            occurrence_count=2,
            confidence_score=0.7,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
            status="active",
        )
        repo.upsert_pattern(
            pattern_id="PAT-F02",
            pattern_type="paraphrase",
            document_group=["b.pdf"],
            avg_similarity=0.6,
            occurrence_count=2,
            confidence_score=0.5,
            severity="Medium",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
            status="resolved",
        )
        active = repo.get_patterns(status="active")
        assert len(active) == 1
        assert active[0]["pattern_id"] == "PAT-F01"

    def test_update_pattern_status(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-S01",
            pattern_type="copy_paste",
            document_group=["a.pdf"],
            avg_similarity=0.9,
            occurrence_count=2,
            confidence_score=0.7,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
        )
        assert repo.update_pattern_status("PAT-S01", "resolved")
        pattern = repo.get_pattern_by_id("PAT-S01")
        assert pattern["status"] == "resolved"

    def test_evolution_snapshot(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-E01",
            pattern_type="copy_paste",
            document_group=["a.pdf"],
            avg_similarity=0.9,
            occurrence_count=2,
            confidence_score=0.7,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
        )
        repo.record_evolution_snapshot(
            pattern_id="PAT-E01",
            occurrence_count=2,
            avg_similarity=0.9,
            confidence_score=0.7,
        )
        repo.record_evolution_snapshot(
            pattern_id="PAT-E01",
            occurrence_count=4,
            avg_similarity=0.88,
            confidence_score=0.8,
            drift_score=0.05,
        )
        ts = repo.get_evolution_timeseries("PAT-E01", days=365)
        assert len(ts) == 2
        assert ts[0]["occurrence_count"] == 2
        assert ts[1]["occurrence_count"] == 4

    def test_risk_score_upsert(self, repo):
        repo.upsert_risk_score(
            document_name="risky.pdf",
            risk_score=0.85,
            risk_level="High",
            contributing_factors=["High similarity", "Prior incidents"],
        )
        score = repo.get_risk_score("risky.pdf")
        assert score is not None
        assert score["risk_score"] == 0.85
        assert score["risk_level"] == "High"

    def test_high_risk_documents(self, repo):
        repo.upsert_risk_score("low.pdf", 0.3, "Low")
        repo.upsert_risk_score("high.pdf", 0.9, "Critical")
        repo.upsert_risk_score("med.pdf", 0.5, "Medium")
        high = repo.get_high_risk_documents(threshold=0.7)
        assert len(high) == 1
        assert high[0]["document_name"] == "high.pdf"

    def test_recommendation_crud(self, repo):
        repo.create_recommendation(
            recommendation_id="REC-TEST01",
            recommendation_type="monitor_author",
            priority=2,
            target="student1",
            message="Monitor this student",
            action_items=["Check submissions"],
        )
        recs = repo.get_recommendations(status="pending")
        assert len(recs) == 1
        assert recs[0]["target"] == "student1"

        repo.update_recommendation_status("REC-TEST01", "acknowledged")
        recs = repo.get_recommendations(status="acknowledged")
        assert len(recs) == 1

    def test_pattern_summary(self, repo):
        repo.upsert_pattern(
            pattern_id="PAT-SUM1",
            pattern_type="copy_paste",
            document_group=["a.pdf"],
            avg_similarity=0.9,
            occurrence_count=2,
            confidence_score=0.7,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
            status="active",
        )
        repo.upsert_pattern(
            pattern_id="PAT-SUM2",
            pattern_type="paraphrase",
            document_group=["b.pdf"],
            avg_similarity=0.6,
            occurrence_count=2,
            confidence_score=0.5,
            severity="Medium",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
            status="active",
        )
        summary = repo.get_pattern_summary()
        assert summary["total"] == 2
        assert summary["by_type"]["copy_paste"] == 1
        assert summary["by_type"]["paraphrase"] == 1


# ═══════════════════════════════════════════════════════════════════════════
#  RecommendationEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestRecommendationEngine:
    """Test the proactive recommendation engine."""

    def test_author_recommendations(self):
        from src.core.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        patterns = [
            {
                "pattern_id": "PAT-R01",
                "pattern_type": "collaborative",
                "author_group": ["repeat_offender"],
                "occurrence_count": 6,
                "avg_similarity": 0.85,
                "confidence_score": 0.9,
                "severity": "Critical",
                "document_group": ["a.pdf", "b.pdf"],
            }
        ]
        recs = engine.generate_recommendations(patterns, [], None)
        author_recs = [r for r in recs if r["target"] == "repeat_offender"]
        assert len(author_recs) >= 1
        assert author_recs[0]["priority"] == 1  # Critical

    def test_assignment_recommendations(self):
        from src.core.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        patterns = [
            {
                "pattern_id": "PAT-A01",
                "pattern_type": "source_sharing",
                "assignment_title": "Midterm Essay",
                "occurrence_count": 5,
                "document_group": ["d1.pdf", "d2.pdf", "d3.pdf", "d4.pdf", "d5.pdf"],
                "avg_similarity": 0.75,
                "confidence_score": 0.8,
                "severity": "High",
                "author_group": [],
            }
        ]
        recs = engine.generate_recommendations(patterns, [], None)
        assignment_recs = [r for r in recs if r["target"] == "Midterm Essay"]
        assert len(assignment_recs) >= 1
        assert assignment_recs[0]["recommendation_type"] == "redesign_assignment"

    def test_risk_recommendations(self):
        from src.core.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        risk_scores = [
            {
                "document_name": f"doc{i}.pdf",
                "risk_score": 0.9,
                "risk_level": "Critical",
            }
            for i in range(6)
        ]
        recs = engine.generate_recommendations([], risk_scores, None)
        batch_recs = [r for r in recs if r["target"] == "batch"]
        assert len(batch_recs) == 1
        assert batch_recs[0]["priority"] == 1  # Critical

    def test_trend_recommendations(self):
        from src.core.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        trend_data = {
            "drift_score": 0.2,
            "drift_direction": "increasing",
            "emerging_techniques": ["paraphrase", "copy_paste"],
        }
        recs = engine.generate_recommendations([], [], trend_data)
        assert len(recs) >= 2  # trend + emerging

    def test_acknowledge_and_dismiss(self, tmp_path):
        from src.core.recommendations import RecommendationEngine
        from src.db.pattern_repository import PatternRepository

        repo = PatternRepository(str(tmp_path / "ack_test.db"))
        engine = RecommendationEngine(repository=repo)
        # Create a pattern first so the FK constraint is satisfied
        repo.upsert_pattern(
            pattern_id="PAT-AD01",
            pattern_type="collaborative",
            document_group=["a.pdf"],
            avg_similarity=0.80,
            occurrence_count=4,
            confidence_score=0.8,
            severity="High",
            first_seen="2026-01-01T00:00:00+00:00",
            last_seen="2026-01-01T00:00:00+00:00",
        )
        patterns = [
            {
                "pattern_id": "PAT-AD01",
                "pattern_type": "collaborative",
                "author_group": ["test_student"],
                "occurrence_count": 4,
                "avg_similarity": 0.80,
                "confidence_score": 0.8,
                "severity": "High",
                "document_group": ["a.pdf"],
            }
        ]
        engine.generate_recommendations(patterns, [], None)
        recs = engine.get_pending_actions()
        assert len(recs) >= 1
        rec_id = recs[0]["recommendation_id"]
        assert engine.acknowledge_recommendation(rec_id)

    def test_empty_inputs(self):
        from src.core.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        recs = engine.generate_recommendations([], [], None)
        assert recs == []


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: Engine + Repository
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Test engine + repository integration."""

    @pytest.fixture
    def repo(self, tmp_path):
        db_path = str(tmp_path / "test_int.db")
        from src.db.pattern_repository import PatternRepository

        return PatternRepository(db_path)

    def test_detect_and_persist(self, repo):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine(repository=repo)
        incidents = [
            _make_incident("a.pdf", "b.pdf", 0.92, "2026-01-01"),
            _make_incident("a.pdf", "b.pdf", 0.88, "2026-02-01"),
        ]
        patterns = engine.detect_recurring_patterns(incidents, min_occurrence=2)
        assert len(patterns) >= 1

        stored = repo.get_patterns()
        assert len(stored) >= 1

    def test_risk_scoring_and_persist(self, repo):
        from src.core.pattern_recognition import PatternDetectionEngine

        engine = PatternDetectionEngine(repository=repo)
        result = engine.score_document_risk(
            "test.pdf",
            {
                "max_similarity": 0.80,
                "avg_similarity": 0.65,
                "author_incident_count": 3,
                "author_avg_similarity": 0.70,
                "assignment_incident_rate": 0.2,
                "document_length": 3000,
                "submission_hour": 22,
                "days_until_deadline": 2,
                "class_section_risk": 0.3,
            },
        )

        stored = repo.get_risk_score("test.pdf")
        assert stored is not None
        assert stored["risk_score"] == result["risk_score"]
