"""tests/api/test_health_score.py

Unit tests for the Document Health Scoring API endpoints and scoring engine.
Tests cover scoring, listing, quality gate, analytics, and the scorer logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.db.health_score_db import HealthScoreRepository, init_health_score_db

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _init_db(tmp_path):
    """Initialise an in-memory test database for each test."""
    import src.db.health_score_db as mod
    mod.configure_db_path(str(tmp_path / "test_health.db"))
    init_health_score_db()
    yield
    mod.close_connections(all_threads=True)


@pytest.fixture()
def auth_headers():
    """Return valid Bearer token headers."""
    token = get_expected_bearer_token()
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Scoring Engine Unit Tests
# ---------------------------------------------------------------------------


class TestScoringEngine:
    """Test the document health scoring engine directly."""

    def test_score_document_basic(self):
        from src.core.document_health_scorer import score_document

        doc = {
            "filename": "test.pdf",
            "file_hash": "abc123",
            "student_name": "Alice",
            "class_section": "CS-401",
            "assignment_title": "Essay 1",
            "detected_language": "English",
            "tags": "NLP, AI",
        }
        report = score_document(
            doc=doc,
            chunk_texts=["This is a test chunk with enough words to pass quality checks. " * 10],
            chunk_word_counts=[100],
            total_chunks=1,
            chunks_with_embeddings=1,
            total_words=100,
            existing_hashes=set(),
        )

        assert report.filename == "test.pdf"
        assert report.overall_score > 50
        assert report.grade in ["A+", "A", "A-", "B+", "B", "B-", "C", "D", "F"]
        assert len(report.dimensions) == 5
        assert report.checked_at is not None

    def test_score_document_empty(self):
        from src.core.document_health_scorer import score_document

        doc = {"filename": "empty.pdf", "file_hash": ""}
        report = score_document(
            doc=doc,
            chunk_texts=[],
            chunk_word_counts=[],
            total_chunks=0,
            chunks_with_embeddings=0,
            total_words=0,
        )

        assert report.overall_score < 50
        assert report.grade in ["D", "F"]

    def test_score_document_duplicate_hash(self):
        from src.core.document_health_scorer import score_document

        doc = {
            "filename": "dup.pdf",
            "file_hash": "same_hash",
            "student_name": "Bob",
            "class_section": "CS-401",
            "assignment_title": "Essay",
            "detected_language": "English",
            "tags": "",
        }
        report = score_document(
            doc=doc,
            chunk_texts=["Some content " * 50],
            chunk_word_counts=[50],
            total_chunks=1,
            chunks_with_embeddings=1,
            total_words=50,
            existing_hashes={"same_hash"},
        )

        # Fingerprint dimension should be 0
        fp_dim = next(d for d in report.dimensions if d.name == "fingerprint_uniqueness")
        assert fp_dim.score == 0.0

    def test_quality_gate_pass(self):
        from src.core.document_health_scorer import HealthReport, compute_quality_gate

        report = HealthReport(
            filename="good.pdf",
            overall_score=85.0,
            grade="B",
            dimensions=[],
            checked_at="2026-08-23T00:00:00",
        )
        gate = compute_quality_gate(report, min_score=60.0, min_grade="D")
        assert gate["passed"] is True

    def test_quality_gate_fail_low_score(self):
        from src.core.document_health_scorer import HealthReport, compute_quality_gate

        report = HealthReport(
            filename="bad.pdf",
            overall_score=45.0,
            grade="F",
            dimensions=[],
            checked_at="2026-08-23T00:00:00",
        )
        gate = compute_quality_gate(report, min_score=60.0, min_grade="D")
        assert gate["passed"] is False
        assert "below minimum" in gate["reason"]

    def test_quality_gate_fail_low_grade(self):
        from src.core.document_health_scorer import HealthReport, compute_quality_gate

        report = HealthReport(
            filename="bad.pdf",
            overall_score=65.0,
            grade="D",
            dimensions=[],
            checked_at="2026-08-23T00:00:00",
        )
        gate = compute_quality_gate(report, min_score=60.0, min_grade="B-")
        assert gate["passed"] is False
        assert "Grade D is below minimum" in gate["reason"]

    def test_aggregate_reports(self):
        from src.core.document_health_scorer import HealthReport, aggregate_reports

        reports = [
            HealthReport("a.pdf", 90.0, "A", [], "2026-01-01", {}),
            HealthReport("b.pdf", 70.0, "C", [], "2026-01-01", {}),
            HealthReport("c.pdf", 85.0, "B+", [], "2026-01-01", {}),
        ]
        agg = aggregate_reports(reports)

        assert agg["count"] == 3
        assert agg["avg_score"] == 81.67
        assert agg["min_score"] == 70.0
        assert agg["max_score"] == 90.0

    def test_aggregate_empty(self):
        from src.core.document_health_scorer import aggregate_reports

        agg = aggregate_reports([])
        assert agg["count"] == 0
        assert agg["avg_score"] == 0.0


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Verify auth enforcement on health score endpoints."""

    def test_list_scores_no_auth(self):
        response = client.get("/api/v1/health/scores")
        assert response.status_code in (401, 403)

    def test_list_scores_invalid_token(self):
        response = client.get(
            "/api/v1/health/scores",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    def test_score_document_no_auth(self):
        response = client.post(
            "/api/v1/health/score",
            params={"filename": "test.pdf"},
        )
        assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestHealthScoreAPI:
    """Test the health score API endpoints."""

    @patch("src.api.routers.health_score.health_repo")
    def test_list_scores_success(self, mock_repo, auth_headers):
        mock_repo.list_scores.return_value = [
            {"id": 1, "filename": "a.pdf", "overall_score": 85.0, "grade": "B"},
            {"id": 2, "filename": "b.pdf", "overall_score": 45.0, "grade": "F"},
        ]
        mock_repo.count_scores.return_value = 2

        response = client.get(
            "/api/v1/health/scores",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) == 2
        assert data["total_items"] == 2

    @patch("src.api.routers.health_score.health_repo")
    def test_list_scores_with_filters(self, mock_repo, auth_headers):
        mock_repo.list_scores.return_value = []
        mock_repo.count_scores.return_value = 0

        response = client.get(
            "/api/v1/health/scores",
            params={"min_score": 80, "grade": "A"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        mock_repo.list_scores.assert_called_once()

    @patch("src.api.routers.health_score.health_repo")
    def test_get_document_score_success(self, mock_repo, auth_headers):
        mock_repo.get_latest_score.return_value = {
            "id": 1,
            "filename": "a.pdf",
            "overall_score": 85.0,
            "grade": "B",
            "dimension_data": [],
            "checked_at": "2026-08-23T10:00:00",
            "gate_passed": 1,
            "gate_reason": "Passed",
        }

        response = client.get(
            "/api/v1/health/scores/a.pdf",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 85.0
        assert data["grade"] == "B"

    @patch("src.api.routers.health_score.health_repo")
    def test_get_document_score_not_found(self, mock_repo, auth_headers):
        mock_repo.get_latest_score.return_value = None

        response = client.get(
            "/api/v1/health/scores/nonexistent.pdf",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @patch("src.api.routers.health_score.health_repo")
    def test_get_score_history(self, mock_repo, auth_headers):
        mock_repo.get_score_history.return_value = [
            {"id": 1, "filename": "a.pdf", "overall_score": 80.0},
            {"id": 2, "filename": "a.pdf", "overall_score": 85.0},
        ]

        response = client.get(
            "/api/v1/health/scores/a.pdf/history",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

    @patch("src.api.routers.health_score.health_repo")
    def test_get_gate_config(self, mock_repo, auth_headers):
        mock_repo.get_gate_config.return_value = {
            "min_score": "60.0",
            "min_grade": "D",
            "enabled": "true",
        }

        response = client.get(
            "/api/v1/health/gate/config",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["min_score"] == 60.0
        assert data["enabled"] is True

    @patch("src.api.routers.health_score.health_repo")
    def test_update_gate_config(self, mock_repo, auth_headers):
        mock_repo.set_gate_config.return_value = None
        mock_repo.get_gate_config.return_value = {
            "min_score": "75.0",
            "min_grade": "B-",
            "enabled": "true",
        }

        response = client.put(
            "/api/v1/health/gate/config",
            params={"min_score": 75.0, "min_grade": "B-"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["min_score"] == 75.0

    @patch("src.api.routers.health_score.health_repo")
    def test_check_quality_gate_pass(self, mock_repo, auth_headers):
        mock_repo.get_latest_score.return_value = {
            "overall_score": 85.0,
            "grade": "B",
            "checked_at": "2026-08-23T10:00:00",
        }
        mock_repo.get_gate_config.return_value = {
            "min_score": "60.0",
            "min_grade": "D",
        }

        response = client.post(
            "/api/v1/health/gate/check",
            params={"filename": "good.pdf"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is True

    @patch("src.api.routers.health_score.health_repo")
    def test_check_quality_gate_fail(self, mock_repo, auth_headers):
        mock_repo.get_latest_score.return_value = {
            "overall_score": 45.0,
            "grade": "F",
            "checked_at": "2026-08-23T10:00:00",
        }
        mock_repo.get_gate_config.return_value = {
            "min_score": "60.0",
            "min_grade": "D",
        }

        response = client.post(
            "/api/v1/health/gate/check",
            params={"filename": "bad.pdf"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is False

    @patch("src.api.routers.health_score.health_repo")
    def test_get_health_summary(self, mock_repo, auth_headers):
        mock_repo.get_score_summary.return_value = {
            "total_scored": 42,
            "avg_score": 78.4,
            "min_score": 24.5,
            "max_score": 97.2,
            "passed_gate": 35,
            "failed_gate": 7,
            "pass_rate": 83.3,
            "grade_distribution": {"A": 5, "B": 7, "C": 5, "F": 3},
            "last_checked_at": "2026-08-23T10:00:00",
        }

        response = client.get(
            "/api/v1/health/analytics/summary",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_scored"] == 42
        assert data["pass_rate"] == 83.3

    @patch("src.api.routers.health_score.health_repo")
    def test_get_dimension_averages(self, mock_repo, auth_headers):
        mock_repo.get_dimension_averages.return_value = {
            "metadata_completeness": 72.5,
            "chunk_balance": 74.8,
        }

        response = client.get(
            "/api/v1/health/analytics/dimensions",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dimensions"]["metadata_completeness"] == 72.5

    @patch("src.api.routers.health_score.health_repo")
    def test_get_worst_documents(self, mock_repo, auth_headers):
        mock_repo.get_worst_documents.return_value = [
            {"filename": "bad.pdf", "overall_score": 24.5},
        ]

        response = client.get(
            "/api/v1/health/analytics/worst",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    @patch("src.api.routers.health_score.health_repo")
    def test_get_best_documents(self, mock_repo, auth_headers):
        mock_repo.get_best_documents.return_value = [
            {"filename": "excellent.pdf", "overall_score": 97.2},
        ]

        response = client.get(
            "/api/v1/health/analytics/best",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    @patch("src.api.routers.health_score.health_repo")
    def test_delete_document_scores(self, mock_repo, auth_headers):
        mock_repo.delete_scores_for_document.return_value = 3

        response = client.delete(
            "/api/v1/health/scores/old_doc.pdf",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["records_deleted"] == 3
