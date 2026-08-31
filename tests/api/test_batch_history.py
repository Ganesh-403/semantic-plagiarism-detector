"""tests/api/test_batch_history.py

Unit tests for the Batch Analysis History API endpoints.
Tests cover CRUD operations for batch runs, timeline events, alerts, and analytics.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.db.batch_history_db import BatchHistoryRepository, init_batch_history_db

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _init_db(tmp_path):
    """Initialise an in-memory test database for each test."""
    import src.db.batch_history_db as mod
    mod.configure_db_path(str(tmp_path / "test_batch.db"))
    init_batch_history_db()
    yield
    mod.close_connections(all_threads=True)


@pytest.fixture()
def auth_headers():
    """Return valid Bearer token headers."""
    token = get_expected_bearer_token()
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Verify auth enforcement on batch history endpoints."""

    def test_list_runs_no_auth(self):
        response = client.get("/api/v1/batch/runs")
        assert response.status_code in (401, 403)

    def test_list_runs_invalid_token(self):
        response = client.get(
            "/api/v1/batch/runs",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    def test_create_run_no_auth(self):
        response = client.post("/api/v1/batch/runs")
        assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Batch Run CRUD Tests
# ---------------------------------------------------------------------------


class TestBatchRunCRUD:
    """Test create, read, update, delete operations for batch runs."""

    @patch("src.api.routers.batch_history.batch_repo")
    def test_create_batch_run_success(self, mock_repo, auth_headers):
        mock_repo.create_batch_run.return_value = 1
        mock_repo.add_timeline_event.return_value = 1

        response = client.post(
            "/api/v1/batch/runs",
            params={"threshold": 0.75, "trigger": "manual"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["run_id"] == 1
        assert data["status"] == "running"
        mock_repo.create_batch_run.assert_called_once()

    @patch("src.api.routers.batch_history.batch_repo")
    def test_create_batch_run_database_error(self, mock_repo, auth_headers):
        mock_repo.create_batch_run.side_effect = Exception("Database locked")

        response = client.post(
            "/api/v1/batch/runs",
            headers=auth_headers,
        )
        assert response.status_code == 500
        assert "Database locked" in response.json()["detail"]

    @patch("src.api.routers.batch_history.batch_repo")
    def test_list_batch_runs_success(self, mock_repo, auth_headers):
        mock_repo.list_batch_runs.return_value = [
            {"run_id": 1, "status": "completed", "documents_scanned": 100},
            {"run_id": 2, "status": "failed", "documents_scanned": 50},
        ]
        mock_repo.count_batch_runs.return_value = 2

        response = client.get(
            "/api/v1/batch/runs",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 2
        assert data["total_items"] == 2

    @patch("src.api.routers.batch_history.batch_repo")
    def test_list_batch_runs_with_filters(self, mock_repo, auth_headers):
        mock_repo.list_batch_runs.return_value = []
        mock_repo.count_batch_runs.return_value = 0

        response = client.get(
            "/api/v1/batch/runs",
            params={"status": "failed", "trigger_source": "api"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        mock_repo.list_batch_runs.assert_called_once_with(
            status="failed",
            trigger_source="api",
            start_date=None,
            end_date=None,
            limit=20,
            offset=0,
        )

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_batch_run_detail_success(self, mock_repo, auth_headers):
        mock_repo.get_batch_run.return_value = {
            "run_id": 1,
            "status": "completed",
            "documents_scanned": 100,
        }
        mock_repo.get_batch_documents.return_value = [
            {"id": 1, "document_name": "test.pdf", "similarity_score": 0.85}
        ]
        mock_repo.get_severity_distribution.return_value = {"high": 1, "none": 99}

        response = client.get(
            "/api/v1/batch/runs/1",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["run"]["run_id"] == 1
        assert len(data["documents"]) == 1
        assert data["severity_distribution"]["high"] == 1

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_batch_run_not_found(self, mock_repo, auth_headers):
        mock_repo.get_batch_run.return_value = None

        response = client.get(
            "/api/v1/batch/runs/999",
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("src.api.routers.batch_history.batch_repo")
    def test_complete_batch_run_success(self, mock_repo, auth_headers):
        mock_repo.get_batch_run.return_value = {"run_id": 1, "status": "running"}
        mock_repo.add_timeline_event.return_value = 1
        mock_repo.create_alert.return_value = 1

        response = client.post(
            "/api/v1/batch/runs/1/complete",
            params={
                "documents_scanned": 150,
                "documents_flagged": 12,
                "avg_similarity": 0.23,
                "max_similarity": 0.87,
                "duration_ms": 180000,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        mock_repo.complete_batch_run.assert_called_once()

    @patch("src.api.routers.batch_history.batch_repo")
    def test_complete_batch_run_creates_alert_for_high_plagiarism(self, mock_repo, auth_headers):
        mock_repo.get_batch_run.return_value = {"run_id": 1, "status": "running"}
        mock_repo.add_timeline_event.return_value = 1
        mock_repo.create_alert.return_value = 1

        response = client.post(
            "/api/v1/batch/runs/1/complete",
            params={
                "documents_scanned": 100,
                "documents_flagged": 5,
                "avg_similarity": 0.5,
                "max_similarity": 0.95,
                "duration_ms": 120000,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        # Should create an alert for high similarity
        mock_repo.create_alert.assert_called()

    @patch("src.api.routers.batch_history.batch_repo")
    def test_fail_batch_run_success(self, mock_repo, auth_headers):
        mock_repo.get_batch_run.return_value = {"run_id": 1, "status": "running"}
        mock_repo.add_timeline_event.return_value = 1
        mock_repo.create_alert.return_value = 1

        response = client.post(
            "/api/v1/batch/runs/1/fail",
            params={"error_message": "Timeout"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        mock_repo.fail_batch_run.assert_called_once_with(1, "Timeout")

    @patch("src.api.routers.batch_history.batch_repo")
    def test_delete_batch_run_success(self, mock_repo, auth_headers):
        mock_repo.delete_batch_run.return_value = True

        response = client.delete(
            "/api/v1/batch/runs/1",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    @patch("src.api.routers.batch_history.batch_repo")
    def test_delete_batch_run_not_found(self, mock_repo, auth_headers):
        mock_repo.delete_batch_run.return_value = False

        response = client.delete(
            "/api/v1/batch/runs/999",
            headers=auth_headers,
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Timeline Tests
# ---------------------------------------------------------------------------


class TestTimeline:
    """Test timeline event retrieval endpoints."""

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_timeline_success(self, mock_repo, auth_headers):
        mock_repo.get_timeline_events.return_value = [
            {
                "event_id": 1,
                "run_id": 1,
                "event_type": "batch_started",
                "severity": "info",
                "message": "Batch started",
                "metadata": None,
                "created_at": "2026-08-23T10:00:00",
            }
        ]

        response = client.get(
            "/api/v1/batch/timeline",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["event_type"] == "batch_started"

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_timeline_filtered(self, mock_repo, auth_headers):
        mock_repo.get_timeline_events.return_value = []

        response = client.get(
            "/api/v1/batch/timeline",
            params={"run_id": 1, "severity": "error"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        mock_repo.get_timeline_events.assert_called_once_with(
            run_id=1,
            event_type=None,
            severity="error",
            start_date=None,
            limit=50,
        )


# ---------------------------------------------------------------------------
# Alerts Tests
# ---------------------------------------------------------------------------


class TestAlerts:
    """Test alert retrieval and management endpoints."""

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_alerts_success(self, mock_repo, auth_headers):
        mock_repo.get_alerts.return_value = [
            {
                "alert_id": 1,
                "run_id": 1,
                "alert_type": "high_plagiarism",
                "title": "High plagiarism",
                "message": "93% similarity detected",
                "is_read": 0,
                "created_at": "2026-08-23T10:00:00",
            }
        ]

        response = client.get(
            "/api/v1/batch/alerts",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["alert_type"] == "high_plagiarism"

    @patch("src.api.routers.batch_history.batch_repo")
    def test_mark_all_alerts_read(self, mock_repo, auth_headers):
        mock_repo.mark_all_alerts_read.return_value = 5

        response = client.put(
            "/api/v1/batch/alerts/read-all",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["marked_read"] == 5

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_unread_alert_count(self, mock_repo, auth_headers):
        mock_repo.get_unread_alert_count.return_value = 3

        response = client.get(
            "/api/v1/batch/alerts/unread-count",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["unread_count"] == 3


# ---------------------------------------------------------------------------
# Analytics Tests
# ---------------------------------------------------------------------------


class TestAnalytics:
    """Test analytics and summary endpoints."""

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_summary_success(self, mock_repo, auth_headers):
        mock_repo.get_summary_stats.return_value = {
            "total_runs": 47,
            "completed_runs": 41,
            "failed_runs": 4,
            "success_rate": 87.2,
            "total_documents_scanned": 6842,
            "total_documents_flagged": 312,
            "avg_similarity": 0.21,
            "avg_duration_ms": 165000,
            "last_run_at": "2026-08-23T10:00:00",
        }

        response = client.get(
            "/api/v1/batch/analytics/summary",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 47
        assert data["success_rate"] == 87.2

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_trends_success(self, mock_repo, auth_headers):
        mock_repo.get_trend_data.return_value = [
            {
                "scan_date": "2026-08-23",
                "total_runs": 3,
                "total_docs_scanned": 450,
                "total_docs_flagged": 22,
                "avg_similarity": 0.19,
                "peak_similarity": 0.87,
                "avg_duration_ms": 180000,
            }
        ]

        response = client.get(
            "/api/v1/batch/analytics/trends",
            params={"days": 30},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["total_runs"] == 3

    @patch("src.api.routers.batch_history.batch_repo")
    def test_get_severity_distribution(self, mock_repo, auth_headers):
        mock_repo.get_severity_distribution.return_value = {
            "high": 5,
            "medium": 10,
            "low": 15,
            "none": 200,
        }

        response = client.get(
            "/api/v1/batch/analytics/severity",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["distribution"]["high"] == 5
        assert data["distribution"]["none"] == 200


# ---------------------------------------------------------------------------
# Maintenance Tests
# ---------------------------------------------------------------------------


class TestMaintenance:
    """Test purge and maintenance endpoints."""

    @patch("src.api.routers.batch_history.batch_repo")
    def test_purge_old_runs_success(self, mock_repo, auth_headers):
        mock_repo.purge_old_runs.return_value = 12
        mock_repo.add_timeline_event.return_value = 1

        response = client.post(
            "/api/v1/batch/purge",
            params={"days": 90},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 12
        assert data["days_threshold"] == 90
