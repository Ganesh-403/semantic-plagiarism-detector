"""tests/api/test_anomalies.py

Unit tests for the Anomaly Detection API endpoints.
Tests cover scan lifecycle, alert CRUD, analytics, and configuration.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.db.anomaly_alerts_db import AnomalyAlertRepository, init_anomaly_alerts_db

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch):
    """Bypass bearer-token auth for every test."""
    monkeypatch.setattr(
        "src.api.middleware.get_expected_bearer_token",
        lambda: "test-token",
    )


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point the DB at a fresh temp file for each test."""
    db_path = str(tmp_path / "anomaly_test.db")
    monkeypatch.setattr(
        "src.db.anomaly_alerts_db.DEFAULT_DB_PATH",
        __import__("pathlib").Path(db_path),
    )
    init_anomaly_alerts_db(__import__("pathlib").Path(db_path))


HEADERS = {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# Scan Lifecycle Tests
# ---------------------------------------------------------------------------

class TestScanLifecycle:
    """Tests for scan creation, completion, and failure."""

    def test_create_scan(self):
        """Creating a scan should succeed."""
        resp = client.post(
            "/api/v1/anomalies/scans",
            headers=HEADERS,
            params={"scan_type": "full"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "scan_id" in data
        assert data["status"] == "running"

    def test_list_scans(self):
        """Listing scans should return results."""
        client.post("/api/v1/anomalies/scans", headers=HEADERS)
        resp = client.get("/api/v1/anomalies/scans", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_get_scan(self):
        """Getting a scan by ID should return details."""
        create = client.post("/api/v1/anomalies/scans", headers=HEADERS)
        scan_id = create.json()["scan_id"]
        resp = client.get(f"/api/v1/anomalies/scans/{scan_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["id"] == scan_id

    def test_complete_scan(self):
        """Completing a scan should update its status."""
        create = client.post("/api/v1/anomalies/scans", headers=HEADERS)
        scan_id = create.json()["scan_id"]
        resp = client.post(
            f"/api/v1/anomalies/scans/{scan_id}/complete",
            headers=HEADERS,
            params={"documents_scanned": 50, "anomalies_found": 3},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_fail_scan(self):
        """Failing a scan should update its status."""
        create = client.post("/api/v1/anomalies/scans", headers=HEADERS)
        scan_id = create.json()["scan_id"]
        resp = client.post(
            f"/api/v1/anomalies/scans/{scan_id}/fail",
            headers=HEADERS,
            params={"error_message": "Timeout"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    def test_get_scan_not_found(self):
        """Getting non-existent scan should return 404."""
        resp = client.get("/api/v1/anomalies/scans/99999", headers=HEADERS)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Alert CRUD Tests
# ---------------------------------------------------------------------------

class TestAlertCRUD:
    """Tests for alert creation, listing, and retrieval."""

    def test_create_alert(self):
        """Creating an alert should succeed."""
        resp = client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={
                "anomaly_type": "collusion",
                "severity": "high",
                "title": "Suspicious collusion",
                "description": "Two students submitted identical work",
                "confidence": 0.92,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["anomaly_type"] == "collusion"
        assert data["severity"] == "high"

    def test_list_alerts(self):
        """Listing alerts should return paginated results."""
        client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "outlier", "severity": "medium", "title": "Test alert"},
        )
        resp = client.get("/api/v1/anomalies/alerts", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_get_alert(self):
        """Getting an alert by ID should return details."""
        create = client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "template", "severity": "low", "title": "Template detected"},
        )
        alert_id = create.json()["alert_id"]
        resp = client.get(f"/api/v1/anomalies/alerts/{alert_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["id"] == alert_id

    def test_acknowledge_alert(self):
        """Acknowledging an alert should update its status."""
        create = client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "pattern", "severity": "info", "title": "Pattern found"},
        )
        alert_id = create.json()["alert_id"]
        resp = client.put(f"/api/v1/anomalies/alerts/{alert_id}/acknowledge", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["acknowledged"] is True

    def test_resolve_alert(self):
        """Resolving an alert should update its status."""
        create = client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "statistical", "severity": "medium", "title": "Stat anomaly"},
        )
        alert_id = create.json()["alert_id"]
        resp = client.put(
            f"/api/v1/anomalies/alerts/{alert_id}/resolve",
            headers=HEADERS,
            params={"notes": "Investigated — false positive"},
        )
        assert resp.status_code == 200
        assert resp.json()["resolved"] is True

    def test_acknowledge_all(self):
        """Acknowledge-all should acknowledge all pending alerts."""
        for i in range(3):
            client.post(
                "/api/v1/anomalies/alerts",
                headers=HEADERS,
                params={"anomaly_type": "outlier", "severity": "low", "title": f"Alert {i}"},
            )
        resp = client.put("/api/v1/anomalies/alerts/read-all", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["acknowledged_count"] >= 3

    def test_delete_alert(self):
        """Deleting an alert should succeed."""
        create = client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "cluster", "severity": "info", "title": "To delete"},
        )
        alert_id = create.json()["alert_id"]
        resp = client.delete(f"/api/v1/anomalies/alerts/{alert_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_get_alert_not_found(self):
        """Getting non-existent alert should return 404."""
        resp = client.get("/api/v1/anomalies/alerts/99999", headers=HEADERS)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analytics Tests
# ---------------------------------------------------------------------------

class TestAnalytics:
    """Tests for analytics endpoints."""

    def test_summary(self):
        """Summary should return valid statistics."""
        resp = client.get("/api/v1/anomalies/analytics/summary", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_alerts" in data
        assert "unresolved" in data

    def test_severity_distribution(self):
        """Severity distribution should return counts per level."""
        resp = client.get("/api/v1/anomalies/analytics/severity", headers=HEADERS)
        assert resp.status_code == 200
        assert "distribution" in resp.json()

    def test_type_distribution(self):
        """Type distribution should return counts per type."""
        resp = client.get("/api/v1/anomalies/analytics/types", headers=HEADERS)
        assert resp.status_code == 200
        assert "distribution" in resp.json()

    def test_high_confidence(self):
        """High-confidence endpoint should return relevant alerts."""
        client.post(
            "/api/v1/anomalies/alerts",
            headers=HEADERS,
            params={"anomaly_type": "collusion", "severity": "critical", "title": "Critical", "confidence": 0.95},
        )
        resp = client.get(
            "/api/v1/anomalies/analytics/high-confidence?min_confidence=0.8",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert "alerts" in resp.json()


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for anomaly detection configuration."""

    def test_get_config(self):
        """Getting config should return default values."""
        resp = client.get("/api/v1/anomalies/config", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["z_score_threshold"] == 2.5

    def test_update_config(self):
        """Updating config should persist changes."""
        resp = client.put(
            "/api/v1/anomalies/config",
            headers=HEADERS,
            params={"z_score_threshold": 3.0, "cluster_min_size": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["z_score_threshold"] == 3.0
        assert resp.json()["cluster_min_size"] == 5


# ---------------------------------------------------------------------------
# Auth Tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Tests for authentication requirements."""

    def test_no_auth(self):
        """Requests without auth should be rejected."""
        assert client.get("/api/v1/anomalies/scans").status_code in (401, 403)
        assert client.get("/api/v1/anomalies/alerts").status_code in (401, 403)

    def test_wrong_token(self):
        """Requests with wrong token should be rejected."""
        h = {"Authorization": "Bearer wrong"}
        assert client.get("/api/v1/anomalies/scans", headers=h).status_code in (401, 403)

    def test_create_alert_no_auth(self):
        """POST without auth should be rejected."""
        resp = client.post(
            "/api/v1/anomalies/alerts",
            params={"anomaly_type": "outlier", "severity": "low", "title": "X"},
        )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# DB Repository Unit Tests
# ---------------------------------------------------------------------------

class TestAnomalyDB:
    """Unit tests for AnomalyAlertRepository."""

    def test_create_and_get_scan(self):
        """Should create and retrieve a scan."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            scan_id = repo.create_scan(scan_type="full", triggered_by="test")
            assert scan_id > 0

            scan = repo.get_scan(scan_id)
            assert scan is not None
            assert scan["status"] == "running"

    def test_complete_scan(self):
        """Should mark scan as completed."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            scan_id = repo.create_scan()
            repo.complete_scan(scan_id, documents_scanned=100, anomalies_found=5)
            scan = repo.get_scan(scan_id)
            assert scan["status"] == "completed"
            assert scan["anomalies_found"] == 5

    def test_create_and_acknowledge_alert(self):
        """Should create, acknowledge, and resolve alerts."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            alert_id = repo.create_alert(
                scan_id=None,
                anomaly_type="collusion",
                severity="high",
                title="Test alert",
                confidence=0.9,
            )
            assert alert_id > 0

            acked = repo.acknowledge_alert(alert_id, by="analyst")
            assert acked is True

            resolved = repo.resolve_alert(alert_id, by="analyst", notes="Done")
            assert resolved is True

            alert = repo.get_alert(alert_id)
            assert alert["is_acknowledged"] == 1
            assert alert["is_resolved"] == 1

    def test_list_alerts_with_filters(self):
        """Should filter alerts by severity and type."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            repo.create_alert(None, "outlier", "low", "Low alert")
            repo.create_alert(None, "collusion", "critical", "Critical alert")
            repo.create_alert(None, "outlier", "medium", "Medium alert")

            low = repo.list_alerts(severity="low")
            assert low["total"] == 1

            outliers = repo.list_alerts(anomaly_type="outlier")
            assert outliers["total"] == 2

    def test_analytics_summary(self):
        """Should return correct analytics."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            for i in range(5):
                repo.create_alert(None, "outlier", "high", f"Alert {i}", confidence=0.8)

            summary = repo.analytics_summary()
            assert summary["total_alerts"] == 5

    def test_severity_distribution(self):
        """Should return correct severity counts."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            repo.create_alert(None, "outlier", "critical", "C1")
            repo.create_alert(None, "outlier", "critical", "C2")
            repo.create_alert(None, "pattern", "low", "L1")

            dist = repo.severity_distribution()
            assert dist.get("critical") == 2
            assert dist.get("low") == 1

    def test_acknowledge_all(self):
        """Should acknowledge all unacknowledged alerts."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            for i in range(4):
                repo.create_alert(None, "outlier", "info", f"Alert {i}")

            count = repo.acknowledge_all()
            assert count == 4

            summary = repo.analytics_summary()
            assert summary["unacknowledged"] == 0

    def test_config_management(self):
        """Should get and update config."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            config = repo.get_config()
            assert config["z_score_threshold"] == 2.5

            updated = repo.update_config(z_score_threshold=3.5, cluster_min_size=5)
            assert updated["z_score_threshold"] == 3.5
            assert updated["cluster_min_size"] == 5

    def test_purge_old_scans(self):
        """Should keep only the most recent scans."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_anomaly_alerts_db(db)
            repo = AnomalyAlertRepository(db)

            for i in range(10):
                repo.create_scan()

            purged = repo.purge_old_scans(keep_count=3)
            assert purged == 7
