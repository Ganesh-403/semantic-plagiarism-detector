from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_healthz_endpoint():
    """Verify that GET /healthz returns 200 OK and status 'ok'."""
    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "db_size_bytes" in data
    assert "db_size_mb" in data
    assert isinstance(data["db_size_bytes"], int)
    assert isinstance(data["db_size_mb"], float)


def test_healthz_db_not_exist():
    """Verify that /healthz handles missing databases gracefully."""
    with patch("os.path.exists", return_value=False):
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["db_size_bytes"] == 0
        assert data["db_size_mb"] == 0.0


def test_metrics_prometheus_endpoint():
    """Verify that GET /metrics returns Prometheus format metrics in plain text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    content = response.text
    # Standard prometheus client metric outputs contain python_info or other telemetry
    assert len(content) > 0


def test_metrics_json_endpoint():
    """Verify that GET /metrics/json returns valid JSON metrics."""
    response = client.get("/metrics/json")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")

    data = response.json()
    assert isinstance(data, dict)
