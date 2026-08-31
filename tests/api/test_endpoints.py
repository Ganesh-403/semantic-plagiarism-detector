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


def test_healthz_high_memory_default_threshold():
    """Verify that /healthz returns 503 degraded when memory exceeds default 95% threshold."""
    from unittest.mock import MagicMock
    mock_memory = MagicMock(percent=96.0, available=1024 * 1024)
    with patch("psutil.virtual_memory", return_value=mock_memory), patch.dict("os.environ", {}, clear=False):
        response = client.get("/healthz")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["memory"] == "unavailable"
        assert "warning" in data
        assert "96.0%" in data["warning"]


def test_healthz_configurable_memory_threshold():
    """Verify that HEALTHZ_MAX_MEMORY_PERCENT environment variable configures memory threshold."""
    from unittest.mock import MagicMock
    mock_memory = MagicMock(percent=85.0, available=1024 * 1024)

    # With threshold at 80%, 85% usage should trigger 503 degraded
    with patch("psutil.virtual_memory", return_value=mock_memory), patch.dict(
        "os.environ", {"HEALTHZ_MAX_MEMORY_PERCENT": "80.0"}
    ):
        response = client.get("/healthz")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["memory"] == "unavailable"
        assert "warning" in data
        assert "85.0%" in data["warning"]
        assert "80.0%" in data["warning"]

    # With threshold at 90%, 85% usage should return 200 OK
    with patch("psutil.virtual_memory", return_value=mock_memory), patch.dict(
        "os.environ", {"HEALTHZ_MAX_MEMORY_PERCENT": "90.0"}
    ):
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["memory"] == "ok"


def test_healthz_invalid_memory_threshold_fallback():
    """Verify that invalid HEALTHZ_MAX_MEMORY_PERCENT values fall back to 95.0% default."""
    from unittest.mock import MagicMock
    mock_memory = MagicMock(percent=92.0, available=1024 * 1024)

    with patch("psutil.virtual_memory", return_value=mock_memory), patch.dict(
        "os.environ", {"HEALTHZ_MAX_MEMORY_PERCENT": "invalid_number"}
    ):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"



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


def test_cache_prometheus_metrics(tmp_path):
    """Verify that translation and Redis cache hit/miss events register in Prometheus /metrics."""
    from src.db.translation_cache import get_cached_translation, save_translation, configure_db_path, init_translation_cache
    from src.utils.redis_cache import get_cache

    # 1. Setup a clean translation cache DB file
    db_file = tmp_path / "test_trans_metrics.db"
    configure_db_path(db_file)
    init_translation_cache()

    # Trigger translation miss
    get_cached_translation("Some source text", "en", "es")

    # Trigger translation hit
    save_translation("Some source text", "en", "es", "Texto traducido")
    get_cached_translation("Some source text", "en", "es")

    # Trigger Redis miss & hit
    cache = get_cache()
    cache.get("non_existent_key_xyz")
    cache.set("existent_key_xyz", "value")
    cache.get("existent_key_xyz")

    # Fetch Prometheus metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Assert metric counters are defined in the exporter output
    assert "spd_cache_hits_total" in content
    assert "spd_cache_misses_total" in content
    
    # Assert labels are present
    assert 'cache_type="translation"' in content
    assert 'cache_type="redis"' in content

