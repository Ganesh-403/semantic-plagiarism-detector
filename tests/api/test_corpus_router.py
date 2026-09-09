import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock exception matching your cache provider client infrastructure
class RedisConnectionError(Exception):
    """Simulated Redis connection failure exception."""
    pass

# --- Redis Failure Resilience Test Suite ---

@pytest.fixture(autouse=True)
def scoped_admin(monkeypatch, mock_db):
    from src.api.middleware import get_valid_tokens
    from src.db.auth import add_user
    add_user("corpus-admin", "StrongPassword9!", role="admin")
    monkeypatch.setenv("API_BEARER_TOKENS_MAPPING", json.dumps({"corpus-test-token": ["admin", "read", "write"]}))
    get_valid_tokens.cache_clear()
    yield
    get_valid_tokens.cache_clear()


def test_clear_all_documents_succeeds_when_redis_is_down(tmp_path, monkeypatch):
    from src.api.app import app
    from src.db.corpus_db import add_document, get_all_documents
    import hashlib
    add_document("example.txt", hashlib.sha256(b"example").hexdigest())
    index = tmp_path / "corpus.index"
    index.write_bytes(b"index")
    monkeypatch.setattr("src.api.routers.corpus.INDEX_PATH", str(index))
    with patch("src.api.routers.corpus.get_cache") as cache:
        cache.return_value.is_available.return_value = True
        cache.return_value.clear_pattern.side_effect = RedisConnectionError("Unavailable")
        response = TestClient(app, headers={"Authorization": "Bearer corpus-test-token"}).post("/api/v1/clear?username=corpus-admin")
    assert response.status_code == 200
    assert get_all_documents() == []
    assert not index.exists()


@patch("src.api.routers.corpus.get_corpus_stats")
def test_get_corpus_stats_returns_correct_structure(mock_get_stats):
    """
    Scenario: Verify GET /api/v1/corpus/stats returns expected keys and HTTP 200 OK.
    Acceptance Criteria:
    - Returns { "total_documents": int, "total_chunks": int, "total_embeddings": int, "last_updated": str }.
    """
    expected_payload = {
        "total_documents": 42,
        "total_chunks": 128,
        "total_embeddings": 128,
        "last_updated": "2026-08-29T16:00:00+00:00",
    }
    mock_get_stats.return_value = expected_payload

    from src.api.app import app
    client = TestClient(app, headers={"Authorization": "Bearer corpus-test-token"})

    response = client.get("/api/v1/corpus/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["total_documents"] == 42
    assert data["total_chunks"] == 128
    assert data["total_embeddings"] == 128
    assert data["last_updated"] == "2026-08-29T16:00:00+00:00"


@patch("src.api.routers.corpus.get_corpus_stats")
def test_get_corpus_stats_handles_internal_error(mock_get_stats):
    """
    Scenario: Verify GET /api/v1/corpus/stats handles database or processing exceptions gracefully.
    """
    mock_get_stats.side_effect = Exception("Database connection timeout")

    from src.api.app import app
    client = TestClient(app, headers={"Authorization": "Bearer corpus-test-token"})

    response = client.get("/api/v1/corpus/stats")
    assert response.status_code == 500
    assert "Database connection timeout" in response.json()["detail"]


def test_get_corpus_stats_db_function():
    """
    Scenario: Verify direct execution of get_corpus_stats() against active DB schema.
    """
    from src.db.corpus_db import get_corpus_stats, init_corpus_db
    init_corpus_db()

    stats = get_corpus_stats()
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert "total_chunks" in stats
    assert "total_embeddings" in stats
    assert "last_updated" in stats
    assert isinstance(stats["total_documents"], int)
    assert isinstance(stats["total_chunks"], int)
    assert isinstance(stats["total_embeddings"], int)
    assert isinstance(stats["last_updated"], str)

