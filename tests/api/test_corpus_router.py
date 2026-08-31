import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock exception matching your cache provider client infrastructure
class RedisConnectionError(Exception):
    """Simulated Redis connection failure exception."""
    pass

# --- Redis Failure Resilience Test Suite ---

@patch("src.api.routers.corpus.get_cache")
@patch("src.api.routers.corpus.get_sqlite_db")
def test_clear_all_documents_succeeds_when_redis_is_down(mock_get_sqlite_db, mock_get_cache):
    """
    Scenario: Verify clear_all_documents catches Redis failures and successfully purges SQLite.
    Acceptance Criteria:
    - get_cache().clear_pattern raises RedisConnectionError.
    - /api/v1/clear returns HTTP 200 OK.
    - SQLite data purging function is still successfully invoked.
    """
    # 1. Setup Mock Cache provider to simulate a connection dropout event
    mock_cache_instance = MagicMock()
    mock_cache_instance.clear_pattern.side_effect = RedisConnectionError("Redis server unreachable.")
    mock_get_cache.return_value = mock_cache_instance

    # 2. Setup Mock SQLite DB execution contexts
    mock_db_session = MagicMock()
    mock_get_sqlite_db.return_value = mock_db_session

    # 3. Initialize your application client router environment
    # Importing inline to ensure clean mock patching context limits
    from src.main import app  
    client = TestClient(app)

    # 4. Trigger the target cleanup routing endpoint
    response = client.post("/api/v1/clear")

    # 5. Assert fallback recovery and execution integrity metrics
    assert response.status_code == 200, f"Expected HTTP 200 but received {response.status_code}"
    
    # Assert cache clear attempt was initiated before dropping out
    mock_cache_instance.clear_pattern.assert_called_once()
    
    # Critical Boundary: Assert SQLite purge query execution was still dispatched completely
    mock_db_session.execute.assert_called()
    mock_db_session.commit.assert_called_once()


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

    from src.main import app
    client = TestClient(app)

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

    from src.main import app
    client = TestClient(app)

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

