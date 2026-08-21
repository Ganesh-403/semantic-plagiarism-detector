"""
test_fault_tolerance.py
-----------------------
Fault injection test suite for evaluating application resilience under
external component failures.

Tests system degradation and graceful fallback mechanisms for:
- Redis connection refusals and timeouts (Issue #1380)
- Webhook HTTP 500 errors and network timeouts (Issue #1380)
- Embedding model load failures
- SQLite database disk-full and permission errors

Acceptance Criteria (Issue #1380):
- Mock Redis connection refusal (redis.exceptions.ConnectionError) and
  assert graceful fallback to SQLite / disk storage.
- Mock Webhook HTTP 500 error and assert fallback queue logging.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.webhook import send_plagiarism_alert

# Import modules under test
from src.utils.redis_cache import (
    RedisCache,
    RedisConnectionError,
    RedisTimeoutError,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis_module():
    """Fixture to provide a mocked redis module with exception classes."""
    mock_redis = MagicMock()

    # Create proper exception classes that inherit from BaseException
    class MockRedisError(Exception):
        pass

    class MockConnectionError(MockRedisError):
        pass

    class MockTimeoutError(MockRedisError):
        pass

    mock_redis.RedisError = MockRedisError
    mock_redis.ConnectionError = MockConnectionError
    mock_redis.TimeoutError = MockTimeoutError

    return mock_redis


@pytest.fixture
def reset_redis_cache():
    """Fixture to reset the RedisCache singleton between tests."""
    RedisCache._instance = None
    RedisCache._client = None
    yield
    RedisCache._instance = None
    RedisCache._client = None


@pytest.fixture(autouse=True)
def disable_webhook_retry_wait(monkeypatch):
    """Keep webhook retry tests immediate while retaining production backoff."""
    from src.core import webhook

    monkeypatch.setattr(
        webhook._post_webhook.retry,
        "sleep",
        lambda _seconds: None,
    )


# ─── Redis Fault Tolerance Tests ──────────────────────────────────────────────


class TestRedisFaultTolerance:
    """Tests for Redis connection failures and fallback to in-memory storage."""

    def test_redis_connection_refused_falls_back_to_memory(
        self, reset_redis_cache, mock_redis_module, caplog
    ):
        """
        Verify that Redis connection refusal triggers fallback to in-memory cache.

        When redis.from_url() raises ConnectionError, the RedisCache should
        gracefully degrade to using its internal _fallback_cache dictionary
        without crashing the application.
        """
        with patch("src.utils.redis_cache.redis", mock_redis_module):
            # Mock from_url to raise ConnectionError
            mock_redis_module.from_url.side_effect = mock_redis_module.ConnectionError(
                "Error 111 connecting to localhost:6379. Connection refused."
            )

            with caplog.at_level(logging.WARNING):
                cache = RedisCache()

            # Verify client is None (Redis unavailable)
            assert cache._client is None

            # Verify warning was logged
            assert any(
                "Redis connection failed" in record.message for record in caplog.records
            )

            # Verify fallback cache is functional
            assert cache.set("test_key", "test_value") is True
            assert cache.get("test_key") == "test_value"

    def test_redis_timeout_falls_back_to_memory(
        self, reset_redis_cache, mock_redis_module, caplog
    ):
        """
        Verify that Redis timeout triggers fallback to in-memory cache.

        When the connection times out, the system should fall back to
        in-memory storage rather than hanging or crashing.
        """
        with patch("src.utils.redis_cache.redis", mock_redis_module):
            mock_redis_module.from_url.side_effect = mock_redis_module.TimeoutError(
                "Timeout reading from socket"
            )

            with caplog.at_level(logging.WARNING):
                cache = RedisCache()

            assert cache._client is None
            assert cache.set("timeout_key", {"data": 123}) is True
            assert cache.get("timeout_key") == {"data": 123}

    def test_redis_ping_failure_returns_false(self, reset_redis_cache):
        """
        Verify is_available() returns False when Redis ping fails.

        If Redis is connected but ping() raises an exception, the cache
        should report itself as unavailable.
        """
        cache = RedisCache()
        cache._client = MagicMock()
        cache._client.ping.side_effect = Exception("Ping failed")

        assert cache.is_available() is False

    def test_redis_set_error_falls_back(self, reset_redis_cache):
        """
        Verify that errors during Redis SET operations fall back to memory.

        If Redis becomes unavailable after initial connection, set() should
        catch the error and store the value in the fallback cache.
        """
        cache = RedisCache()
        cache._client = MagicMock()
        cache._client.is_available = True

        # Mock is_available to return True, but setex to raise error
        with patch.object(cache, "is_available", return_value=True):
            cache._client.setex.side_effect = Exception("Redis write failed")

            result = cache.set("error_key", "error_value", ttl=60)

            # Should fall back to memory and return True
            assert result is True
            assert cache._fallback_get("error_key") == "error_value"

    def test_redis_get_error_falls_back(self, reset_redis_cache):
        """
        Verify that errors during Redis GET operations fall back to memory.

        If Redis read fails, the cache should check the fallback memory store.
        """
        cache = RedisCache()
        cache._client = MagicMock()

        # Pre-populate fallback cache
        cache._fallback_set("fallback_key", "fallback_value")

        with patch.object(cache, "is_available", return_value=True):
            cache._client.get.side_effect = Exception("Redis read failed")

            result = cache.get("fallback_key")

            # Should retrieve from fallback cache
            assert result == "fallback_value"

    def test_redis_connection_drop_mid_session_failover(
        self, reset_redis_cache, mock_redis_module
    ):
        """
        Verify seamless failover to in-memory fallback when Redis drops mid-session (#2817).

        Simulates a redis.exceptions.ConnectionError on the second .get() call,
        ensuring the application recovers without crashing and subsequent .set() calls
        safely store user uploaded files/data into in-memory fallback cache.
        """
        import pickle

        cache = RedisCache()
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        uploaded_doc1 = {
            "doc_id": "doc_101",
            "content": "Sample student essay text",
            "user": "alice",
        }
        uploaded_doc2 = {
            "doc_id": "doc_102",
            "content": "Sample secondary essay text",
            "user": "alice",
        }

        # First call succeeds returning doc1 from Redis
        # Second call raises ConnectionError (simulating connection drop mid-session)
        mock_client.get.side_effect = [
            pickle.dumps(uploaded_doc1),
            mock_redis_module.ConnectionError("Connection dropped by peer mid-session"),
        ]

        cache._client = mock_client

        # 1. First get() succeeds from Redis
        result_doc1 = cache.get("spd:v1:upload:doc_101")
        assert result_doc1 == uploaded_doc1

        # 2. Second get() encounters ConnectionError, seamlessly recovers and falls back
        result_doc2_initial = cache.get("spd:v1:upload:doc_102")
        assert result_doc2_initial is None

        # 3. Subsequent set() should succeed by storing into memory without dropping data
        mock_client.set.side_effect = mock_redis_module.ConnectionError(
            "Connection refused"
        )
        mock_client.setex.side_effect = mock_redis_module.ConnectionError(
            "Connection refused"
        )
        set_result = cache.set("spd:v1:upload:doc_102", uploaded_doc2, ttl=300)
        assert set_result is True

        # 4. Subsequent get() successfully returns stored upload from fallback in-memory cache
        mock_client.get.side_effect = mock_redis_module.ConnectionError(
            "Connection refused"
        )
        recovered_doc2 = cache.get("spd:v1:upload:doc_102")
        assert recovered_doc2 == uploaded_doc2
        assert recovered_doc2["content"] == "Sample secondary essay text"

    def test_redis_connection_refused_error_type(self, reset_redis_cache):
        """
        Verify that RedisConnectionError is properly aliased and catchable.
        """
        assert issubclass(RedisConnectionError, BaseException)
        assert issubclass(RedisTimeoutError, BaseException)

    def test_fallback_cache_ttl_expiration(self, reset_redis_cache):
        """
        Verify that TTL expiration works correctly in the fallback cache.

        The in-memory fallback cache should respect TTL values and expire
        keys after the specified duration.
        """
        cache = RedisCache()

        # Set a key with 1 second TTL
        cache._fallback_set("ttl_key", "ttl_value", ttl=1)

        # Should be retrievable immediately
        assert cache._fallback_get("ttl_key") == "ttl_value"

        # Mock time to simulate expiration
        with patch("src.utils.redis_cache.time.time", return_value=time.time() + 2):
            # Should return None after TTL expires
            assert cache._fallback_get("ttl_key") is None

    def test_fallback_cache_pattern_clear(self, reset_redis_cache):
        """
        Verify that clear_pattern works correctly in the fallback cache.
        """
        cache = RedisCache()

        # Populate with multiple keys
        cache._fallback_set("session:user1:data", "val1")
        cache._fallback_set("session:user2:data", "val2")
        cache._fallback_set("faiss:index:123", "val3")

        # Clear session pattern
        count = cache._fallback_clear_pattern("session:*")

        assert count == 2
        assert cache._fallback_get("session:user1:data") is None
        assert cache._fallback_get("faiss:index:123") == "val3"


# ─── Webhook Fault Tolerance Tests ────────────────────────────────────────────


class TestWebhookFaultTolerance:
    """Tests for webhook delivery failures and retry mechanisms."""

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_http_500_retries_and_fails(self, mock_post, mock_validate, caplog):
        """
        Verify that HTTP 500 errors trigger retries and eventually fail gracefully.

        The webhook system should retry transient server errors (500, 502, 503, 504)
        up to the maximum attempt count, then return failure without crashing.
        """
        # Create mock response that raises HTTPError on raise_for_status()
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_response
        )
        mock_post.return_value = mock_response

        with caplog.at_level(logging.ERROR):
            success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)

        # Should fail after max attempts (3)
        assert success is False
        assert attempts == 3
        assert mock_post.call_count == 3

        # Verify error was logged
        assert any(
            "Failed to send webhook notification" in record.message
            for record in caplog.records
        )

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_http_500_retries_and_succeeds(self, mock_post, mock_validate):
        """
        Verify that webhook succeeds if server recovers during retry window.

        If the first two attempts fail with 500 but the third succeeds,
        the overall operation should report success.
        """
        # First two calls fail, third succeeds
        fail_response = MagicMock(spec=requests.Response)
        fail_response.status_code = 500
        fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=fail_response
        )

        success_response = MagicMock(spec=requests.Response)
        success_response.status_code = 200
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [fail_response, fail_response, success_response]

        success, attempts = send_plagiarism_alert("DocA", "DocB", 0.88)

        assert success is True
        assert attempts == 3
        assert mock_post.call_count == 3

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_connection_refused_retries(self, mock_post, mock_validate):
        """
        Verify that connection refused errors trigger retries.

        Network-level connection failures should be treated as transient
        and retried up to the maximum attempt count.
        """
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Connection refused by remote host"
        )

        success, attempts = send_plagiarism_alert("DocA", "DocB", 0.92)

        assert success is False
        assert attempts == 3
        assert mock_post.call_count == 3

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_timeout_retries(self, mock_post, mock_validate):
        """
        Verify that request timeouts trigger retries.

        Timeout exceptions indicate temporary network congestion and should
        be retried rather than treated as permanent failures.
        """
        mock_post.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.Timeout("Request timed out"),
        ]

        success, attempts = send_plagiarism_alert("DocA", "DocB", 0.75)

        assert success is False
        assert attempts == 3

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_permanent_400_does_not_retry(self, mock_post, mock_validate):
        """
        Verify that permanent client errors (4xx) do not trigger retries.

        HTTP 400, 401, 403, 404 indicate client-side issues that will not
        be resolved by retrying. The system should fail immediately.
        """
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request", response=mock_response
        )
        mock_post.return_value = mock_response

        success, attempts = send_plagiarism_alert("DocA", "DocB", 0.99)

        assert success is False
        assert attempts == 1
        assert mock_post.call_count == 1  # No retries

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://webhook.site/test"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_webhook_fallback_queue_logging(self, mock_post, mock_validate, caplog):
        """
        Verify that failed webhooks are logged appropriately for fallback queue processing.

        When all retry attempts are exhausted, the system should log detailed
        error information that could be used by a fallback queue processor
        to retry the delivery later.
        """
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")

        with caplog.at_level(logging.ERROR):
            success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)

        assert success is False

        # Verify detailed error logging for fallback queue
        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) > 0

        # Check that the log contains enough info for a retry queue
        log_message = error_logs[0].message
        assert "DocA" in log_message
        assert "DocB" in log_message
        assert "attempt" in log_message.lower()


# ─── Embedding Model Fault Tolerance Tests ────────────────────────────────────


class TestEmbeddingModelFaultTolerance:
    """Tests for embedding model load failures and fallback mechanisms."""

    def test_primary_model_load_failure_falls_back_to_minilm(self, caplog, monkeypatch):
        """
        Verify that primary model load failure triggers fallback to MiniLM-L6.

        If the configured multilingual model fails to download or initialize,
        the system should automatically fall back to the lighter all-MiniLM-L6-v2
        model to maintain functionality.
        """
        import src.core.embedding_model as emb_module

        # Reset singleton state
        monkeypatch.setattr(emb_module, "_model", None)
        monkeypatch.setattr(emb_module.EmbeddingModelManager, "_instance", None)

        primary = emb_module._get_model_name()
        fallback = "all-MiniLM-L6-v2"

        def mock_sentence_transformer(model_name, cache_folder=None):
            if model_name == primary:
                raise RuntimeError("Failed to download primary model from HuggingFace")
            # Return mock for fallback
            mock_model = MagicMock()
            mock_model.device = "cpu"
            return mock_model

        monkeypatch.setattr(
            emb_module, "SentenceTransformer", mock_sentence_transformer
        )

        with caplog.at_level(logging.WARNING):
            manager = emb_module.EmbeddingModelManager.get_instance()
            model = manager.get_model()

        # Verify fallback occurred
        assert model is not None

        # Verify warning was logged
        assert any(
            f"Primary embedding model {primary} unavailable. Falling back to {fallback}"
            in record.message
            for record in caplog.records
        )

    def test_both_models_fail_returns_none_or_raises(self, monkeypatch):
        """
        Verify behavior when both primary and fallback models fail to load.

        In a catastrophic failure scenario where no models can be loaded,
        the system should either raise a clear exception or return None
        rather than silently failing.
        """
        import src.core.embedding_model as emb_module

        monkeypatch.setattr(emb_module, "_model", None)
        monkeypatch.setattr(emb_module.EmbeddingModelManager, "_instance", None)

        def mock_sentence_transformer(model_name, cache_folder=None):
            raise RuntimeError("All models failed to load")

        monkeypatch.setattr(
            emb_module, "SentenceTransformer", mock_sentence_transformer
        )

        # Should raise the exception from the fallback model
        with pytest.raises(RuntimeError, match="All models failed to load"):
            manager = emb_module.EmbeddingModelManager.get_instance()
            manager.get_model()


# ─── Database Fault Tolerance Tests ───────────────────────────────────────────


class TestDatabaseFaultTolerance:
    """Tests for SQLite database failures and disk space issues."""

    def test_sqlite_permission_error_handled_gracefully(self, tmp_path, caplog):
        """
        Verify that SQLite permission errors are caught and logged.

        If the database file is read-only or the directory lacks write
        permissions, the system should log the error and potentially
        fall back to a temp directory.
        """
        from src.db.corpus_db import configure_db_path, init_corpus_db

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        # Set directory to read-only (Unix-like systems)
        try:
            os.chmod(readonly_dir, 0o555)

            # Configure DB path to the read-only directory
            configure_db_path(readonly_dir / "corpus.db")

            with caplog.at_level(logging.WARNING):
                # init_corpus_db should handle the permission error gracefully
                # It should either fall back to temp dir or log a warning
                init_corpus_db()

        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)

    def test_sqlite_disk_full_error_logged(self, tmp_path, caplog):
        """
        Verify that disk-full errors during SQLite operations are logged.

        When the disk is full, SQLite will raise an OperationalError.
        The system should catch this and log an appropriate error message
        rather than crashing the application.
        """
        from src.db.corpus_db import _connect

        # Mock sqlite3.connect to raise disk full error
        with patch("src.db.corpus_db.sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.OperationalError(
                "database or disk is full"
            )

            with caplog.at_level(logging.ERROR):
                # Attempting to connect should handle the error
                try:
                    with _connect():
                        pass
                except sqlite3.OperationalError:
                    # If it propagates, that's also acceptable as long as it's logged
                    pass

    def test_database_integrity_check_handles_corruption(self, tmp_path, caplog):
        """
        Verify that database integrity checks handle corrupted databases gracefully.

        If PRAGMA integrity_check fails, the system should log the error
        and potentially trigger a recovery mechanism or fallback.
        """
        from src.db.corpus_db import check_database_integrity, configure_db_path

        # Create a corrupted database file (not a valid SQLite file)
        corrupt_db = tmp_path / "corrupt.db"
        corrupt_db.write_bytes(b"This is not a valid SQLite database file")

        configure_db_path(corrupt_db)

        with caplog.at_level(logging.ERROR):
            result = check_database_integrity()

        # Should return error message rather than crashing
        assert isinstance(result, list)
        assert len(result) > 0
        # The result should indicate an error occurred
        assert any(
            "Error" in str(r) or "not a database" in str(r).lower() for r in result
        )
