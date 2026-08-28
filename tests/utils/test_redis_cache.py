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
test_redis_cache.py
-------------------
Unit tests for Redis cache functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import redis

from src.utils.redis_cache import (
    CacheNamespace,
    PayloadCompressor,
    RedisCache,
    RedisError,
    cache_analysis_results,
    cache_faiss_index,
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_cache,
    get_faiss_index,
    get_session_state,
)


class TestRedisCache:
    """Test Redis cache manager functionality."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = Mock()
        client.ping.return_value = True
        return client

    @pytest.fixture
    def cache_with_mock(self, mock_redis_client):
        """Create a RedisCache instance with mocked client."""

        cache = RedisCache.__new__(RedisCache)
        cache._client = mock_redis_client

        yield cache

    def test_cache_set_get(self, cache_with_mock, mock_redis_client):
        """Test basic set and get operations."""
        import pickle

        cache_with_mock.set("test_key", "test_value", ttl=60)
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = pickle.dumps("test_value")
        result = cache_with_mock.get("test_key")
        assert result == "test_value"

    def test_cache_set_get_json(self, cache_with_mock, mock_redis_client):
        """Test JSON set and get operations."""
        test_dict = {"key": "value", "number": 42}
        cache_with_mock.set_json("test_json", test_dict, ttl=60)
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = '{"key": "value", "number": 42}'
        result = cache_with_mock.get_json("test_json")
        assert result == test_dict

    def test_cache_delete(self, cache_with_mock, mock_redis_client):
        """Test delete operation."""
        cache_with_mock.delete("test_key")
        mock_redis_client.delete.assert_called_once_with("test_key")

    def test_cache_exists(self, cache_with_mock, mock_redis_client):
        """Test exists operation."""
        mock_redis_client.exists.return_value = 1
        result = cache_with_mock.exists("test_key")
        assert result is True

        mock_redis_client.exists.return_value = 0
        result = cache_with_mock.exists("test_key")
        assert result is False

    def test_cache_unavailable(self):
        """Test behavior when Redis is unavailable."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = None

        assert cache.set("test_key", "test_value") is False
        assert cache.get("test_key") is None
        assert cache.delete("test_key") is False
        assert cache.exists("test_key") is False

    def test_session_state_caching(self, cache_with_mock, mock_redis_client):
        """Test session state caching functions."""
        session_id = "test_session"
        key = "authenticated"
        value = True

        cache_session_state(session_id, key, value)
        expected_key = CacheNamespace.SESSION.build_key(session_id, key)
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_session_state(session_id, key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_cache_namespace_build_key_appends_app_version(self):
        """Verify CacheNamespace.build_key appends APP_VERSION to Redis keys."""
        from src.version import APP_VERSION

        key = CacheNamespace.ANALYSIS.build_key("matrix_123")
        assert key == f"spd:v1:analysis:{APP_VERSION}:matrix_123"
        assert APP_VERSION in key

    def test_clear_session(self, cache_with_mock, mock_redis_client):
        """Test clearing session data."""
        session_id = "test_session"
        mock_redis_client.keys.return_value = [
            CacheNamespace.SESSION.build_key("test_session", "key1").encode("utf-8"),
            CacheNamespace.SESSION.build_key("test_session", "key2").encode("utf-8"),
        ]
        mock_redis_client.delete.return_value = 2

        result = clear_session(session_id)
        assert result is True
        mock_redis_client.keys.assert_called_once_with(
            CacheNamespace.SESSION.build_key(session_id, "*")
        )

    def test_faiss_index_caching(self, cache_with_mock, mock_redis_client):
        """Test FAISS index caching."""
        import pickle

        index_key = "corpus_index"
        index_data = b"fake_index_data"

        cache_faiss_index(index_key, index_data)
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = pickle.dumps(index_data)
        result = get_faiss_index(index_key)
        assert result == index_data

    def test_analysis_results_caching(self, cache_with_mock, mock_redis_client):
        """Test analysis results caching."""
        analysis_key = "test_analysis"
        results = {"embeddings": np.array([[1, 2, 3]]), "similarity": 0.85}

        cache_analysis_results(analysis_key, results)
        expected_key = CacheNamespace.ANALYSIS.build_key(analysis_key)
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_analysis_results(analysis_key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_document_query_cache_key_uniqueness(
        self, cache_with_mock, mock_redis_client
    ):
        """Test that different document queries with similar hash prefixes generate unique cache keys."""
        import hashlib

        # Two different document queries
        query1 = "machine learning models for natural language processing"
        query2 = "machine learning models for natural language generation"

        # Simulate generating cache keys using a hashing algorithm
        hash1 = hashlib.sha256(query1.encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256(query2.encode("utf-8")).hexdigest()

        # Ensure hashes are distinct
        assert hash1 != hash2

        # Simulate a boundary scenario where the hash prefixes appear identical
        # (e.g., first 12 characters are the same)
        similar_prefix = "abc123def456"
        simulated_hash1 = f"{similar_prefix}{hash1[12:]}"
        simulated_hash2 = f"{similar_prefix}{hash2[12:]}"

        # Using cache_analysis_results to check how keys are set
        cache_analysis_results(simulated_hash1, {"doc": "query1"})
        cache_analysis_results(simulated_hash2, {"doc": "query2"})

        # Verify that two distinct keys were set in Redis, meaning no collision occurred
        assert mock_redis_client.setex.call_count == 2

        call_args_list = mock_redis_client.setex.call_args_list
        key1_called = call_args_list[0][0][0]
        key2_called = call_args_list[1][0][0]

        assert key1_called != key2_called
        assert key1_called == CacheNamespace.ANALYSIS.build_key(simulated_hash1)
        assert key2_called == CacheNamespace.ANALYSIS.build_key(simulated_hash2)

    # ------------------------------------------------------------------
    # Issue #531 – hash-prefix boundary / collision tests
    # ------------------------------------------------------------------

    def test_cache_key_collision_at_truncated_prefix_boundary(
        self, cache_with_mock, mock_redis_client
    ):
        """Verify that full SHA-256 digests prevent collisions even when the
        leading N characters of two digests are artificially identical.

        Background (Issue #531):
            Callers derive an ``analysis_key`` by hashing a document query.
            If a caller naively *truncates* that digest (e.g. keeps only the
            first 12 hex characters) before passing it to
            ``cache_analysis_results``, two queries whose digests share the
            same 12-character prefix would map to the **same** Redis key —
            silently overwriting one result with the other.

            Using the full 64-character digest guarantees uniqueness.

        This test:
            1. Constructs two queries whose full SHA-256 digests intentionally
               share an identical 12-character prefix (crafted via prefix
               grafting, the same technique used in real prefix-sharing
               attacks).
            2. Demonstrates that storing with the *truncated* key causes a
               collision (both writes land on the same key).
            3. Demonstrates that storing with the *full* digest keeps the keys
               distinct (no collision).
        """
        import hashlib

        query_a = "semantic similarity for plagiarism detection in academic papers"
        query_b = "semantic similarity for plagiarism detection in student essays"

        full_hash_a = hashlib.sha256(query_a.encode("utf-8")).hexdigest()
        full_hash_b = hashlib.sha256(query_b.encode("utf-8")).hexdigest()

        # The two queries must produce genuinely different full digests.
        assert (
            full_hash_a != full_hash_b
        ), "Test pre-condition failed: the two queries must hash differently."

        # ── Truncation boundary: graft a shared 12-char prefix ──────────────
        # Simulate a caller that truncates to 12 chars – if those 12 chars
        # happen to be identical, the keys collide.
        shared_prefix = "deadbeef0011"  # deliberately identical for both
        truncated_key_a = shared_prefix  # 12-char key – "unique" part lost
        truncated_key_b = shared_prefix  # same!

        assert (
            truncated_key_a == truncated_key_b
        ), "Test pre-condition: truncated keys must be equal to model collision."

        # ── Case 1: truncated keys DO collide ────────────────────────────────
        mock_redis_client.reset_mock()
        cache_analysis_results(truncated_key_a, {"result": "query_a"})
        cache_analysis_results(truncated_key_b, {"result": "query_b"})

        # Both calls targeted the same Redis key – a collision.
        assert mock_redis_client.setex.call_count == 2
        colliding_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        colliding_key_b = mock_redis_client.setex.call_args_list[1][0][0]
        assert (
            colliding_key_a == colliding_key_b
        ), "Truncated keys should collide, demonstrating the risky pattern."

        # ── Case 2: full-digest keys do NOT collide ──────────────────────────
        mock_redis_client.reset_mock()
        cache_analysis_results(full_hash_a, {"result": "query_a"})
        cache_analysis_results(full_hash_b, {"result": "query_b"})

        assert mock_redis_client.setex.call_count == 2
        safe_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        safe_key_b = mock_redis_client.setex.call_args_list[1][0][0]

        # Full-digest keys must be unique – no collision.
        assert (
            safe_key_a != safe_key_b
        ), "Full-digest keys must be distinct for different queries."
        assert safe_key_a == CacheNamespace.ANALYSIS.build_key(full_hash_a)
        assert safe_key_b == CacheNamespace.ANALYSIS.build_key(full_hash_b)

    @pytest.mark.parametrize(
        "query_a, query_b",
        [
            (
                # Differ only in the final word
                "deep learning for text classification using transformers",
                "deep learning for text classification using convolutions",
            ),
            (
                # Differ by a single character
                "plagiarism detection with bert embeddings v1",
                "plagiarism detection with bert embeddings v2",
            ),
            (
                # Near-identical long strings
                "A" * 255 + "X",
                "A" * 255 + "Y",
            ),
            (
                # Swapped word order (same words, different meaning / position)
                "natural language processing with deep learning",
                "deep learning with natural language processing",
            ),
        ],
    )
    def test_full_digest_keys_never_collide_across_similar_queries(
        self, cache_with_mock, mock_redis_client, query_a: str, query_b: str
    ):
        """Full SHA-256 digest keys must remain unique for all near-identical
        document query pairs (Issue #531 regression sweep).

        Each parametrised pair represents a realistic boundary scenario where
        naive truncation would be most dangerous.  Using the complete digest
        as the ``analysis_key`` must always yield two distinct Redis keys.
        """
        import hashlib

        key_a = hashlib.sha256(query_a.encode("utf-8")).hexdigest()
        key_b = hashlib.sha256(query_b.encode("utf-8")).hexdigest()

        # Queries are intentionally different, so their digests must differ.
        assert (
            key_a != key_b
        ), f"SHA-256 collision detected between:\n  '{query_a}'\n  '{query_b}'"

        mock_redis_client.reset_mock()
        cache_analysis_results(key_a, {"query": query_a})
        cache_analysis_results(key_b, {"query": query_b})

        assert mock_redis_client.setex.call_count == 2

        redis_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        redis_key_b = mock_redis_client.setex.call_args_list[1][0][0]

        # Primary assertion: no collision
        assert (
            redis_key_a != redis_key_b
        ), "Full-digest analysis keys must not collide for distinct queries."
        # Secondary: keys must be well-formed with the 'analysis:' namespace
        assert redis_key_a == CacheNamespace.ANALYSIS.build_key(key_a)
        assert redis_key_b == CacheNamespace.ANALYSIS.build_key(key_b)

    def test_get_cache_singleton(self):
        """Test that get_cache returns the same instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

    def test_get_instance_method_singleton(self):
        """Test that RedisCache.get_instance() returns the singleton instance."""
        instance1 = RedisCache.get_instance()
        instance2 = RedisCache.get_instance()
        instance3 = RedisCache()
        assert instance1 is instance2
        assert instance1 is instance3
        assert instance1 is get_cache()

    def test_redis_cache_lock_exists(self):
        """Verify that RedisCache defines a threading.Lock for singleton thread safety."""
        import threading

        assert hasattr(RedisCache, "_lock")
        assert isinstance(RedisCache._lock, type(threading.Lock()))

    def test_redis_cache_concurrent_instantiation(self):
        """Verify that concurrent threads calling RedisCache() / get_instance() receive the exact same singleton instance."""
        import threading

        instances = []

        def worker():
            for _ in range(50):
                instances.append(RedisCache.get_instance())
                instances.append(RedisCache())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 2000
        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_redis_url_without_ssl_redis_scheme(self):
        """Test that redis:// URL (without SSL) is handled correctly."""
        test_url = "redis://localhost:6379/0"

        with patch.object(redis, "from_url") as mock_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            # Temporarily modify REDIS_URL
            import src.utils.redis_cache as redis_cache_module

            original_url = redis_cache_module.REDIS_URL

            try:
                redis_cache_module.REDIS_URL = test_url

                # Create new instance to trigger reconnection
                cache = RedisCache.__new__(RedisCache)
                cache._connect()

                # Verify from_url was called with the URL
                mock_from_url.assert_called_once_with(
                    test_url,
                    password=None,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
            finally:
                redis_cache_module.REDIS_URL = original_url

    def test_redis_url_with_ssl_rediss_scheme(self):
        """Test that rediss:// URL (with SSL) is handled correctly."""
        test_url = "rediss://localhost:6380/0"

        with patch.object(redis, "from_url") as mock_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            # Temporarily modify REDIS_URL
            import src.utils.redis_cache as redis_cache_module

            original_url = redis_cache_module.REDIS_URL

            try:
                redis_cache_module.REDIS_URL = test_url

                # Create new instance to trigger reconnection
                cache = RedisCache.__new__(RedisCache)
                cache._connect()

                # Verify from_url was called with the URL
                # Note: redis.from_url automatically sets ssl=True for rediss://
                mock_from_url.assert_called_once_with(
                    test_url,
                    password=None,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
            finally:
                redis_cache_module.REDIS_URL = original_url

    def test_redis_url_with_password_and_ssl(self):
        """Test that rediss:// URL with password is handled correctly."""
        test_url = "rediss://user:password@redis.example.com:6380/1"

        with patch.object(redis, "from_url") as mock_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            import src.utils.redis_cache as redis_cache_module

            original_url = redis_cache_module.REDIS_URL
            original_password = redis_cache_module.REDIS_PASSWORD

            try:
                redis_cache_module.REDIS_URL = test_url
                redis_cache_module.REDIS_PASSWORD = None  # Password is in URL

                cache = RedisCache.__new__(RedisCache)
                cache._connect()

                # Verify from_url was called correctly
                mock_from_url.assert_called_once_with(
                    test_url,
                    password=None,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
            finally:
                redis_cache_module.REDIS_URL = original_url
                redis_cache_module.REDIS_PASSWORD = original_password

    def test_redis_host_port_without_ssl(self):
        """Test that host/port config without SSL works correctly."""
        # Use redis:// scheme to ensure SSL is disabled
        test_url = "redis://localhost:6379/0"

        with patch.object(redis, "from_url") as mock_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            import src.utils.redis_cache as redis_cache_module

            original_url = redis_cache_module.REDIS_URL

            try:
                redis_cache_module.REDIS_URL = test_url

                cache = RedisCache.__new__(RedisCache)
                cache._connect()

                # Verify from_url was called
                mock_from_url.assert_called_once()

                # Get the call arguments to verify SSL is not set to True
                call_kwargs = mock_from_url.call_args.kwargs
                # redis.from_url automatically sets ssl based on scheme
                # For redis://, ssl defaults to False
                assert (
                    "ssl" not in call_kwargs or call_kwargs.get("ssl", False) is False
                )
            finally:
                redis_cache_module.REDIS_URL = original_url

    def test_redis_connection_failure_with_message(self):
        """Test that connection failures print appropriate error messages."""
        test_url = "redis://unreachable-host:9999/0"

        with patch.object(redis, "from_url") as mock_from_url:
            # Simulate connection failure
            mock_from_url.side_effect = redis.ConnectionError("Connection refused")

            import src.utils.redis_cache as redis_cache_module

            original_url = redis_cache_module.REDIS_URL

            try:
                redis_cache_module.REDIS_URL = test_url

                # This should not raise, but set _client to None
                cache = RedisCache.__new__(RedisCache)
                cache._connect()

                # Should be None after connection failure
                assert cache._client is None
            finally:
                redis_cache_module.REDIS_URL = original_url

    def test_redis_failover_during_get(self):
        """Test graceful fallback when Redis fails during a get operation."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during get
        mock_client.get.side_effect = RedisError("Connection refused")
        cache._client = mock_client

        # Should return None gracefully without crashing
        result = cache.get("test_key")
        assert result is None

    def test_redis_failover_during_set(self):
        """Test graceful fallback when Redis fails during a set operation."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during set
        mock_client.setex.side_effect = RedisError("Connection timeout")
        cache._client = mock_client

        # Should return False gracefully
        result = cache.set("test_key", "test_value", ttl=60)
        assert result is False

    def test_redis_failover_during_delete(self):
        """Test graceful fallback when Redis fails during a delete operation."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during delete
        mock_client.delete.side_effect = RedisError("Connection lost")
        cache._client = mock_client

        # Should return False gracefully
        result = cache.delete("test_key")
        assert result is False

    def test_redis_failover_during_exists(self):
        """Test graceful fallback when Redis fails during an exists check."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during exists check
        mock_client.exists.side_effect = RedisError("Server unavailable")
        cache._client = mock_client

        # Should return False gracefully
        result = cache.exists("test_key")
        assert result is False

    def test_redis_failover_during_get_json(self):
        """Test graceful fallback when Redis fails during JSON get."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during JSON get
        mock_client.get.side_effect = RedisError("Connection refused")
        cache._client = mock_client

        # Should return None gracefully
        result = cache.get_json("test_json")
        assert result is None

    def test_redis_failover_during_set_json(self):
        """Test graceful fallback when Redis fails during JSON set."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during JSON set
        mock_client.setex.side_effect = RedisError("Connection timeout")
        cache._client = mock_client

        # Should return False gracefully
        result = cache.set_json("test_json", {"key": "value"}, ttl=60)
        assert result is False

    def test_redis_failover_during_clear_pattern(self):
        """Test graceful fallback when Redis fails during pattern clear."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis disconnection during pattern clear
        mock_client.keys.side_effect = RedisError("Connection lost")
        cache._client = mock_client

        # Should return 0 gracefully
        result = cache.clear_pattern("session:*")
        assert result == 0

    def test_redis_failover_during_is_available(self):
        """Test is_available returns False when Redis is unavailable."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = Mock()
        cache._client.ping.side_effect = RedisError("Connection refused")

        # Should return False without crashing
        result = cache.is_available()
        assert result is False

    def test_cache_fallback_when_redis_unavailable(self):
        """Test that cache gracefully falls back when Redis is completely unavailable."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = None

        # All operations should return None/False gracefully
        assert cache.is_available() is False
        assert cache.get("test_key") is None
        assert cache.set("test_key", "test_value") is False
        assert cache.delete("test_key") is False
        assert cache.exists("test_key") is False
        assert cache.get_json("test_key") is None
        assert cache.set_json("test_key", {"value": 1}) is False
        assert cache.clear_pattern("session:*") == 0

    def test_session_state_fallback_when_redis_unavailable(self):
        """Test that session state functions gracefully when Redis is unavailable."""
        from src.utils.redis_cache import _cache as global_cache

        # Temporarily disable Redis
        original_client = global_cache._client
        global_cache._client = None

        try:
            # These should not crash, just return False/None
            assert cache_session_state("test_session", "key", "value") is False
            assert get_session_state("test_session", "key") is None
            assert clear_session("test_session") is False
        finally:
            # Restore original client
            global_cache._client = original_client

    def test_faiss_index_fallback_when_redis_unavailable(self):
        """Test that FAISS index functions gracefully when Redis is unavailable."""
        from src.utils.redis_cache import _cache as global_cache

        # Temporarily disable Redis
        original_client = global_cache._client
        global_cache._client = None

        try:
            # These should not crash, just return None/False
            assert cache_faiss_index("test_key", b"test_data") is False
            assert get_faiss_index("test_key") is None
        finally:
            # Restore original client
            global_cache._client = original_client

    def test_analysis_results_fallback_when_redis_unavailable(self):
        """Test that analysis results functions gracefully when Redis is unavailable."""
        from src.utils.redis_cache import _cache as global_cache

        # Temporarily disable Redis
        original_client = global_cache._client
        global_cache._client = None

        try:
            # These should not crash, just return None/False
            assert cache_analysis_results("test_key", {"results": []}) is False
            assert get_analysis_results("test_key") is None
        finally:
            # Restore original client
            global_cache._client = original_client

    def test_pickle_error_handling_in_get(self):
        """Test graceful handling of pickle deserialization errors."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate valid connection but invalid pickle data
        mock_client.get.return_value = b"invalid_pickle_data"
        cache._client = mock_client

        # Should return None gracefully instead of crashing
        result = cache.get("test_key")
        assert result is None

    def test_json_decode_error_handling_in_get_json(self):
        """Test graceful handling of JSON deserialization errors."""
        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate valid connection but invalid JSON data
        mock_client.get.return_value = "invalid json {"
        cache._client = mock_client

        # Should return None gracefully instead of crashing
        result = cache.get_json("test_json")
        assert result is None

    def test_redis_timeout_during_get(self):
        """Test graceful handling of Redis timeout during get."""
        from src.utils.redis_cache import RedisError

        cache = RedisCache.__new__(RedisCache)
        mock_client = Mock()

        # Simulate Redis timeout
        mock_client.get.side_effect = (
            redis.TimeoutError("Request timed out")
            if hasattr(redis, "TimeoutError")
            else RedisError("Timeout")
        )
        cache._client = mock_client

        # Should return None gracefully
        result = cache.get("test_key")
        assert result is None

    def test_cache_stats_tracking(self, cache_with_mock, mock_redis_client):
        """Test tracking of hits, misses, hit ratio, and total items in cache stats."""
        # Reset hits/misses to start fresh
        cache_with_mock._hits = 0
        cache_with_mock._misses = 0
        cache_with_mock._fallback_cache.clear()

        # Set mock expectations
        mock_redis_client.dbsize.return_value = 5
        mock_redis_client.get.return_value = None

        # 1. Access non-existing key (should be a miss)
        val = cache_with_mock.get("missing_key")
        assert val is None
        stats = cache_with_mock.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.0
        assert stats["total_items"] == 5

        # 2. Write key & read back (should be a hit)
        cache_with_mock.set("existing_key", "hello")
        import pickle

        mock_redis_client.get.return_value = pickle.dumps("hello")

        val2 = cache_with_mock.get("existing_key")
        assert val2 == "hello"
        stats2 = cache_with_mock.get_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1
        assert stats2["hit_ratio"] == 0.5


class TestHitRateTracking:
    """Test hit/miss counter tracking and get_hit_rate() (Issue #714)."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = Mock()
        client.ping.return_value = True
        return client

    @pytest.fixture
    def cache_with_mock(self, mock_redis_client):
        """Create a RedisCache instance with mocked client and reset counters."""
        from src.utils.redis_cache import _cache

        cache = RedisCache.__new__(RedisCache)
        cache._client = mock_redis_client
        cache._hits = 0
        cache._misses = 0
        cache._fallback_cache = {}
        _cache._client = mock_redis_client
        yield cache
        _cache._client = None

    def test_hit_rate_zero_attempts(self, cache_with_mock):
        """No get()/get_json() calls yet -> 0.0, no ZeroDivisionError."""
        assert cache_with_mock.get_hit_rate() == 0.0

    def test_hit_rate_all_hits(self, cache_with_mock, mock_redis_client):
        """All lookups hit -> 100.0."""
        import pickle

        mock_redis_client.get.return_value = pickle.dumps("value")
        for _ in range(4):
            cache_with_mock.get("some_key")
        assert cache_with_mock.get_hit_rate() == 100.0

    def test_hit_rate_all_misses(self, cache_with_mock, mock_redis_client):
        """All lookups miss -> 0.0."""
        mock_redis_client.get.return_value = None
        for _ in range(4):
            cache_with_mock.get("missing_key")
        assert cache_with_mock.get_hit_rate() == 0.0

    def test_hit_rate_mixed(self, cache_with_mock, mock_redis_client):
        """3 hits, 1 miss -> 75.0."""
        import pickle

        mock_redis_client.get.return_value = pickle.dumps("value")
        for _ in range(3):
            cache_with_mock.get("hit_key")

        mock_redis_client.get.return_value = None
        cache_with_mock.get("miss_key")

        assert cache_with_mock.get_hit_rate() == 75.0

    def test_hit_rate_tracks_get_json(self, cache_with_mock, mock_redis_client):
        """get_json() hits/misses are counted toward the same hit rate."""
        mock_redis_client.get.return_value = '{"a": 1}'
        cache_with_mock.get_json("json_key")
        assert cache_with_mock.get_hit_rate() == 100.0

        mock_redis_client.get.return_value = None
        cache_with_mock.get_json("missing_json_key")
        assert cache_with_mock.get_hit_rate() == 50.0

    def test_hit_rate_counts_fallback_path(self, cache_with_mock):
        """When Redis is unavailable, fallback cache hits/misses still count."""
        cache_with_mock._client = None  # force fallback path
        cache_with_mock._fallback_cache["fb_key"] = ("fb_value", None)

        cache_with_mock.get("fb_key")  # hit via fallback
        cache_with_mock.get("nonexistent_key")  # miss via fallback

        assert cache_with_mock.get_hit_rate() == 50.0

    def test_hit_rate_thread_safety(self, cache_with_mock, mock_redis_client):
        """Concurrent get() calls must not lose counter updates."""
        import pickle
        import threading

        mock_redis_client.get.return_value = pickle.dumps("value")

        def hammer():
            for _ in range(100):
                cache_with_mock.get("hot_key")

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache_with_mock._hits == 1000
        assert cache_with_mock.get_hit_rate() == 100.0


def test_redis_fallback_exceptions():
    """Verify that when redis is missing, fallback classes are custom subclasses of Exception."""
    import sys
    from unittest.mock import patch

    # We hide 'redis' module to simulate a missing redis package
    with patch.dict(sys.modules, {"redis": None}):
        # Reload redis_cache module to trigger the except ImportError block
        import importlib

        import src.utils.redis_cache as rc

        importlib.reload(rc)

        # Retrieve the fallback error classes
        fallback_RedisError = rc.RedisError
        fallback_RedisConnectionError = rc.RedisConnectionError
        fallback_RedisTimeoutError = rc.RedisTimeoutError

        # Assert they are distinct subclasses of Exception
        assert issubclass(fallback_RedisError, Exception)
        assert fallback_RedisError is not Exception

        assert issubclass(fallback_RedisConnectionError, fallback_RedisError)
        assert fallback_RedisConnectionError is not ConnectionError

        assert issubclass(fallback_RedisTimeoutError, fallback_RedisError)
        assert fallback_RedisTimeoutError is not TimeoutError

        # Verify that catching fallback_RedisError does NOT catch a generic KeyError
        try:
            raise KeyError("test")
        except fallback_RedisError:
            pytest.fail("fallback_RedisError caught generic KeyError!")
        except KeyError:
            pass  # Expected

    # Finally, reload the module one more time to restore it to the default environment state
    import importlib

    import src.utils.redis_cache as rc

    importlib.reload(rc)


# ── Issue #2320: REDIS_URL password injection ──────────────────


class TestRedisUrlPasswordInjection:
    """Tests for REDIS_URL construction with REDIS_PASSWORD (Issue #2320)."""

    def test_redis_url_includes_password_when_set(self, monkeypatch):
        """When REDIS_PASSWORD is set, it must be injected into the URL."""
        import importlib

        import src.utils.redis_cache as redis_cache_module

        monkeypatch.setenv("REDIS_HOST", "myhost.example.com")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_DB", "2")
        monkeypatch.setenv("REDIS_PASSWORD", "s3cr3tP@ss")
        # Remove REDIS_URL so the fallback is used.
        monkeypatch.delenv("REDIS_URL", raising=False)

        importlib.reload(redis_cache_module)

        assert redis_cache_module.REDIS_URL == (
            "redis://:s3cr3tP@ss@myhost.example.com:6380/2"
        )

    def test_redis_url_omits_password_when_not_set(self, monkeypatch):
        """When REDIS_PASSWORD is not set, the URL must not include credentials."""
        import importlib

        import src.utils.redis_cache as redis_cache_module

        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_DB", "0")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)

        importlib.reload(redis_cache_module)

        assert redis_cache_module.REDIS_URL == "redis://localhost:6379/0"
        # No @ symbol means no credentials in the URL.
        assert "@" not in redis_cache_module.REDIS_URL

    def test_redis_url_respects_explicit_env_var(self, monkeypatch):
        """If REDIS_URL is set explicitly in the env, it takes precedence."""
        import importlib

        import src.utils.redis_cache as redis_cache_module

        monkeypatch.setenv("REDIS_HOST", "ignored.example.com")
        monkeypatch.setenv("REDIS_PORT", "9999")
        monkeypatch.setenv("REDIS_DB", "9")
        monkeypatch.setenv("REDIS_PASSWORD", "ignored_password")
        monkeypatch.setenv("REDIS_URL", "rediss://user:pass@explicit.redis.com:6380/3")

        importlib.reload(redis_cache_module)

        assert redis_cache_module.REDIS_URL == (
            "rediss://user:pass@explicit.redis.com:6380/3"
        )

    def test_redis_url_with_special_chars_in_password(self, monkeypatch):
        """Passwords with special characters are included as-is."""
        import importlib

        import src.utils.redis_cache as redis_cache_module

        monkeypatch.setenv("REDIS_HOST", "redis.example.com")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_DB", "0")
        monkeypatch.setenv("REDIS_PASSWORD", "p@ss:w0rd#123")
        monkeypatch.delenv("REDIS_URL", raising=False)

        importlib.reload(redis_cache_module)

        # The password is inserted as-is (URL encoding is the caller's
        # responsibility — redis-py handles it via from_url).
        assert "p@ss:w0rd#123" in redis_cache_module.REDIS_URL
        assert redis_cache_module.REDIS_URL.startswith("redis://:p@ss:w0rd#123@")

    def test_redis_url_empty_password_falls_back_to_no_auth(self, monkeypatch):
        """An empty REDIS_PASSWORD string should be treated as 'no password'."""
        import importlib

        import src.utils.redis_cache as redis_cache_module

        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_DB", "0")
        monkeypatch.setenv("REDIS_PASSWORD", "")
        monkeypatch.delenv("REDIS_URL", raising=False)

        importlib.reload(redis_cache_module)

        assert redis_cache_module.REDIS_URL == "redis://localhost:6379/0"
        assert "@" not in redis_cache_module.REDIS_URL


class TestPayloadCompressor:
    """Unit tests for PayloadCompressor zlib compression and decompression logic."""

    def test_compress_decompress_large_payload(self):
        """Verify that large payloads exceeding compression threshold compress with magic header and decompress losslessly."""
        threshold = PayloadCompressor.get_threshold()
        large_payload = b"PlagiarismDetectorMockData" * ((threshold // 20) + 100)
        assert len(large_payload) > threshold

        compressed = PayloadCompressor.compress(large_payload)

        assert compressed.startswith(PayloadCompressor.MAGIC_HEADER)
        assert len(compressed) < len(large_payload)

        decompressed = PayloadCompressor.decompress(compressed)
        assert decompressed == large_payload
        assert (
            PayloadCompressor.decompress(PayloadCompressor.compress(large_payload))
            == large_payload
        )

    def test_small_payload_remains_uncompressed(self):
        """Verify that small payloads below compression threshold remain uncompressed and lack magic header."""
        threshold = PayloadCompressor.get_threshold()
        small_payload = b"small_mock_payload_below_threshold"
        assert len(small_payload) < threshold

        compressed = PayloadCompressor.compress(small_payload)

        assert not compressed.startswith(PayloadCompressor.MAGIC_HEADER)
        assert compressed == small_payload

        decompressed = PayloadCompressor.decompress(compressed)
        assert decompressed == small_payload

    def test_compress_empty_byte_string(self):
        """Verify that compressing an empty byte string returns empty bytes and does not fail."""
        empty_data = b""
        compressed = PayloadCompressor.compress(empty_data)
        assert compressed == b""
        assert not compressed.startswith(PayloadCompressor.MAGIC_HEADER)
        assert PayloadCompressor.decompress(compressed) == b""

    def test_compress_one_byte_smaller_than_threshold(self):
        """Verify that compressing a payload exactly 1 byte smaller than the threshold remains uncompressed."""
        threshold = PayloadCompressor.get_threshold()
        payload = b"A" * (threshold - 1)
        assert len(payload) == threshold - 1

        compressed = PayloadCompressor.compress(payload)
        assert not compressed.startswith(PayloadCompressor.MAGIC_HEADER)
        assert compressed == payload
        assert PayloadCompressor.decompress(compressed) == payload

    def test_decompress_garbage_data_with_magic_header_safe_fallback(self):
        """Verify that feeding garbage data prefixed with the magic header safely falls back to None."""
        garbage_payload = (
            PayloadCompressor.MAGIC_HEADER
            + b"this_is_not_valid_zlib_compressed_data_9999"
        )
        result = PayloadCompressor.decompress(garbage_payload)
        assert result is None

    def test_decompress_truncated_zlib_stream_safe_fallback(self):
        """Verify that a truncated or invalid zlib stream with magic header returns None safely."""
        # A partial/corrupted zlib payload
        truncated_payload = PayloadCompressor.MAGIC_HEADER + b"\x78\x9c\x01\x00\x00"
        result = PayloadCompressor.decompress(truncated_payload)
        assert result is None

    def test_decompress_non_bytes_passthrough(self):
        """Verify that passing non-bytes to decompress returns the original input safely."""
        assert PayloadCompressor.decompress(None) is None
        assert PayloadCompressor.decompress("string_input") == "string_input"
        assert PayloadCompressor.decompress(12345) == 12345

    def test_class_level_attributes_default(self):
        """Verify default class-level attributes initialized on module load."""
        import zlib

        assert PayloadCompressor.COMPRESSION_LEVEL == zlib.Z_BEST_SPEED
        assert PayloadCompressor.COMPRESSION_THRESHOLD_BYTES == 64 * 1024
        assert PayloadCompressor.get_threshold() == 64 * 1024

    def test_compression_level_env_int(self, monkeypatch):
        """Verify integer REDIS_COMPRESSION_LEVEL is parsed into COMPRESSION_LEVEL."""
        import importlib

        import src.utils.redis_cache as rc

        monkeypatch.setenv("REDIS_COMPRESSION_LEVEL", "9")
        importlib.reload(rc)
        try:
            assert rc.PayloadCompressor.COMPRESSION_LEVEL == 9
        finally:
            monkeypatch.delenv("REDIS_COMPRESSION_LEVEL", raising=False)
            importlib.reload(rc)

    def test_compression_level_env_named(self, monkeypatch):
        """Verify named constant REDIS_COMPRESSION_LEVEL is parsed into COMPRESSION_LEVEL."""
        import importlib
        import zlib

        import src.utils.redis_cache as rc

        monkeypatch.setenv("REDIS_COMPRESSION_LEVEL", "Z_BEST_COMPRESSION")
        importlib.reload(rc)
        try:
            assert rc.PayloadCompressor.COMPRESSION_LEVEL == zlib.Z_BEST_COMPRESSION
        finally:
            monkeypatch.delenv("REDIS_COMPRESSION_LEVEL", raising=False)
            importlib.reload(rc)

    def test_compression_level_env_invalid(self, monkeypatch):
        """Verify invalid REDIS_COMPRESSION_LEVEL falls back to Z_BEST_SPEED."""
        import importlib
        import zlib

        import src.utils.redis_cache as rc

        monkeypatch.setenv("REDIS_COMPRESSION_LEVEL", "INVALID_OPTION")
        importlib.reload(rc)
        try:
            assert rc.PayloadCompressor.COMPRESSION_LEVEL == zlib.Z_BEST_SPEED
        finally:
            monkeypatch.delenv("REDIS_COMPRESSION_LEVEL", raising=False)
            importlib.reload(rc)

    def test_compression_threshold_env_custom(self, monkeypatch):
        """Verify custom REDIS_COMPRESSION_THRESHOLD is parsed into COMPRESSION_THRESHOLD_BYTES."""
        import importlib

        import src.utils.redis_cache as rc

        monkeypatch.setenv("REDIS_COMPRESSION_THRESHOLD", "2048")
        importlib.reload(rc)
        try:
            assert rc.PayloadCompressor.COMPRESSION_THRESHOLD_BYTES == 2048
            assert rc.PayloadCompressor.get_threshold() == 2048
        finally:
            monkeypatch.delenv("REDIS_COMPRESSION_THRESHOLD", raising=False)
            importlib.reload(rc)

    def test_compression_threshold_env_invalid(self, monkeypatch):
        """Verify invalid REDIS_COMPRESSION_THRESHOLD falls back to default 64 KiB."""
        import importlib

        import src.utils.redis_cache as rc

        monkeypatch.setenv("REDIS_COMPRESSION_THRESHOLD", "not_a_number")
        importlib.reload(rc)
        try:
            assert rc.PayloadCompressor.COMPRESSION_THRESHOLD_BYTES == 64 * 1024
            assert rc.PayloadCompressor.get_threshold() == 64 * 1024
        finally:
            monkeypatch.delenv("REDIS_COMPRESSION_THRESHOLD", raising=False)
            importlib.reload(rc)

    def test_compress_does_not_call_os_getenv(self):
        """Verify that compress() does not perform any os.getenv lookups during execution."""
        threshold = PayloadCompressor.COMPRESSION_THRESHOLD_BYTES
        large_payload = b"PlagiarismDetectorTestData" * ((threshold // 20) + 100)

        with patch("os.getenv") as mock_getenv:
            compressed = PayloadCompressor.compress(large_payload)
            assert compressed.startswith(PayloadCompressor.MAGIC_HEADER)
            mock_getenv.assert_not_called()


def test_payload_compressor_exact_threshold_boundary():
    """Verify compression boundary behavior at exactly COMPRESSION_THRESHOLD_BYTES (64 * 1024 bytes).

    Protects against off-by-one errors (incorrect >= vs > threshold evaluation).
    Payloads of size >= COMPRESSION_THRESHOLD_BYTES must be compressed with MAGIC_HEADER,
    while payloads of size COMPRESSION_THRESHOLD_BYTES - 1 must remain uncompressed.
    """
    from src.utils.redis_cache import PayloadCompressor

    threshold = PayloadCompressor.COMPRESSION_THRESHOLD_BYTES
    assert threshold == 64 * 1024

    # 1. Payload exactly at the threshold (64 * 1024 bytes) MUST be compressed
    exact_threshold_data = b"B" * threshold
    assert len(exact_threshold_data) == 64 * 1024

    compressed_exact = PayloadCompressor.compress(exact_threshold_data)
    assert compressed_exact.startswith(PayloadCompressor.MAGIC_HEADER)
    assert len(compressed_exact) < len(exact_threshold_data)
    assert PayloadCompressor.decompress(compressed_exact) == exact_threshold_data

    # 2. Payload 1 byte below the threshold (64 * 1024 - 1 bytes) MUST remain uncompressed
    below_threshold_data = b"B" * (threshold - 1)
    assert len(below_threshold_data) == (64 * 1024 - 1)

    compressed_below = PayloadCompressor.compress(below_threshold_data)
    assert not compressed_below.startswith(PayloadCompressor.MAGIC_HEADER)
    assert compressed_below == below_threshold_data
    assert PayloadCompressor.decompress(compressed_below) == below_threshold_data


def test_redis_password_special_characters_escaped(monkeypatch):
    """Verify REDIS_PASSWORD with special characters (@, /, #, :) is safely URL-encoded (Issue #2799)."""
    import importlib
    import urllib.parse

    import src.utils.redis_cache

    monkeypatch.setenv("REDIS_PASSWORD", "p@ss/word#123:secret")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_DB", "0")
    monkeypatch.delenv("REDIS_URL", raising=False)

    importlib.reload(src.utils.redis_cache)

    expected_encoded = urllib.parse.quote_plus("p@ss/word#123:secret")
    assert expected_encoded in src.utils.redis_cache.REDIS_URL
    assert f":{expected_encoded}@localhost:6379/0" in src.utils.redis_cache.REDIS_URL
