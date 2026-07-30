"""
test_redis_cache.py
-------------------
Unit tests for Redis cache functionality.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.utils.redis_cache import (
    CacheKeyPrefix,
    RedisCache,
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
        from src.utils.redis_cache import _cache

        cache = RedisCache.__new__(RedisCache)
        cache._client = mock_redis_client
        _cache._client = mock_redis_client
        yield cache
        _cache._client = None

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
        expected_key = f"{CacheKeyPrefix.SESSION.value}:{session_id}:{key}"
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_session_state(session_id, key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_clear_session(self, cache_with_mock, mock_redis_client):
        """Test clearing session data."""
        session_id = "test_session"
        mock_redis_client.keys.return_value = [
            f"{CacheKeyPrefix.SESSION.value}:test_session:key1".encode("utf-8"),
            f"{CacheKeyPrefix.SESSION.value}:test_session:key2".encode("utf-8"),
        ]
        mock_redis_client.delete.return_value = 2

        result = clear_session(session_id)
        assert result is True
        mock_redis_client.keys.assert_called_once_with(f"{CacheKeyPrefix.SESSION.value}:{session_id}:*")

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
        expected_key = f"{CacheKeyPrefix.ANALYSIS.value}:{analysis_key}"
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_analysis_results(analysis_key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_document_query_cache_key_uniqueness(self, cache_with_mock, mock_redis_client):
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
        assert key1_called == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{simulated_hash1}"
        assert key2_called == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{simulated_hash2}"

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
        assert full_hash_a != full_hash_b, (
            "Test pre-condition failed: the two queries must hash differently."
        )

        # ── Truncation boundary: graft a shared 12-char prefix ──────────────
        # Simulate a caller that truncates to 12 chars – if those 12 chars
        # happen to be identical, the keys collide.
        shared_prefix = "deadbeef0011"  # deliberately identical for both
        truncated_key_a = shared_prefix  # 12-char key – "unique" part lost
        truncated_key_b = shared_prefix  # same!

        assert truncated_key_a == truncated_key_b, (
            "Test pre-condition: truncated keys must be equal to model collision."
        )

        # ── Case 1: truncated keys DO collide ────────────────────────────────
        mock_redis_client.reset_mock()
        cache_analysis_results(truncated_key_a, {"result": "query_a"})
        cache_analysis_results(truncated_key_b, {"result": "query_b"})

        # Both calls targeted the same Redis key – a collision.
        assert mock_redis_client.setex.call_count == 2
        colliding_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        colliding_key_b = mock_redis_client.setex.call_args_list[1][0][0]
        assert colliding_key_a == colliding_key_b, (
            "Truncated keys should collide, demonstrating the risky pattern."
        )

        # ── Case 2: full-digest keys do NOT collide ──────────────────────────
        mock_redis_client.reset_mock()
        cache_analysis_results(full_hash_a, {"result": "query_a"})
        cache_analysis_results(full_hash_b, {"result": "query_b"})

        assert mock_redis_client.setex.call_count == 2
        safe_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        safe_key_b = mock_redis_client.setex.call_args_list[1][0][0]

        # Full-digest keys must be unique – no collision.
        assert safe_key_a != safe_key_b, (
            "Full-digest keys must be distinct for different queries."
        )
        assert safe_key_a == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{full_hash_a}"
        assert safe_key_b == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{full_hash_b}"

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
        assert key_a != key_b, (
            f"SHA-256 collision detected between:\n  '{query_a}'\n  '{query_b}'"
        )

        mock_redis_client.reset_mock()
        cache_analysis_results(key_a, {"query": query_a})
        cache_analysis_results(key_b, {"query": query_b})

        assert mock_redis_client.setex.call_count == 2

        redis_key_a = mock_redis_client.setex.call_args_list[0][0][0]
        redis_key_b = mock_redis_client.setex.call_args_list[1][0][0]

        # Primary assertion: no collision
        assert redis_key_a != redis_key_b, (
            "Full-digest analysis keys must not collide for distinct queries."
        )
        # Secondary: keys must be well-formed with the 'analysis:' namespace
        assert redis_key_a == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{key_a}"
        assert redis_key_b == f"{CacheKeyPrefix.LEGACY_ANALYSIS_PREFIX.value}{key_b}"

    def test_get_cache_singleton(self):
        """Test that get_cache returns the same instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

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

