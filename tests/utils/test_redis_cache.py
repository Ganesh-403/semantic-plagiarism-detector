"""
test_redis_cache.py
-------------------
Unit tests for Redis cache functionality.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.utils.redis_cache import (RedisCache, cache_analysis_results,
                                   cache_faiss_index, cache_session_state,
                                   clear_session, get_analysis_results,
                                   get_cache, get_faiss_index,
                                   get_session_state)


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
        expected_key = f"session:{session_id}:{key}"
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_session_state(session_id, key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_clear_session(self, cache_with_mock, mock_redis_client):
        """Test clearing session data."""
        session_id = "test_session"
        mock_redis_client.keys.return_value = [
            b"session:test_session:key1",
            b"session:test_session:key2",
        ]
        mock_redis_client.delete.return_value = 2

        result = clear_session(session_id)
        assert result is True
        mock_redis_client.keys.assert_called_once_with(f"session:{session_id}:*")

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
        expected_key = f"analysis:{analysis_key}"
        mock_redis_client.setex.assert_called_once()

        mock_redis_client.get.return_value = b"\x80"
        get_analysis_results(analysis_key)
        mock_redis_client.get.assert_called_once_with(expected_key)

    def test_get_cache_singleton(self):
        """Test that get_cache returns the same instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2
