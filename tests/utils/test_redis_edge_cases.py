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
test_redis_edge_cases.py
------------------------
Edge-case tests verifying graceful fallback when Redis becomes unavailable.
"""

from unittest.mock import Mock, patch

import pytest
import redis

from src.utils.redis_cache import RedisCache


@pytest.fixture
def mock_redis_refused():
    """Mock Redis client to raise ConnectionRefusedError on all operations."""
    client = Mock()
    # Simulate ConnectionRefusedError on typical operations
    client.ping.side_effect = ConnectionRefusedError("Connection refused")
    client.get.side_effect = ConnectionRefusedError("Connection refused")
    client.set.side_effect = ConnectionRefusedError("Connection refused")
    client.setex.side_effect = ConnectionRefusedError("Connection refused")
    client.delete.side_effect = ConnectionRefusedError("Connection refused")
    client.exists.side_effect = ConnectionRefusedError("Connection refused")
    client.keys.side_effect = ConnectionRefusedError("Connection refused")
    return client


def test_redis_unavailable_during_initialization():
    """Test scenario: Redis unavailable during initialization."""
    with patch("src.utils.redis_cache.redis") as mock_redis_module:
        # When Redis attempts to connect, raise ConnectionRefusedError
        mock_redis_module.from_url.side_effect = ConnectionRefusedError(
            "Connection refused"
        )
        mock_redis_module.Redis.side_effect = ConnectionRefusedError(
            "Connection refused"
        )
        # Ensure we still have access to the exceptions for catching
        mock_redis_module.ConnectionError = redis.ConnectionError
        mock_redis_module.TimeoutError = redis.TimeoutError

        cache = RedisCache.__new__(RedisCache)
        cache._client = None

        # This shouldn't raise an exception
        cache._connect()

        # Verify it falls back
        assert cache._client is None
        assert cache.is_available() is False


def test_redis_disconnects_during_cache_access(mock_redis_refused):
    """Test scenario: Redis disconnects during cache access (ping fails)."""
    cache = RedisCache.__new__(RedisCache)
    cache._client = mock_redis_refused

    # is_available uses ping()
    assert cache.is_available() is False

    # ping() directly
    status, latency = cache.ping()
    assert status is False
    assert latency is None


def test_cache_write_and_read_fallback_on_failure(mock_redis_refused):
    """Test scenario: Cache fallback to in-memory on write/read failure when Redis is refused."""
    cache = RedisCache.__new__(RedisCache)
    cache._client = mock_redis_refused

    # Override is_available to simulate that the connection WAS available
    # but the write/read operations fail on Redis due to connection refusal
    with patch.object(cache, "is_available", return_value=True):
        # 1. Write operations fall back and return True/success
        set_result = cache.set("some_key", "value")
        assert set_result is True

        set_json_result = cache.set_json("some_json_key", {"a": 1})
        assert set_json_result is True

        # 2. Read operations successfully retrieve from fallback cache
        assert cache.get("some_key") == "value"
        assert cache.get_json("some_json_key") == {"a": 1}

        # 3. Exists operation returns True from fallback cache
        assert cache.exists("some_key") is True

        # 4. Pattern clearing works on fallback cache
        clear_result = cache.clear_pattern("some_json_*")
        assert clear_result == 1
        assert cache.get_json("some_json_key") is None

        # 5. Delete operation works on fallback cache
        delete_result = cache.delete("some_key")
        assert delete_result is True
        assert cache.exists("some_key") is False


def test_redis_configurable_timeout():
    """Verify that REDIS_TIMEOUT_SECONDS is read and passed to Redis constructor."""
    import importlib
    import os
    from unittest.mock import patch

    import src.utils.redis_cache

    with patch.dict(os.environ, {"REDIS_TIMEOUT_SECONDS": "4.5"}), patch(
        "redis.from_url"
    ) as mock_from_url:
        # Reload the module to pick up the new environment variable value
        importlib.reload(src.utils.redis_cache)

        assert src.utils.redis_cache.REDIS_TIMEOUT_SECONDS == 4.5
        mock_from_url.reset_mock()

        # Instantiate cache and verify mock connection parameters
        cache = src.utils.redis_cache.RedisCache.__new__(
            src.utils.redis_cache.RedisCache
        )
        cache._client = None
        cache._connect()

        mock_from_url.assert_called_once()
        kwargs = mock_from_url.call_args[1]
        assert kwargs.get("socket_connect_timeout") == 4.5

    # Reload the module once more with environment cleared to restore defaults
    importlib.reload(src.utils.redis_cache)
