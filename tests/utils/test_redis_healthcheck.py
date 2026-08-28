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
tests/utils/test_redis_healthcheck.py
-------------------------------------
Unit tests for RedisCache.ping() health check method.
"""

from unittest.mock import MagicMock

from src.utils.redis_cache import RedisCache


def test_redis_cache_ping_when_client_is_none():
    cache = RedisCache.__new__(RedisCache)
    cache._client = None
    connected, latency = cache.ping()
    assert connected is False
    assert latency is None


def test_redis_cache_ping_success():
    cache = RedisCache.__new__(RedisCache)
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    cache._client = mock_client

    connected, latency = cache.ping()
    assert connected is True
    assert latency is not None
    assert latency >= 0.0
    mock_client.ping.assert_called_once()


def test_redis_cache_ping_failure_on_exception():
    cache = RedisCache.__new__(RedisCache)
    mock_client = MagicMock()
    mock_client.ping.side_effect = ConnectionError("Connection refused")
    cache._client = mock_client

    connected, latency = cache.ping()
    assert connected is False
    assert latency is None
    mock_client.ping.assert_called_once()
