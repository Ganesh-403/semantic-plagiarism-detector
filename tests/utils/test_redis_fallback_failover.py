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
tests/utils/test_redis_fallback_failover.py
-------------------------------------------
Dedicated test suite verifying that RedisCache seamlessly fails over to fallback_cache
upon mid-session connection drop without dropping user uploaded files (Issue #2817).
"""

import pickle
from unittest.mock import MagicMock

from src.utils.redis_cache import RedisCache, RedisConnectionError


def test_redis_connection_drop_on_second_get_fails_over_to_memory():
    """
    Test simulating redis ConnectionError on the second .get() call.
    Verifies that system recovers and stores subsequent .set() calls in memory.
    """
    cache = RedisCache.__new__(RedisCache)
    cache._fallback_cache = {}
    cache._hits = 0
    cache._misses = 0

    mock_client = MagicMock()
    mock_client.ping.return_value = True

    file_1 = {"name": "assignment1.pdf", "bytes": b"binary_data_1"}
    file_2 = {"name": "assignment2.docx", "bytes": b"binary_data_2"}

    # First .get() returns file_1 from Redis
    # Second .get() raises RedisConnectionError simulating mid-session network drop
    mock_client.get.side_effect = [
        pickle.dumps(file_1),
        RedisConnectionError("Connection lost mid-session"),
    ]
    cache._client = mock_client

    # 1. First get succeeds from Redis
    val1 = cache.get("file:1")
    assert val1 == file_1

    # 2. Second get encounters connection error on Redis, gracefully falls back without crashing
    val2 = cache.get("file:2")
    assert val2 is None

    # 3. Subsequent set() should seamlessly write to fallback memory cache
    mock_client.set.side_effect = RedisConnectionError("Connection lost")
    mock_client.setex.side_effect = RedisConnectionError("Connection lost")
    set_ok = cache.set("file:2", file_2, ttl=600)
    assert set_ok is True

    # 4. Subsequent get() successfully retrieves file_2 from fallback cache
    mock_client.get.side_effect = RedisConnectionError("Connection lost")
    retrieved_file2 = cache.get("file:2")
    assert retrieved_file2 == file_2
    assert retrieved_file2["name"] == "assignment2.docx"
    assert retrieved_file2["bytes"] == b"binary_data_2"
