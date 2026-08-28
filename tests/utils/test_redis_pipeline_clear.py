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
tests/utils/test_redis_pipeline_clear.py
----------------------------------------
Unit tests verifying that clear_session and clear_pattern leverage Redis pipelines
for batched deletions in a single network round-trip (Issue #2816).
"""

from unittest.mock import MagicMock, patch

from src.utils.redis_cache import RedisCache, clear_all_large_data, clear_session


def test_clear_pattern_uses_redis_pipeline():
    cache = RedisCache.__new__(RedisCache)
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    # 5 matching keys
    test_keys = [f"spd:v1:session:sess123:key_{i}".encode() for i in range(5)]
    mock_client.scan_iter.return_value = iter(test_keys)
    mock_client.keys.return_value = test_keys

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = [5]
    mock_client.pipeline.return_value = mock_pipeline

    cache._client = mock_client
    cache._fallback_cache = {}

    deleted_count = cache.clear_pattern("spd:v1:session:sess123:*")

    mock_client.pipeline.assert_called_once()
    mock_pipeline.delete.assert_called_once_with(*test_keys)
    mock_pipeline.execute.assert_called_once()
    assert deleted_count == 5


def test_clear_session_delegates_to_pipelined_clear_pattern():
    with patch("src.utils.redis_cache._cache") as mock_cache:
        mock_cache.clear_pattern.return_value = 10
        result = clear_session("session_abc")
        mock_cache.clear_pattern.assert_called_once_with("spd:v1:session:session_abc:*")
        assert result is True


def test_clear_pattern_batches_large_key_sets():
    cache = RedisCache.__new__(RedisCache)
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    # 2500 matching keys (exceeds 1000 chunk size)
    test_keys = [f"key_{i}".encode() for i in range(2500)]
    mock_client.scan_iter.return_value = iter(test_keys)
    mock_client.keys.return_value = test_keys

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = [1000, 1000, 500]
    mock_client.pipeline.return_value = mock_pipeline

    cache._client = mock_client
    cache._fallback_cache = {}

    deleted_count = cache.clear_pattern("key_*")

    mock_client.pipeline.assert_called_once()
    assert mock_pipeline.delete.call_count == 3
    mock_pipeline.execute.assert_called_once()
    assert deleted_count == 2500


def test_clear_all_large_data_uses_pipeline():
    with patch("src.utils.redis_cache.get_cache") as mock_get_cache:
        cache_mock = MagicMock()
        cache_mock.is_available.return_value = True
        mock_client = MagicMock()
        test_keys = [b"spd:v1:large:sess1:chunk1", b"spd:v1:large:sess1:chunk2"]
        mock_client.scan_iter.return_value = iter(test_keys)
        mock_client.keys.return_value = test_keys
        mock_pipeline = MagicMock()
        mock_client.pipeline.return_value = mock_pipeline
        cache_mock._client = mock_client
        mock_get_cache.return_value = cache_mock

        clear_all_large_data("sess1")

        mock_client.pipeline.assert_called_once()
        mock_pipeline.delete.assert_called_once_with(*test_keys)
        mock_pipeline.execute.assert_called_once()
