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
tests/utils/test_redis_cache_compression.py
-------------------------------------------
Comprehensive unit tests for the PayloadCompressor and RedisCache
graceful error handling during zlib decompression failures (Issue #2801).

Verifies that corrupted payloads do not crash the application, but are
instead treated as cache misses, triggering safe recomputation.
"""

import zlib
from unittest.mock import MagicMock

import pytest

from src.utils.redis_cache import PayloadCompressor, RedisCache


class TestPayloadCompressorDecompress:
    """Test suite for PayloadCompressor.decompress error handling."""

    def test_decompress_valid_payload(self):
        """Verify valid compressed payloads are decompressed correctly."""
        original_data = b"This is a test payload for compression."
        compressed = PayloadCompressor.compress(original_data)

        # Force compression by bypassing the threshold check
        raw_compressed = zlib.compress(original_data)
        forced_compressed = PayloadCompressor.MAGIC_HEADER + raw_compressed

        result = PayloadCompressor.decompress(forced_compressed)
        assert result == original_data

    def test_decompress_uncompressed_payload(self):
        """Verify payloads without the magic header are returned as-is."""
        raw_data = b"Uncompressed raw data"
        result = PayloadCompressor.decompress(raw_data)
        assert result == raw_data

    def test_decompress_corrupted_zlib_returns_none(self):
        """Verify corrupted zlib payloads return None instead of raising zlib.error."""
        # Create a payload with the magic header but garbage zlib data
        corrupted_payload = (
            PayloadCompressor.MAGIC_HEADER + b"this is not valid zlib data"
        )

        # Should NOT raise zlib.error
        result = PayloadCompressor.decompress(corrupted_payload)

        # Must return None to signal a cache miss
        assert result is None

    def test_decompress_truncated_zlib_returns_none(self):
        """Verify truncated zlib streams (common in network drops) return None."""
        original_data = b"A" * 10000
        raw_compressed = zlib.compress(original_data)

        # Truncate the compressed stream halfway through
        truncated = (
            PayloadCompressor.MAGIC_HEADER + raw_compressed[: len(raw_compressed) // 2]
        )

        result = PayloadCompressor.decompress(truncated)
        assert result is None

    def test_decompress_non_bytes_returns_input(self):
        """Verify non-bytes input is returned unchanged."""
        assert PayloadCompressor.decompress("string") == "string"
        assert PayloadCompressor.decompress(123) == 123
        assert PayloadCompressor.decompress(None) is None

    def test_decompress_logs_critical_on_corruption(self, caplog):
        """Verify a CRITICAL log is emitted when decompression fails."""
        import logging

        corrupted_payload = PayloadCompressor.MAGIC_HEADER + b"garbage"

        with caplog.at_level(logging.CRITICAL):
            PayloadCompressor.decompress(corrupted_payload)

        assert any(
            "CRITICAL" in record.message
            or "zlib decompression failed" in record.message
            for record in caplog.records
        )


class TestRedisCacheGetCorruptionHandling:
    """Test suite for RedisCache.get handling of corrupted cache entries."""

    @pytest.fixture
    def mock_redis_client(self):
        """Provide a mocked Redis client."""
        client = MagicMock()
        client.ping.return_value = True
        return client

    @pytest.fixture
    def cache_instance(self, mock_redis_client):
        """Provide a RedisCache instance with a mocked client."""
        cache = RedisCache()
        cache._client = mock_redis_client
        cache._fallback_cache = {}
        return cache

    def test_get_handles_corrupted_pickle_payload(
        self, cache_instance, mock_redis_client
    ):
        """Verify get() treats corrupted pickle payloads as cache misses."""
        # Mock Redis returning a corrupted compressed payload
        corrupted_data = PayloadCompressor.MAGIC_HEADER + b"corrupted zlib"
        mock_redis_client.get.return_value = corrupted_data

        # Should not raise an exception
        result = cache_instance.get("test_key")

        # Should return None (cache miss)
        assert result is None

        # Should attempt to delete the corrupted key from Redis
        mock_redis_client.delete.assert_called_once_with("test_key")

    def test_get_json_handles_corrupted_json_payload(
        self, cache_instance, mock_redis_client
    ):
        """Verify get_json() treats corrupted JSON payloads as cache misses."""
        corrupted_data = PayloadCompressor.MAGIC_HEADER + b"corrupted zlib"
        mock_redis_client.get.return_value = corrupted_data

        result = cache_instance.get_json("test_json_key")

        assert result is None
        mock_redis_client.delete.assert_called_once_with("test_json_key")

    def test_get_falls_back_to_memory_on_redis_corruption(
        self, cache_instance, mock_redis_client
    ):
        """Verify get() checks the in-memory fallback if Redis payload is corrupted."""
        # Redis returns corrupted data
        mock_redis_client.get.return_value = PayloadCompressor.MAGIC_HEADER + b"bad"

        # Fallback cache has the valid data
        cache_instance._fallback_cache["test_key"] = ({"valid": "data"}, None)

        result = cache_instance.get("test_key")

        # Should retrieve from fallback cache since Redis failed
        assert result == {"valid": "data"}

    def test_get_increments_misses_on_corruption(
        self, cache_instance, mock_redis_client
    ):
        """Verify cache miss counter increments when payload is corrupted."""
        mock_redis_client.get.return_value = PayloadCompressor.MAGIC_HEADER + b"bad"
        initial_misses = cache_instance._misses

        cache_instance.get("test_key")

        assert cache_instance._misses == initial_misses + 1
