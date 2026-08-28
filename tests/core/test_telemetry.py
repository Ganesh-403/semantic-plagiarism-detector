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

from unittest.mock import patch

from src.core.telemetry import TelemetryService

# ---------------------------------------------------------
# Test Active User Count
# ---------------------------------------------------------


def test_telemetry_cache_hit():
    """
    Test that TelemetryService.get_active_user_count correctly returns a cached value
    without querying the DB.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_user_count"
    ) as mock_get_user_count:
        mock_get_cache.return_value = "42"

        count = TelemetryService.get_active_user_count()

        assert count == 42
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_USER_COUNT)
        mock_get_user_count.assert_not_called()


def test_telemetry_cache_miss():
    """
    Test that on cache miss, the service queries the DB and populates the cache.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_user_count"
    ) as mock_get_user_count, patch("src.core.telemetry.set_cache") as mock_set_cache:
        mock_get_cache.return_value = None
        mock_get_user_count.return_value = 17

        count = TelemetryService.get_active_user_count()

        assert count == 17
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_USER_COUNT)
        mock_get_user_count.assert_called_once()
        mock_set_cache.assert_called_once_with(
            TelemetryService.CACHE_KEY_USER_COUNT,
            "17",
            expire=TelemetryService.CACHE_TTL_SECONDS,
        )


def test_telemetry_db_failure():
    """
    Test that if the database lookup fails entirely, the service handles it gracefully.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_user_count"
    ) as mock_get_user_count:
        mock_get_cache.return_value = None
        mock_get_user_count.side_effect = Exception("DB Connection Lost")

        count = TelemetryService.get_active_user_count()

        assert count == 0


# ---------------------------------------------------------
# Test Document Count
# ---------------------------------------------------------


def test_telemetry_doc_count_cache_hit():
    """
    Test that TelemetryService.get_document_count hits the cache.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_document_count_fast"
    ) as mock_get_doc_count_fast:
        mock_get_cache.return_value = "99"

        count = TelemetryService.get_document_count()

        assert count == 99
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_DOC_COUNT)
        mock_get_doc_count_fast.assert_not_called()


def test_telemetry_doc_count_cache_miss():
    """
    Test that on cache miss, get_document_count queries DB and populates cache.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_document_count_fast"
    ) as mock_get_doc_count_fast, patch(
        "src.core.telemetry.set_cache"
    ) as mock_set_cache:
        mock_get_cache.return_value = None
        mock_get_doc_count_fast.return_value = 3

        count = TelemetryService.get_document_count()

        assert count == 3
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_DOC_COUNT)
        mock_get_doc_count_fast.assert_called_once()
        mock_set_cache.assert_called_once_with(
            TelemetryService.CACHE_KEY_DOC_COUNT,
            "3",
            expire=TelemetryService.CACHE_TTL_SECONDS,
        )


def test_telemetry_doc_db_failure():
    """
    Test that if document DB query fails, service falls back safely.
    """
    with patch("src.core.telemetry.get_cache") as mock_get_cache, patch(
        "src.core.telemetry.get_document_count_fast"
    ) as mock_get_doc_count_fast:
        mock_get_cache.return_value = None
        mock_get_doc_count_fast.side_effect = Exception("DB Fault")

        count = TelemetryService.get_document_count()

        assert count == 0


# ---------------------------------------------------------
# Test Force Refresh
# ---------------------------------------------------------


def test_telemetry_force_refresh():
    """
    Test that force_refresh bypasses the get_cache check and immediately updates cache.
    """
    with patch("src.core.telemetry.get_user_count") as mock_get_user_count, patch(
        "src.core.telemetry.get_document_count_fast"
    ) as mock_get_doc_count_fast, patch(
        "src.core.telemetry.set_cache"
    ) as mock_set_cache:
        mock_get_user_count.return_value = 100
        mock_get_doc_count_fast.return_value = 550

        TelemetryService.force_refresh_metrics()

        mock_get_user_count.assert_called_once()
        mock_get_doc_count_fast.assert_called_once()
        assert mock_set_cache.call_count == 2


# ---------------------------------------------------------
# Test Clear Telemetry Data
# ---------------------------------------------------------


def test_telemetry_clear_telemetry_data():
    """
    Test that clear_telemetry_data calls delete_cache for both telemetry keys without AttributeError.
    """
    with patch("src.utils.redis_cache.delete_cache") as mock_delete_cache:
        TelemetryService.clear_telemetry_data()

        assert mock_delete_cache.call_count == 2
        mock_delete_cache.assert_any_call(TelemetryService.CACHE_KEY_USER_COUNT)
        mock_delete_cache.assert_any_call(TelemetryService.CACHE_KEY_DOC_COUNT)


def test_telemetry_clear_telemetry_data_exception_handling(caplog):
    """
    Test that clear_telemetry_data handles exceptions from delete_cache gracefully.
    """
    with patch(
        "src.utils.redis_cache.delete_cache", side_effect=Exception("Redis offline")
    ):
        TelemetryService.clear_telemetry_data()

    assert "Failed to clear telemetry cache key" in caplog.text
