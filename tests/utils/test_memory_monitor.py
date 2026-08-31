"""
tests/utils/test_memory_monitor.py
-----------------------------------
Unit tests for src/utils/memory_monitor.py, focused on
check_memory_threshold() (Issue #3473).
"""

from unittest.mock import patch

import pytest

from src.utils.memory_monitor import check_memory_threshold, get_memory_usage


def _fake_usage(percent: float) -> dict:
    """Build a fake get_memory_usage() result with a fixed percent value."""
    return {
        "rss_mb": 123.4,
        "vms_mb": 456.7,
        "percent": percent,
        "cpu_percent": 1.0,
    }


class TestCheckMemoryThreshold:
    """Test suite for check_memory_threshold() (Issue #3473)."""

    def test_returns_false_when_below_threshold(self):
        """Usage below the threshold returns False."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(50.0),
        ):
            result = check_memory_threshold(threshold_percent=85.0)
        assert result is False

    def test_returns_true_when_above_threshold(self):
        """Usage above the threshold returns True."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(90.0),
        ):
            result = check_memory_threshold(threshold_percent=85.0)
        assert result is True

    def test_exactly_at_threshold_is_not_exceeded(self):
        """Usage exactly equal to the threshold is NOT considered exceeded
        (strictly greater-than semantics)."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(85.0),
        ):
            result = check_memory_threshold(threshold_percent=85.0)
        assert result is False

    def test_default_threshold_is_85_percent(self):
        """Calling with no threshold_percent argument uses 85.0 as the default."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(84.9),
        ):
            assert check_memory_threshold() is False
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(85.1),
        ):
            assert check_memory_threshold() is True

    def test_on_exceeded_not_called_when_below_threshold(self):
        """The callback must NOT fire when usage is under the threshold."""
        called = []
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(50.0),
        ):
            check_memory_threshold(
                threshold_percent=85.0, on_exceeded=lambda: called.append(1)
            )
        assert called == []

    def test_on_exceeded_called_when_above_threshold(self):
        """The callback fires exactly once when usage exceeds the threshold."""
        called = []
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(90.0),
        ):
            check_memory_threshold(
                threshold_percent=85.0, on_exceeded=lambda: called.append(1)
            )
        assert called == [1]

    def test_no_callback_provided_does_not_raise(self):
        """Omitting on_exceeded is safe even when the threshold is exceeded."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(99.0),
        ):
            result = check_memory_threshold(threshold_percent=85.0)
        assert result is True

    def test_failing_callback_is_caught_and_logged(self):
        """An exception raised inside on_exceeded must not propagate to the
        caller — it should be caught, logged, and the function should still
        return the correct boolean."""

        def bad_callback():
            raise RuntimeError("boom")

        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(95.0),
        ):
            result = check_memory_threshold(on_exceeded=bad_callback)
        assert result is True

    def test_custom_threshold_percent(self):
        """A custom, non-default threshold_percent is respected."""
        with patch(
            "src.utils.memory_monitor.get_memory_usage",
            return_value=_fake_usage(60.0),
        ):
            assert check_memory_threshold(threshold_percent=50.0) is True
            assert check_memory_threshold(threshold_percent=70.0) is False

    def test_integrates_with_real_get_memory_usage(self):
        """Smoke test against the real (unmocked) get_memory_usage() to
        confirm the function works end-to-end without mocking, using an
        unreachable threshold so it deterministically returns False."""
        result = check_memory_threshold(threshold_percent=100.0)
        assert result is False
        # Sanity: get_memory_usage() itself still returns the expected shape.
        usage = get_memory_usage()
        assert "percent" in usage
