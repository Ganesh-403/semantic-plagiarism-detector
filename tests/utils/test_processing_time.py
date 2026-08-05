"""
Unit tests for src.utils.processing_time helpers and ProcessingTimer.
"""

import time
import pytest

from src.utils.processing_time import (
    BYTES_PER_MB,
    ProcessingTimer,
    calculate_mb_per_minute,
    calculate_page_throughput,
    calculate_processing_throughput,
    estimate_processing_seconds,    format_duration,
    format_processing_duration,
    format_throughput_human_readable,
    processing_eta_text,
    uploaded_files_total_bytes,
)


# ============================================================================
# Page Throughput Tests
# ============================================================================


@pytest.mark.parametrize(
    ("total_pages", "elapsed_seconds", "expected"),
    [
        (100, 10.0, 10.0),
        (50, 4.0, 12.5),
        (0, 5.0, 0.0),
        (100, 0.0, 0.0),
        (100, -1.0, 0.0),
    ],
)
def test_calculate_page_throughput(total_pages, elapsed_seconds, expected):
    assert calculate_page_throughput(total_pages, elapsed_seconds) == expected


# ============================================================================
# ProcessingTimer Tests
# ============================================================================

def test_timer_initialization():
    timer = ProcessingTimer()
    assert timer.durations == []
    assert timer._active_timers == 0


def test_single_time_block(monkeypatch):
    timer = ProcessingTimer()

    times = [0.0, 1.5]

    def mock_perf_counter():
        return times.pop(0)

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    with timer.time_block():
        assert timer._active_timers == 1

    assert timer._active_timers == 0
    assert len(timer.durations) == 1
    assert timer.durations[0] == 1.5


def test_nested_time_blocks(monkeypatch):
    timer = ProcessingTimer()

    # Enter outer (0.0), enter inner (1.0), exit inner (2.0), exit outer (3.5)
    times = [0.0, 1.0, 2.0, 3.5]

    def mock_perf_counter():
        return times.pop(0)

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    with timer.time_block():
        assert timer._active_timers == 1
        with timer.time_block():
            assert timer._active_timers == 2
        assert timer._active_timers == 1

    assert timer._active_timers == 0
    assert len(timer.durations) == 2
    # Inner duration: 2.0 - 1.0 = 1.0
    assert timer.durations[0] == 1.0
    # Outer duration: 3.5 - 0.0 = 3.5
    assert timer.durations[1] == 3.5


def test_exception_handling_in_timer(monkeypatch):
    timer = ProcessingTimer()

    times = [0.0, 1.2]

    def mock_perf_counter():
        return times.pop(0)

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    with pytest.raises(ValueError, match="Test error"):
        with timer.time_block():
            assert timer._active_timers == 1
            raise ValueError("Test error")

    assert timer._active_timers == 0
    assert len(timer.durations) == 1
    assert timer.durations[0] == 1.2


def test_nested_timers_with_inner_exception(monkeypatch):
    timer = ProcessingTimer()

    # Enter outer (0.0), enter inner (1.0), exit inner exception (2.0), exit outer exception (3.0)
    times = [0.0, 1.0, 2.0, 3.0]

    def mock_perf_counter():
        return times.pop(0)

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    with pytest.raises(RuntimeError):
        with timer.time_block():
            with timer.time_block():
                raise RuntimeError("Failed")

    assert timer._active_timers == 0
    assert len(timer.durations) == 2
    assert timer.durations[0] == 1.0
    assert timer.durations[1] == 3.0


# ============================================================================
# Throughput Tests
# ============================================================================


class TestProcessingThroughput:
    """Test suite for throughput calculation and formatting."""

    @pytest.mark.parametrize(
        ("total_bytes", "elapsed_seconds", "expected_throughput"),
        [
            (1024, 1.0, 1.0),  # Exactly 1 KB in 1 second
            (2048, 2.0, 1.0),  # 2 KB in 2 seconds
            (1048576, 1.0, 1024.0),  # 1 MB in 1 second -> 1024 KB/s
            (512, 0.5, 1.0),  # 0.5 KB in 0.5 seconds
            (10240, 3.33, 3.08),  # 10 KB in 3.33 seconds (rounded)
            (0, 5.0, 0.0),  # 0 bytes processed
        ],
    )
    def test_calculate_processing_throughput_valid(
        self, total_bytes, elapsed_seconds, expected_throughput
    ):
        """Test throughput calculation with valid inputs."""
        result = calculate_processing_throughput(total_bytes, elapsed_seconds)
        assert result == pytest.approx(expected_throughput, rel=1e-2)

    @pytest.mark.parametrize("elapsed_seconds", [0.0, -1.0, -0.001])
    def test_calculate_processing_throughput_zero_or_negative_time(
        self, elapsed_seconds
    ):
        """Test that throughput returns 0.0 when elapsed_seconds <= 0."""
        result = calculate_processing_throughput(1024, elapsed_seconds)
        assert result == 0.0

    @pytest.mark.parametrize(
        ("throughput_kbps", "expected_string"),
        [
            (0.0, "0.00 KB/s"),
            (1.5, "1.50 KB/s"),
            (500.25, "500.25 KB/s"),
            (1023.99, "1023.99 KB/s"),
            (1024.0, "1.00 MB/s"),
            (2048.5, "2.00 MB/s"),
            (15360.0, "15.00 MB/s"),
        ],
    )
    def test_format_throughput_human_readable(self, throughput_kbps, expected_string):
        """Test human-readable formatting of throughput values."""
        result = format_throughput_human_readable(throughput_kbps)
        assert result == expected_string


# ============================================================================
# Duration and Helper Function Tests
# ============================================================================


class TestProcessingDurationHelpers:
    """Test suite for duration estimation and formatting."""

    @pytest.mark.parametrize(
        ("total_bytes", "seconds_per_mb", "expected_seconds"),
        [
            (0, 2.0, 0),
            (100, 2.0, 1),  # Minimum 1 second
            (1048576, 2.0, 2),  # 1 MB at 2 sec/MB
            (5242880, 1.5, 8),  # 5 MB at 1.5 sec/MB
        ],
    )
    def test_estimate_processing_seconds(
        self, total_bytes, seconds_per_mb, expected_seconds
    ):
        """Test processing time estimation."""
        result = estimate_processing_seconds(total_bytes, seconds_per_mb=seconds_per_mb)
        assert result == expected_seconds

    @pytest.mark.parametrize(
        ("seconds", "expected_string"),
        [
            (0, "less than a second"),
            (1, "1 second"),
            (45, "45 seconds"),
            (60, "1 minute"),
            (125, "2 minutes 5 seconds"),
            (3600, "1 hour"),
            (3665, "1 hour 1 minute 5 seconds"),
        ],
    )
    def test_format_processing_duration(self, seconds, expected_string):
        """Test human-readable duration formatting."""
        result = format_processing_duration(seconds)
        assert result == expected_string

    def test_format_processing_duration_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            format_processing_duration("not a number")

    def test_format_processing_duration_negative(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError):
            format_processing_duration(-10)

    @pytest.mark.parametrize(
        ("seconds", "expected_string"),
        [
            (0.0, "0.0s"),
            (45.2, "45.2s"),
            (59.9, "59.9s"),
            (60.0, "1m 0s"),
            (125.0, "2m 5s"),
        ],
    )
    def test_format_duration(self, seconds, expected_string):
        """Test concise float duration formatting."""
        result = format_duration(seconds)
        assert result == expected_string

    def test_format_duration_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            format_duration("not a number")

    def test_format_duration_negative(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError):
            format_duration(-1.5)


@pytest.mark.parametrize(
    ("total_bytes", "expected"),
    [
        (0, 0),
        (1, 1),
        (BYTES_PER_MB // 2, 1),
        (BYTES_PER_MB, 2),
        (10 * BYTES_PER_MB, 20),
        (50 * BYTES_PER_MB, 100),
    ],
)
def test_estimate_processing_seconds_default_rate(total_bytes, expected):
    assert estimate_processing_seconds(total_bytes) == expected


def test_custom_rate_is_supported():
    assert (
        estimate_processing_seconds(
            5 * BYTES_PER_MB,
            seconds_per_mb=3.0,
        )
        == 15
    )


@pytest.mark.parametrize(
    "value",
    [-1, float("inf"), float("nan")],
)
def test_invalid_total_bytes_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        estimate_processing_seconds(value)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "less than a second"),
        (1, "1 second"),
        (45, "45 seconds"),
        (60, "1 minute"),
        (75, "1 minute 15 seconds"),
        (120, "2 minutes"),
        (3600, "1 hour"),
        (3720, "1 hour 2 minutes"),
    ],
)
def test_format_processing_duration_cases(seconds, expected):
    assert format_processing_duration(seconds) == expected


# ============================================================================
# File Upload Helpers Tests
# ============================================================================


class UploadedWithSize:
    def __init__(self, size):
        self.size = size


class UploadedWithValue:
    def __init__(self, value: bytes):
        self._value = value

    def getvalue(self):
        return self._value


def test_uploaded_file_sizes_are_summed():
    files = [
        UploadedWithSize(10),
        UploadedWithValue(b"12345"),
    ]
    assert uploaded_files_total_bytes(files) == 15


def test_upload_without_size_or_getvalue_is_rejected():
    with pytest.raises(TypeError):
        uploaded_files_total_bytes([object()])


def test_eta_text_uses_default_rate():
    assert processing_eta_text(2 * BYTES_PER_MB) == (
        "Estimated processing time: about 4 seconds"
    )
def test_calculate_mb_per_minute():
    # Test normal calculation: 10 MB in 60 seconds (1 minute) = 10.0 MB/min
    ten_mb_in_bytes = 10 * 1024 * 1024
    assert calculate_mb_per_minute(ten_mb_in_bytes, 60.0) == 10.0

    # Test zero or negative elapsed time returns 0.0
    assert calculate_mb_per_minute(ten_mb_in_bytes, 0.0) == 0.0
    assert calculate_mb_per_minute(ten_mb_in_bytes, -5.0) == 0.0

    # Test zero bytes processed
    assert calculate_mb_per_minute(0, 60.0) == 0.0