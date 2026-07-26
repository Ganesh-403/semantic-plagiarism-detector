import pytest
import time
from src.utils.processing_time import ProcessingTimer

def test_timer_initialization():
    timer = ProcessingTimer()
    assert timer.durations == []
    assert timer._active_timers == 0

def test_single_time_block(monkeypatch):
    timer = ProcessingTimer()
    
    times = [0.0, 1.5]
    def mock_perf_counter():
        return times.pop(0)
    
    monkeypatch.setattr(time, 'perf_counter', mock_perf_counter)
    
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
    
    monkeypatch.setattr(time, 'perf_counter', mock_perf_counter)
    
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
    
    monkeypatch.setattr(time, 'perf_counter', mock_perf_counter)
    
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
    
    monkeypatch.setattr(time, 'perf_counter', mock_perf_counter)
    
    with pytest.raises(RuntimeError):
        with timer.time_block():
            with timer.time_block():
                raise RuntimeError("Failed")
                
    assert timer._active_timers == 0
    assert len(timer.durations) == 2
    assert timer.durations[0] == 1.0
    assert timer.durations[1] == 3.0

from src.utils.processing_time import (BYTES_PER_MB,
                                       estimate_processing_seconds,
                                       format_processing_duration,
                                       processing_eta_text,
                                       uploaded_files_total_bytes)


class UploadedWithSize:
    def __init__(self, size):
        self.size = size


class UploadedWithValue:
    def __init__(self, value: bytes):
        self._value = value

    def getvalue(self):
        return self._value


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
def test_estimate_processing_seconds(total_bytes, expected):
    assert estimate_processing_seconds(total_bytes) == expected


def test_custom_rate_is_supported():
    assert estimate_processing_seconds(
        5 * BYTES_PER_MB,
        seconds_per_mb=3.0,
    ) == 15


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
def test_format_processing_duration(seconds, expected):
    assert format_processing_duration(seconds) == expected


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
