from __future__ import annotations

import time
from contextlib import contextmanager

class ProcessingTimer:
    def __init__(self):
        self.durations = []
        self._active_timers = 0

    @contextmanager
    def time_block(self):
        start = time.perf_counter()
        self._active_timers += 1
        try:
            yield self
        finally:
            end = time.perf_counter()
            self._active_timers -= 1
            self.durations.append(end - start)
"""Helpers for estimating and formatting document processing time."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any

BYTES_PER_MB = 1024 * 1024
DEFAULT_SECONDS_PER_MB = 2.0


def _validate_non_negative_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def estimate_processing_seconds(
    total_bytes: int,
    *,
    seconds_per_mb: float = DEFAULT_SECONDS_PER_MB,
) -> int:
    """Estimate processing time from total uploaded bytes."""
    byte_count = _validate_non_negative_number(
        "total_bytes",
        total_bytes,
    )
    rate = _validate_non_negative_number(
        "seconds_per_mb",
        seconds_per_mb,
    )

    if byte_count == 0 or rate == 0:
        return 0

    estimated = (byte_count / BYTES_PER_MB) * rate
    return max(1, math.ceil(estimated))


def uploaded_files_total_bytes(files: Iterable[Any]) -> int:
    """Return the total byte size for Streamlit-like uploaded files."""
    total = 0

    for uploaded_file in files:
        size = getattr(uploaded_file, "size", None)

        if size is None:
            getvalue = getattr(uploaded_file, "getvalue", None)
            if not callable(getvalue):
                raise TypeError(
                    "Each uploaded file must expose either size "
                    "or getvalue()."
                )
            size = len(getvalue())

        numeric_size = _validate_non_negative_number(
            "file size",
            size,
        )
        if not numeric_size.is_integer():
            raise ValueError(
                "file size must be an integer number of bytes."
            )
        total += int(numeric_size)

    return total


def format_processing_duration(seconds: int) -> str:
    """Return a concise human-readable duration."""
    numeric = _validate_non_negative_number("seconds", seconds)
    if not numeric.is_integer():
        raise ValueError("seconds must be an integer.")

    total_seconds = int(numeric)

    if total_seconds == 0:
        return "less than a second"
    if total_seconds < 60:
        unit = "second" if total_seconds == 1 else "seconds"
        return f"{total_seconds} {unit}"

    minutes, remaining_seconds = divmod(total_seconds, 60)

    if minutes < 60:
        minute_unit = "minute" if minutes == 1 else "minutes"
        if remaining_seconds == 0:
            return f"{minutes} {minute_unit}"
        return (
            f"{minutes} {minute_unit} "
            f"{remaining_seconds} seconds"
        )

    hours, remaining_minutes = divmod(minutes, 60)
    hour_unit = "hour" if hours == 1 else "hours"

    if remaining_minutes == 0:
        return f"{hours} {hour_unit}"

    minute_unit = (
        "minute" if remaining_minutes == 1 else "minutes"
    )
    return (
        f"{hours} {hour_unit} "
        f"{remaining_minutes} {minute_unit}"
    )


def processing_eta_text(
    total_bytes: int,
    *,
    seconds_per_mb: float = DEFAULT_SECONDS_PER_MB,
) -> str:
    """Return the user-facing ETA sentence."""
    seconds = estimate_processing_seconds(
        total_bytes,
        seconds_per_mb=seconds_per_mb,
    )
    duration = format_processing_duration(seconds)
    return f"Estimated processing time: about {duration}"
