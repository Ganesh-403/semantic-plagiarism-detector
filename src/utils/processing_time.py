# filepath: src/utils/processing_time.py
"""
processing_time.py
------------------
Helpers for estimating, tracking, and beautifully rendering document processing time.
Includes full hierarchical profiling logic, throughput calculation, and dark/light mode Streamlit UI table generation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from numbers import Real
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StageTiming:
    """Data class representing timing for a specific pipeline stage."""

    stage_name: str
    duration_seconds: float


BYTES_PER_MB = 1024 * 1024
BYTES_PER_KB = 1024
DEFAULT_SECONDS_PER_MB = 2.0


# ============================================================================
# HIERARCHICAL EXECUTION PROFILER
# ============================================================================


class ProfilerSpan:
    """Represents a single measurable unit of work."""

    def __init__(self, name: str, parent: Optional["ProfilerSpan"] = None):
        self.name = name
        self.parent = parent
        self.children: List["ProfilerSpan"] = []
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration: float = 0.0

    def end(self) -> None:
        """Finalize the span and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert the span and its children to a dictionary."""
        return {
            "name": self.name,
            "duration_sec": self.duration,
            "children": [child.to_dict() for child in self.children],
        }


class ProcessingTimer:
    """
    Advanced hierarchical timer for capturing execution breakdowns.
    Automatically maintains tree structure of nested time_block calls.
    """

    def __init__(self):
        self.durations: List[float] = []
        self.spans: List[ProfilerSpan] = []
        self._active_stack: List[ProfilerSpan] = []
        self._active_timers: int = 0
        self._aggregate_stats: Dict[str, float] = defaultdict(float)

    @contextmanager
    def time_block(self, name: str = "Unnamed Block"):
        """
        Context manager for timing a block of code.
        Can be nested recursively.
        """
        parent = self._active_stack[-1] if self._active_stack else None
        span = ProfilerSpan(name, parent)

        if parent:
            parent.children.append(span)
        else:
            self.spans.append(span)

        self._active_stack.append(span)
        self._active_timers += 1

        try:
            yield self
        finally:
            span.end()
            self._active_stack.pop()
            self._active_timers -= 1

            if parent is None:
                self.durations.append(span.duration)

            self._aggregate_stats[name] += span.duration

    def get_summary(self) -> Dict[str, float]:
        """Returns aggregated durations for all named blocks."""
        return dict(self._aggregate_stats)


# ============================================================================
# THROUGHPUT & SPEED METRICS
# ============================================================================


def calculate_processing_throughput(total_bytes: int, elapsed_seconds: float) -> float:
    """
    Calculate the processing throughput in Kilobytes per second (KB/sec).

    Args:
        total_bytes: The total number of bytes processed.
        elapsed_seconds: The total time elapsed in seconds.

    Returns:
        float: The throughput in KB/sec. Returns 0.0 if elapsed_seconds <= 0.
    """
    if elapsed_seconds <= 0:
        return 0.0

    total_kb = total_bytes / 1024 
    throughput = total_kb / elapsed_seconds
    return round(throughput, 2)


def calculate_mb_per_minute(total_bytes: int, elapsed_seconds: float) -> float:
    """
    Calculate throughput in Megabytes (MB) per minute.

    Args:
        total_bytes: The total number of bytes processed.
        elapsed_seconds: The total time elapsed in seconds.

    Returns:
        float: The throughput in MB/min. Returns 0.0 if elapsed_seconds <= 0.
    """
    if elapsed_seconds <= 0:
        return 0.0

    total_mb = total_bytes / (1024 * 1024)
    minutes = elapsed_seconds / 60.0
    return round(total_mb / minutes, 2)


def calculate_average_latency(latencies: list[float]) -> float:
    """Calculate the average latency from a list of latencies."""
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies), 3)


def calculate_page_throughput(total_pages: int, elapsed_seconds: float) -> float:
    """
    Calculate the processing throughput in pages per second.

    Args:
        total_pages: The total number of pages processed.
        elapsed_seconds: The total time elapsed in seconds.

    Returns:
        float: The throughput in pages/sec, rounded to 2 decimal places.
            Returns 0.0 if elapsed_seconds <= 0.
    """
    if elapsed_seconds <= 0:
        return 0.0

    throughput = total_pages / elapsed_seconds
    return round(throughput, 2)


def format_throughput_human_readable(throughput_kbps: float) -> str:
    """
    Format a throughput value in KB/sec to a human-readable string.

    Args:
        throughput_kbps: The throughput in KB/sec.

    Returns:
        str: A formatted string (e.g., "1.50 KB/s", "2.30 MB/s").
    """
    throughput = _validate_non_negative_number("throughput_kbps", throughput_kbps)
    if throughput < 1024:
        return f"{throughput:.2f} KB/s"

    return f"{throughput / 1024:.2f} MB/s"

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================


class TimingUIRenderer:
    """
    Handles rendering the timing summary table into a Streamlit debug expander.
    Generates dynamic CSS-injected HTML tables to match the active theme.
    """

    @staticmethod
    def _generate_css(is_dark_mode: bool) -> str:
        """Generates appropriate CSS variables for light or dark themes."""
        if is_dark_mode:
            return """
            <style>
                .timing-table { width: 100%; border-collapse: collapse; color: #FAFAFA; }
                .timing-table th, .timing-table td { padding: 8px; text-align: left; border-bottom: 1px solid #444; }
                .timing-table th { background-color: #333; }
            </style>
            """
        else:
            return """
            <style>
                .timing-table { width: 100%; border-collapse: collapse; color: #333; }
                .timing-table th, .timing-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .timing-table th { background-color: #f5f5f5; }
            </style>
            """

    @classmethod
    def render_debug_expander(
        cls, timer: ProcessingTimer, is_dark_mode: bool = False, st_module: Any = None
    ) -> None:
        """
        Renders the timing breakdown in a Streamlit expander.

        Args:
            timer: Completed ProcessingTimer instance.
            is_dark_mode: Boolean indicating user's current theme preference.
            st_module: Streamlit library reference (passed dependency).
        """
        if st_module is None:
            import streamlit as st_module

        summary = timer.get_summary()
        if not summary:
            st_module.info("No timing data available.")
            return

        total_time = sum(summary.values())
        sorted_items = sorted(summary.items(), key=lambda x: x[1], reverse=True)

        with st_module.expander("⏱️ Execution Time Breakdown", expanded=False):
            css = cls._generate_css(is_dark_mode)
            html_rows = []

            for name, duration in sorted_items:
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                html_rows.append(
                    f"<tr><td>{name}</td><td>{duration:.3f}s</td><td>{percentage:.1f}%</td></tr>"
                )

            html_table = f"""
            <table class="timing-table">
                <thead>
                    <tr><th>Operation Phase</th><th>Duration (Seconds)</th><th>% of Total</th></tr>
                </thead>
                <tbody>
                    {"".join(html_rows)}
                    <tr style="font-weight: bold;">
                        <td>Total Measured</td>
                        <td>{total_time:.3f}s</td>
                        <td>100.0%</td>
                    </tr8>
                </tbody>
            </table>
            """
            st_module.markdown(css + html_table, unsafe_allow_html=True)


# ============================================================================
# ESTIMATION MATH & VALIDATION
# ============================================================================


def _validate_non_negative_number(name: str, value: Real) -> float:
    """Validate that a value is a finite, non-negative real number."""
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
    byte_count = _validate_non_negative_number("total_bytes", total_bytes)
    rate = _validate_non_negative_number("seconds_per_mb", seconds_per_mb)

    if byte_count == 0 or rate == 0:
        return 0

    estimated = (byte_count / BYTES_PER_MB) * rate
    return max(1, math.ceil(estimated))


def format_duration(seconds: float) -> str:
    """
    Format a floating-point execution time into a concise string.

    Args:
        seconds: The duration in seconds.

    Returns:
        str: A formatted string (e.g., "45.2s" or "2m 5s").
    """
    numeric = _validate_non_negative_number("seconds", seconds)

    if numeric < 60.0:
        return f"{numeric:.1f}s"

    minutes, remaining_seconds = divmod(numeric, 60)
    return f"{int(minutes)}m {int(remaining_seconds)}s"


def uploaded_files_total_bytes(files: Iterable[Any]) -> int:
    """Return the total byte size for Streamlit-like uploaded files."""
    total = 0
    for uploaded_file in files:
        size = getattr(uploaded_file, "size", None)
        if size is None:
            getvalue = getattr(uploaded_file, "getvalue", None)
            if not callable(getvalue):
                raise TypeError(
                    "Each uploaded file must expose either size or getvalue()."
                )
            size = len(getvalue())

        numeric_size = _validate_non_negative_number("file size", size)
        if not numeric_size.is_integer():
            raise ValueError("file size must be an integer number of bytes.")
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
        return f"{minutes} {minute_unit} {remaining_seconds} seconds"

    hours, remaining_minutes = divmod(minutes, 60)
    hour_unit = "hour" if hours == 1 else "hours"
    if remaining_minutes == 0:
        return f"{hours} {hour_unit}"

    minute_unit = "minute" if remaining_minutes == 1 else "minutes"
    return f"{hours} {hour_unit} {remaining_minutes} {minute_unit}"


def processing_eta_text(
    total_bytes: int,
    *,
    seconds_per_mb: float = DEFAULT_SECONDS_PER_MB,
) -> str:
    """Return the user-facing ETA sentence."""
    seconds = estimate_processing_seconds(total_bytes, seconds_per_mb=seconds_per_mb)
    duration = format_processing_duration(seconds)
    return f"Estimated processing time: about {duration}"
def calculate_mb_per_minute(total_bytes: int, elapsed_seconds: float) -> float:
    """
    Calculate document processing throughput in megabytes per minute (MB/min).
    
    Args:
        total_bytes (int): Total size processed in bytes.
        elapsed_seconds (float): Time elapsed in seconds.
        
    Returns:
        float: Throughput in MB/min rounded to 2 decimal places. Returns 0.0 if elapsed_seconds <= 0.
    >"""
    if elapsed_seconds <= 0 or total_bytes <= 0:
        return 0.0
    
    megabytes = total_bytes / (1024 * 1024)
    minutes = elapsed_seconds / 60.0
    
    return round(megabytes / minutes, 2)
