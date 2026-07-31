# filepath: src/utils/processing_time.py
"""
processing_time.py
------------------
Helpers for estimating, tracking, and beautifully rendering document processing time.
Includes full hierarchical profiling logic and dark/light mode Streamlit UI table generation.
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
DEFAULT_SECONDS_PER_MB = 2.0


# ============================================================================
# HIERARCHICAL EXECUTION PROFILER
# ============================================================================

class ProfilerSpan:
    """Represents a single measurable unit of work."""
    
    def __init__(self, name: str, parent: Optional['ProfilerSpan'] = None):
        self.name = name
        self.parent = parent
        self.children: List['ProfilerSpan'] = []
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration: float = 0.0
        
    def end(self):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_sec": self.duration,
            "children": [child.to_dict() for child in self.children]
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
                .timing-table-wrapper { width: 100%; margin: 1rem 0; overflow-x: auto; }
                .timing-table { width: 100%; border-collapse: collapse; font-family: monospace; font-size: 14px; }
                .timing-table th { background-color: #262730; color: #fafafa; text-align: left; padding: 10px; border-bottom: 2px solid #4B4B4B; }
                .timing-table td { background-color: #0e1117; color: #dcdcdc; padding: 10px; border-bottom: 1px solid #2B2B2B; }
                .timing-table tr:hover td { background-color: #1e2129; }
                .timing-metric { font-weight: bold; color: #4DA2FF; }
            </style>
            """
        else:
            return """
            <style>
                .timing-table-wrapper { width: 100%; margin: 1rem 0; overflow-x: auto; }
                .timing-table { width: 100%; border-collapse: collapse; font-family: monospace; font-size: 14px; }
                .timing-table th { background-color: #f0f2f6; color: #31333F; text-align: left; padding: 10px; border-bottom: 2px solid #d3d5db; }
                .timing-table td { background-color: #ffffff; color: #31333F; padding: 10px; border-bottom: 1px solid #eceeef; }
                .timing-table tr:hover td { background-color: #f9f9f9; }
                .timing-metric { font-weight: bold; color: #0068c9; }
            </style>
            """

    @classmethod
    def render_debug_expander(
        cls, 
        timer: ProcessingTimer, 
        is_dark_mode: bool = False,
        st_module: Any = None
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
        
        # Sort by duration descending
        sorted_items = sorted(summary.items(), key=lambda x: x[1], reverse=True)

        with st_module.expander("⏱️ Execution Time Breakdown", expanded=False):
            css = cls._generate_css(is_dark_mode)
            
            html_rows = []
            for name, duration in sorted_items:
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                html_rows.append(
                    f"<tr>"
                    f"<td>{name}</td>"
                    f"<td class='timing-metric'>{duration:.3f}s</td>"
                    f"<td>{percentage:.1f}%</td>"
                    f"</tr>"
                )
                
            html_table = f"""
            <div class="timing-table-wrapper">
                <table class="timing-table">
                    <thead>
                        <tr>
                            <th>Operation Phase</th>
                            <th>Duration (Seconds)</th>
                            <th>% of Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(html_rows)}
                        <tr style="border-top: 2px solid #888;">
                            <td><strong>Total Measured</strong></td>
                            <td class='timing-metric'><strong>{total_time:.3f}s</strong></td>
                            <td><strong>100.0%</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """
            
            st_module.markdown(css + html_table, unsafe_allow_html=True)


# ============================================================================
# ESTIMATION MATH & VALIDATION
# ============================================================================

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


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds.

    Examples:
        0.0   -> "0.0s"
        45.2  -> "45.2s"
        125.0 -> "2m 5.0s"
    """

    seconds = _validate_non_negative_number("seconds", seconds)

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    return f"{minutes}m {remaining:.1f}s"
