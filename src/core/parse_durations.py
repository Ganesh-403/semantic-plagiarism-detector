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
src/core/parse_durations.py
---------------------------
In-memory registry for file-parsing execution durations (Issue #1728).

Tracks the elapsed wall-clock time for each file's text extraction so
the UI can display "Parsed in X.XXs" on document detail summary cards.

The registry is intentionally process-local (not persisted to disk)
because parse duration is a transient diagnostic metric — it only
matters for the current session's UI, not for historical analysis.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

_lock = threading.Lock()
_durations: dict[str, float] = {}


def record_parse_duration(filename: str, duration_sec: float) -> None:
    """Record the parse duration for a filename.

    Args:
        filename: The name of the file that was parsed.
        duration_sec: The elapsed time in seconds (monotonic preferred).
    """
    with _lock:
        _durations[filename] = float(duration_sec)


def get_parse_duration(filename: str) -> float | None:
    """Return the recorded parse duration for a filename, or None.

    Args:
        filename: The name of the file.

    Returns:
        The elapsed seconds as a float, or None if no duration was
        recorded for this filename.
    """
    with _lock:
        return _durations.get(filename)


def get_all_parse_durations() -> dict[str, float]:
    """Return a snapshot copy of all recorded durations."""
    with _lock:
        return dict(_durations)


def clear_parse_durations() -> None:
    """Clear all recorded durations. Called at the start of each scan."""
    with _lock:
        _durations.clear()


def format_duration(duration_sec: float | None) -> str:
    """Format a duration for display, e.g. '0.42s'.

    Args:
        duration_sec: The elapsed seconds, or None.

    Returns:
        A string like '0.42s', or an empty string if duration_sec is None.
    """
    if duration_sec is None:
        return ""
    return f"{duration_sec:.2f}s"
