"""
src/core/revision_timeseries_analyzer.py
----------------------------------------
Document Revision Time-Series Analyzer.

Parses document revision timestamps and keystroke telemetry to compute
inter-keystroke and inter-burst time deltas for ghostwriting detection.
"""

import logging
from typing import List, Dict, Any
from statistics import variance, mean

logger = logging.getLogger(__name__)


def compute_inter_keystroke_deltas(timestamps: List[float]) -> List[float]:
    """Compute time deltas between consecutive keystrokes.

    Args:
        timestamps: List of keystroke timestamps in seconds.

    Returns:
        List of time deltas in seconds.
    """
    if not timestamps or len(timestamps) < 2:
        return []

    deltas = []
    for i in range(len(timestamps) - 1):
        delta = timestamps[i + 1] - timestamps[i]
        if delta >= 0:
            deltas.append(delta)

    return deltas


def compute_burst_metrics(
    timestamps: List[float], burst_threshold: float = 0.05
) -> Dict[str, Any]:
    """Identify copy-paste bursts based on keystroke timing.

    A burst is defined as a sequence of characters typed faster than
    humanly possible (e.g., < 50ms per keystroke), indicating a paste event.

    Args:
        timestamps: List of keystroke timestamps.
        burst_threshold: Minimum time between keystrokes to be considered organic.

    Returns:
        Dictionary containing burst metrics.
    """
    deltas = compute_inter_keystroke_deltas(timestamps)
    if not deltas:
        return {"burst_count": 0, "burst_ratio": 0.0, "mean_delta": 0.0}

    burst_chars = sum(1 for d in deltas if d < burst_threshold)
    total_chars = len(deltas)
    burst_ratio = burst_chars / total_chars if total_chars > 0 else 0.0

    mean_delta = mean(deltas)

    return {
        "burst_count": burst_chars,
        "burst_ratio": round(burst_ratio, 4),
        "mean_delta": round(mean_delta, 4),
        "total_chars": total_chars,
    }
