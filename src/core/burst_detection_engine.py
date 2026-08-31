"""
src/core/burst_detection_engine.py
----------------------------------
Burst Detection Engine for Ghostwriting Detection.

Applies statistical anomaly detection (e.g., Poisson process deviation)
to identify unnatural copy-paste bursts in document revision histories.
"""

import math
import logging
from typing import List, Dict, Any
from statistics import variance, mean

logger = logging.getLogger(__name__)


def compute_poisson_deviation(deltas: List[float]) -> Dict[str, float]:
    """Compute deviation from a Poisson process for keystroke intervals.

    Human typing intervals roughly follow a Poisson-like distribution
    where the variance is approximately equal to the mean. Copy-paste
    bursts create massive variance spikes.

    Args:
        deltas: List of inter-keystroke time deltas.

    Returns:
        Dictionary containing Poisson deviation metrics.
    """
    if len(deltas) < 10:
        return {"variance_mean_ratio": 0.0, "is_anomalous": False}

    m = mean(deltas)
    v = variance(deltas)

    if m == 0:
        return {"variance_mean_ratio": 0.0, "is_anomalous": False}

    # For a Poisson process, variance ≈ mean, so ratio ≈ 1.
    # A ratio >> 1 indicates overdispersion (bursts).
    ratio = v / m

    # Threshold for anomaly detection (heuristic)
    is_anomalous = ratio > 3.0

    return {"variance_mean_ratio": round(ratio, 4), "is_anomalous": is_anomalous}


def analyze_revision_bursts(timestamps: List[float]) -> Dict[str, Any]:
    """Analyze revision telemetry for ghostwriting bursts.

    Args:
        timestamps: List of keystroke timestamps.

    Returns:
        Dictionary containing burst metrics and anomaly flags.
    """
    from src.core.revision_timeseries_analyzer import (
        compute_inter_keystroke_deltas,
        compute_burst_metrics,
    )

    deltas = compute_inter_keystroke_deltas(timestamps)
    burst_metrics = compute_burst_metrics(timestamps)
    poisson_dev = compute_poisson_deviation(deltas)

    # Overall ghostwriting risk score
    risk_score = 0.0
    if burst_metrics["burst_ratio"] > 0.20:
        risk_score += 0.5
    if poisson_dev["is_anomalous"]:
        risk_score += 0.5

    risk_score = min(1.0, risk_score)

    return {
        "burst_metrics": burst_metrics,
        "poisson_deviation": poisson_dev,
        "risk_score": round(risk_score, 4),
        "is_ghostwritten": risk_score > 0.6,
    }
