"""
tests/core/test_revision_burst.py
---------------------------------
Unit tests for Document Revision Time-Series and Burst Analysis.
"""

import pytest
from src.core.revision_timeseries_analyzer import (
    compute_inter_keystroke_deltas,
    compute_burst_metrics,
)
from src.core.burst_detection_engine import (
    compute_poisson_deviation,
    analyze_revision_bursts,
)


class TestRevisionTimeseriesAnalyzer:
    def test_compute_inter_keystroke_deltas(self):
        timestamps = [0.0, 0.1, 0.2, 0.5]
        deltas = compute_inter_keystroke_deltas(timestamps)
        assert len(deltas) == 3
        assert deltas[0] == 0.1

    def test_compute_burst_metrics_burst(self):
        # Fast typing (burst)
        timestamps = [0.0, 0.01, 0.02, 0.03, 0.04]
        metrics = compute_burst_metrics(timestamps, burst_threshold=0.05)
        assert metrics["burst_ratio"] > 0.5

    def test_compute_burst_metrics_organic(self):
        # Organic typing
        timestamps = [0.0, 0.2, 0.5, 0.9, 1.5]
        metrics = compute_burst_metrics(timestamps, burst_threshold=0.05)
        assert metrics["burst_ratio"] == 0.0


class TestBurstDetectionEngine:
    def test_compute_poisson_deviation_anomalous(self):
        # High variance (burst)
        deltas = [0.01, 0.01, 0.01, 5.0, 0.01, 0.01]
        dev = compute_poisson_deviation(deltas)
        assert dev["is_anomalous"] is True

    def test_analyze_revision_bursts_ghostwritten(self):
        timestamps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        result = analyze_revision_bursts(timestamps)
        assert result["is_ghostwritten"] is True
        assert result["risk_score"] > 0.6
