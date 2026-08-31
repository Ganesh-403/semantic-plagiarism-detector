"""Tests for src.core.trend_tracker."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.trend_tracker import (
    AlertSeverity,
    PlagiarismTrendTracker,
    ScanSnapshot,
    TrendAlert,
    TrendAnalysisResult,
    TrendDirection,
    TrendMetrics,
    _compute_linear_regression,
    _compute_moving_average,
    _compute_volatility,
    _detect_outliers_zscore,
    create_demo_snapshots,
    snapshots_from_dicts,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestComputeLinearRegression:
    def test_perfect_linear(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        slope, intercept, r2 = _compute_linear_regression(x, y)
        assert abs(slope - 1.0) < 1e-10
        assert abs(intercept - 1.0) < 1e-10
        assert abs(r2 - 1.0) < 1e-10

    def test_no_correlation(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 1.0])
        slope, intercept, r2 = _compute_linear_regression(x, y)
        assert abs(slope) < 1e-10
        assert abs(r2) < 1e-10

    def test_single_point(self):
        slope, intercept, r2 = _compute_linear_regression(np.array([0.0]), np.array([5.0]))
        assert abs(intercept - 5.0) < 1e-10

    def test_empty(self):
        slope, intercept, r2 = _compute_linear_regression(np.array([]), np.array([]))
        assert slope == 0.0


class TestComputeMovingAverage:
    def test_basic(self):
        values = [1, 2, 3, 4, 5]
        ma = _compute_moving_average(values, 3)
        assert len(ma) == 5
        assert abs(ma[0] - 1.0) < 1e-10
        assert abs(ma[2] - 2.0) < 1e-10
        assert abs(ma[4] - 4.0) < 1e-10

    def test_window_equals_length(self):
        values = [1, 2, 3]
        ma = _compute_moving_average(values, 3)
        assert abs(ma[2] - 2.0) < 1e-10

    def test_single_value(self):
        ma = _compute_moving_average([5.0], 3)
        assert len(ma) == 1


class TestComputeVolatility:
    def test_constant(self):
        vol = _compute_volatility([1.0, 1.0, 1.0])
        assert vol == 0.0

    def test_varying(self):
        vol = _compute_volatility([0.0, 1.0, 0.0, 1.0])
        assert vol > 0.0

    def test_single(self):
        vol = _compute_volatility([1.0])
        assert vol == 0.0


class TestDetectOutliersZscore:
    def test_no_outliers(self):
        values = [1.0, 1.1, 0.9, 1.0, 1.05]
        outliers = _detect_outliers_zscore(values, 3.0)
        assert len(outliers) == 0

    def test_with_outlier(self):
        values = [1.0, 1.0, 1.0, 1.0, 10.0]
        outliers = _detect_outliers_zscore(values, 2.0)
        assert len(outliers) >= 1

    def test_too_few_values(self):
        assert _detect_outliers_zscore([1.0, 2.0], 2.0) == []


# ---------------------------------------------------------------------------
# ScanSnapshot
# ---------------------------------------------------------------------------

class TestScanSnapshot:
    def test_to_dict(self):
        s = ScanSnapshot("2025-01-01T00:00:00", 10, 0.45, 0.88, 3, 0.59)
        d = s.to_dict()
        assert d["document_count"] == 10
        assert d["avg_similarity"] == 0.45

    def test_from_dict(self):
        d = {
            "timestamp": "2025-01-01T00:00:00",
            "document_count": 10,
            "avg_similarity": 0.45,
            "max_similarity": 0.88,
            "flagged_count": 3,
            "threshold_used": 0.59,
        }
        s = ScanSnapshot.from_dict(d)
        assert s.document_count == 10

    def test_frozen(self):
        s = ScanSnapshot("2025-01-01T00:00:00", 10, 0.45, 0.88, 3, 0.59)
        with pytest.raises(AttributeError):
            s.document_count = 20


# ---------------------------------------------------------------------------
# PlagiarismTrendTracker
# ---------------------------------------------------------------------------

class TestPlagiarismTrendTracker:
    def _make_snapshots(self, n=10):
        """Create synthetic increasing-trend snapshots."""
        return [
            ScanSnapshot(
                timestamp=f"2025-01-{i+1:02d}T00:00:00",
                document_count=20,
                avg_similarity=0.40 + i * 0.02,
                max_similarity=0.70 + i * 0.02,
                flagged_count=2 + i,
                threshold_used=0.59,
            )
            for i in range(n)
        ]

    def test_add_and_count(self):
        tracker = PlagiarismTrendTracker()
        tracker.add_snapshot(ScanSnapshot("2025-01-01T00:00:00", 10, 0.4, 0.7, 2, 0.59))
        assert tracker.snapshot_count == 1

    def test_load_snapshots(self):
        tracker = PlagiarismTrendTracker()
        snapshots = self._make_snapshots(5)
        tracker.load_snapshots(snapshots)
        assert tracker.snapshot_count == 5

    def test_analyze_empty(self):
        tracker = PlagiarismTrendTracker()
        result = tracker.analyze()
        assert result.snapshots_analyzed == 0
        assert len(result.metrics) == 0

    def test_analyze_basic(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(10))
        result = tracker.analyze()
        assert result.snapshots_analyzed == 10
        assert "avg_similarity" in result.metrics
        assert "max_similarity" in result.metrics

    def test_trend_direction_increasing(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(10))
        result = tracker.analyze()
        avg = result.metrics["avg_similarity"]
        assert avg.direction == TrendDirection.INCREASING

    def test_trend_direction_stable(self):
        tracker = PlagiarismTrendTracker()
        snapshots = [
            ScanSnapshot(f"2025-01-{i+1:02d}T00:00:00", 20, 0.50, 0.80, 3, 0.59)
            for i in range(10)
        ]
        tracker.load_snapshots(snapshots)
        result = tracker.analyze()
        avg = result.metrics["avg_similarity"]
        assert avg.direction == TrendDirection.STABLE

    def test_window_parameter(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(10))
        result = tracker.analyze(window=5)
        assert result.snapshots_analyzed == 5

    def test_alerts_generated_on_rising_trend(self):
        tracker = PlagiarismTrendTracker(trend_change_threshold=0.005)
        tracker.load_snapshots(self._make_snapshots(15))
        result = tracker.analyze()
        assert len(result.alerts) > 0

    def test_no_alerts_on_stable_data(self):
        tracker = PlagiarismTrendTracker()
        snapshots = [
            ScanSnapshot(f"2025-01-{i+1:02d}T00:00:00", 20, 0.50, 0.80, 3, 0.59)
            for i in range(5)
        ]
        tracker.load_snapshots(snapshots)
        result = tracker.analyze()
        assert len(result.critical_alerts) == 0

    def test_result_serialization(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(5))
        result = tracker.analyze()
        d = result.to_dict()
        assert "metrics" in d
        assert "alerts" in d
        assert "alert_summary" in d

    def test_compare_periods(self):
        tracker = PlagiarismTrendTracker()
        snapshots = self._make_snapshots(30)
        tracker.load_snapshots(snapshots)
        comparison = tracker.compare_periods(recent_days=7, baseline_days=14)
        assert "avg_similarity" in comparison
        assert "recent_mean" in comparison["avg_similarity"]

    def test_compare_periods_empty(self):
        tracker = PlagiarismTrendTracker()
        comparison = tracker.compare_periods(recent_days=7, baseline_days=14)
        assert comparison == {}

    def test_get_summary(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(5))
        summary = tracker.get_summary()
        assert summary["total_snapshots"] == 5
        assert "avg_similarity" in summary

    def test_get_summary_empty(self):
        tracker = PlagiarismTrendTracker()
        summary = tracker.get_summary()
        assert summary["total_snapshots"] == 0

    def test_get_snapshots_filtered(self):
        tracker = PlagiarismTrendTracker()
        tracker.load_snapshots(self._make_snapshots(10))
        filtered = tracker.get_snapshots(
            start_date="2025-01-05T00:00:00",
            end_date="2025-01-08T00:00:00",
        )
        assert len(filtered) == 4


# ---------------------------------------------------------------------------
# TrendAlert
# ---------------------------------------------------------------------------

class TestTrendAlert:
    def test_to_dict(self):
        alert = TrendAlert(
            alert_id="ALERT-0001",
            alert_severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test description",
            metric_name="avg_similarity",
            current_value=0.8,
            expected_value=0.5,
            deviation=0.3,
            detected_at="2025-01-01T00:00:00",
        )
        d = alert.to_dict()
        assert d["severity"] == "critical"
        assert d["deviation"] == 0.3


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestSnapshotsFromDicts:
    def test_basic(self):
        dicts = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "document_count": 10,
                "avg_similarity": 0.5,
                "max_similarity": 0.8,
                "flagged_count": 2,
                "threshold_used": 0.59,
            }
        ]
        result = snapshots_from_dicts(dicts)
        assert len(result) == 1
        assert result[0].document_count == 10

    def test_skips_malformed(self):
        dicts = [{"timestamp": "bad"}, {"document_count": 1}]
        result = snapshots_from_dicts(dicts)
        assert len(result) == 0


class TestCreateDemoSnapshots:
    def test_creates_snapshots(self):
        snapshots = create_demo_snapshots(10)
        assert len(snapshots) == 10
        assert all(isinstance(s, ScanSnapshot) for s in snapshots)

    def test_default_count(self):
        snapshots = create_demo_snapshots()
        assert len(snapshots) == 30
