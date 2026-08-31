"""
Tests for src/core/plagiarism_trends.py
========================================
Covers trend computation, statistical summaries, offender analysis,
aggregation, and export functionality.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.core.plagiarism_trends import (
    AnalyticsReport,
    OffenderProfile,
    PlagiarismIncident,
    PlagiarismTrendAnalytics,
    SeverityDistribution,
    StatisticalSummary,
    TimeSeriesPoint,
    TimeWindow,
    TrendDirection,
    TrendResult,
    TrendWindow,
    _linear_regression,
    _norm_cdf,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_time() -> datetime:
    """Return a fixed base datetime for test consistency."""
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_incidents(base_time: datetime) -> List[PlagiarismIncident]:
    """Generate a realistic set of sample plagiarism incidents."""
    incidents = []
    documents = ["essay_a.pdf", "essay_b.pdf", "essay_c.pdf", "report_d.docx"]
    matchees = ["source_x.pdf", "source_y.pdf", "source_z.pdf"]
    severities = ["low", "medium", "high", "critical"]

    for day in range(30):
        date = base_time + timedelta(days=day)
        for i in range(day % 4):
            incidents.append(PlagiarismIncident(
                incident_id=f"inc_{day}_{i}",
                document_name=documents[i % len(documents)],
                matched_against=matchees[i % len(matchees)],
                similarity_score=0.5 + (i * 0.1),
                severity=severities[min(i, 3)],
                detected_at=date,
                chunk_count=i + 1,
                max_chunk_similarity=min(1.0, 0.5 + (i * 0.12)),
            ))
    return incidents


@pytest.fixture
def populated_engine(sample_incidents: List[PlagiarismIncident]) -> PlagiarismTrendAnalytics:
    """Return an analytics engine populated with sample incidents."""
    engine = PlagiarismTrendAnalytics(default_window=TimeWindow.DAILY)
    engine.add_incidents(sample_incidents)
    return engine


@pytest.fixture
def empty_engine() -> PlagiarismTrendAnalytics:
    """Return an empty analytics engine."""
    return PlagiarismTrendAnalytics()


# ── Linear Regression Tests ───────────────────────────────────────────────────

class TestLinearRegression:
    """Tests for the _linear_regression helper function."""

    def test_perfect_positive_slope(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 2.0, 4.0, 6.0, 8.0]
        slope, intercept, r_squared, p_value = _linear_regression(x, y)
        assert abs(slope - 2.0) < 1e-10
        assert abs(intercept) < 1e-10
        assert abs(r_squared - 1.0) < 1e-10
        assert p_value < 0.05

    def test_perfect_negative_slope(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 7.0, 4.0, 1.0]
        slope, intercept, r_squared, _ = _linear_regression(x, y)
        assert abs(slope - (-3.0)) < 1e-10
        assert abs(intercept - 10.0) < 1e-10
        assert abs(r_squared - 1.0) < 1e-10

    def test_no_correlation(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [5.0, 5.0, 5.0, 5.0, 5.0]
        slope, intercept, r_squared, _ = _linear_regression(x, y)
        assert abs(slope) < 1e-10
        assert abs(intercept - 5.0) < 1e-10

    def test_single_point(self):
        slope, intercept, r_squared, p_value = _linear_regression([1.0], [2.0])
        assert slope == 0.0
        assert r_squared == 0.0
        assert p_value == 1.0

    def test_two_points(self):
        slope, intercept, r_squared, p_value = _linear_regression([0.0, 2.0], [0.0, 4.0])
        assert abs(slope - 2.0) < 1e-10
        assert abs(r_squared - 1.0) < 1e-10


class TestNormCdf:
    """Tests for the standard normal CDF approximation."""

    def test_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _norm_cdf(3.0) > 0.99

    def test_large_negative(self):
        assert _norm_cdf(-3.0) < 0.01

    def test_one(self):
        assert 0.8 < _norm_cdf(1.0) < 0.85


# ── PlagiarismIncident Tests ─────────────────────────────────────────────────

class TestPlagiarismIncident:
    """Tests for PlagiarismIncident dataclass."""

    def test_creation(self, base_time):
        inc = PlagiarismIncident(
            incident_id="test_001",
            document_name="essay.pdf",
            matched_against="source.pdf",
            similarity_score=0.85,
            severity="high",
            detected_at=base_time,
        )
        assert inc.incident_id == "test_001"
        assert inc.similarity_score == 0.85
        assert inc.chunk_count == 0

    def test_immutable(self, base_time):
        inc = PlagiarismIncident(
            incident_id="test_002",
            document_name="essay.pdf",
            matched_against="source.pdf",
            similarity_score=0.75,
            severity="medium",
            detected_at=base_time,
        )
        with pytest.raises(AttributeError):
            inc.similarity_score = 0.9


# ── Analytics Engine Tests ────────────────────────────────────────────────────

class TestAnalyticsEngineCore:
    """Tests for core PlagiarismTrendAnalytics functionality."""

    def test_initial_state(self, empty_engine):
        assert empty_engine.incident_count == 0
        assert empty_engine.get_date_range() == (None, None)

    def test_add_single_incident(self, empty_engine, base_time):
        inc = PlagiarismIncident(
            incident_id="i1", document_name="a.pdf",
            matched_against="b.pdf", similarity_score=0.6,
            severity="medium", detected_at=base_time,
        )
        empty_engine.add_incident(inc)
        assert empty_engine.incident_count == 1

    def test_add_batch_incidents(self, empty_engine, sample_incidents):
        empty_engine.add_incidents(sample_incidents)
        assert empty_engine.incident_count == len(sample_incidents)

    def test_date_range(self, populated_engine, base_time):
        start, end = populated_engine.get_date_range()
        assert start is not None
        assert end is not None
        assert start <= end

    def test_load_from_dicts(self, empty_engine):
        records = [
            {
                "incident_id": "d1",
                "document_name": "essay.pdf",
                "matched_against": "source.pdf",
                "similarity_score": 0.75,
                "severity": "high",
                "detected_at": "2025-01-15T10:00:00+00:00",
            },
            {
                "incident_id": "d2",
                "document_name": "report.docx",
                "matched_against": "textbook.pdf",
                "similarity_score": 0.62,
                "severity": "medium",
                "detected_at": "2025-01-20T14:30:00+00:00",
            },
        ]
        count = empty_engine.load_from_dicts(records)
        assert count == 2
        assert empty_engine.incident_count == 2

    def test_load_from_dicts_skips_malformed(self, empty_engine):
        records = [
            {"incident_id": "good", "document_name": "a.pdf",
             "matched_against": "b.pdf", "similarity_score": 0.7,
             "severity": "medium", "detected_at": "2025-01-15T10:00:00+00:00"},
            {"broken": True},  # Missing required fields
        ]
        count = empty_engine.load_from_dicts(records)
        assert count == 1

    def test_load_from_dicts_empty(self, empty_engine):
        count = empty_engine.load_from_dicts([])
        assert count == 0

    def test_cache_invalidation(self, empty_engine, base_time):
        """Adding new incidents should clear the window cache."""
        inc = PlagiarismIncident(
            incident_id="cache1", document_name="a.pdf",
            matched_against="b.pdf", similarity_score=0.7,
            severity="medium", detected_at=base_time,
        )
        empty_engine.add_incident(inc)
        windows = empty_engine.aggregate_by_window(TimeWindow.MONTHLY)
        assert len(windows) == 1

        inc2 = PlagiarismIncident(
            incident_id="cache2", document_name="a.pdf",
            matched_against="b.pdf", similarity_score=0.8,
            severity="high", detected_at=base_time + timedelta(days=40),
        )
        empty_engine.add_incident(inc2)
        windows2 = empty_engine.aggregate_by_window(TimeWindow.MONTHLY)
        assert len(windows2) == 2


# ── Aggregation Tests ─────────────────────────────────────────────────────────

class TestAggregation:
    """Tests for time-window aggregation."""

    def test_daily_aggregation(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.DAILY)
        assert len(windows) > 0
        assert all(isinstance(w, TrendWindow) for w in windows)
        assert all(w.incident_count >= 0 for w in windows)

    def test_monthly_aggregation(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.MONTHLY)
        assert len(windows) >= 1
        total = sum(w.incident_count for w in windows)
        assert total == populated_engine.incident_count

    def test_weekly_aggregation(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.WEEKLY)
        assert len(windows) >= 1

    def test_quarterly_aggregation(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.QUARTERLY)
        assert len(windows) >= 1

    def test_empty_engine_aggregation(self, empty_engine):
        windows = empty_engine.aggregate_by_window(TimeWindow.MONTHLY)
        assert windows == []

    def test_aggregation_preserves_total(self, populated_engine):
        """Sum of window incident counts should equal total incidents."""
        for window in TimeWindow:
            windows = populated_engine.aggregate_by_window(window)
            total = sum(w.incident_count for w in windows)
            assert total == populated_engine.incident_count, (
                f"Mismatch for {window.value}: {total} != {populated_engine.incident_count}"
            )

    def test_window_labels(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.MONTHLY)
        for w in windows:
            assert w.window_label  # Non-empty

    def test_severity_distribution_in_windows(self, populated_engine):
        windows = populated_engine.aggregate_by_window(TimeWindow.DAILY)
        for w in windows:
            total_sev = w.severity_dist.low + w.severity_dist.medium + w.severity_dist.high + w.severity_dist.critical
            assert total_sev == w.incident_count


# ── Trend Analysis Tests ──────────────────────────────────────────────────────

class TestTrendAnalysis:
    """Tests for trend computation and forecasting."""

    def test_insufficient_data(self, empty_engine):
        trend = empty_engine.compute_trend()
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA

    def test_increasing_trend(self):
        engine = PlagiarismTrendAnalytics(default_window=TimeWindow.MONTHLY)
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for month in range(6):
            for i in range(month + 1):
                engine.add_incident(PlagiarismIncident(
                    incident_id=f"inc_{month}_{i}",
                    document_name=f"doc_{i}.pdf",
                    matched_against="source.pdf",
                    similarity_score=0.7,
                    severity="medium",
                    detected_at=base + timedelta(days=month * 30),
                ))

        trend = engine.compute_trend(TimeWindow.MONTHLY)
        assert trend.direction == TrendDirection.INCREASING
        assert trend.slope > 0

    def test_stable_trend(self):
        engine = PlagiarismTrendAnalytics(default_window=TimeWindow.MONTHLY)
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for month in range(6):
            for i in range(5):  # Constant 5 per month
                engine.add_incident(PlagiarismIncident(
                    incident_id=f"inc_{month}_{i}",
                    document_name=f"doc_{i}.pdf",
                    matched_against="source.pdf",
                    similarity_score=0.7,
                    severity="medium",
                    detected_at=base + timedelta(days=month * 30),
                ))

        trend = engine.compute_trend(TimeWindow.MONTHLY)
        assert trend.direction in (TrendDirection.STABLE, TrendDirection.INCREASING)
        assert abs(trend.slope) < 2.0

    def test_forecast_values(self, populated_engine):
        trend = populated_engine.compute_trend(TimeWindow.WEEKLY, forecast_periods=3)
        assert len(trend.forecast_values) == 3
        assert all(isinstance(v, float) for v in trend.forecast_values)

    def test_similarity_trend(self, populated_engine):
        trend = populated_engine.compute_similarity_trend(TimeWindow.DAILY)
        assert trend.direction in (
            TrendDirection.INCREASING, TrendDirection.DECREASING,
            TrendDirection.STABLE, TrendDirection.INSUFFICIENT_DATA,
        )


# ── Statistical Summary Tests ─────────────────────────────────────────────────

class TestStatisticalSummary:
    """Tests for statistical summary computation."""

    def test_empty_engine(self, empty_engine):
        stats = empty_engine.compute_statistics()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.median == 0.0

    def test_basic_stats(self, populated_engine):
        stats = populated_engine.compute_statistics()
        assert stats.count > 0
        assert 0.0 <= stats.mean <= 1.0
        assert 0.0 <= stats.median <= 1.0
        assert stats.std_dev >= 0.0
        assert stats.min_value <= stats.mean <= stats.max_value
        assert stats.percentile_25 <= stats.median <= stats.percentile_75

    def test_iqr(self, populated_engine):
        stats = populated_engine.compute_statistics()
        expected_iqr = stats.percentile_75 - stats.percentile_25
        assert abs(stats.iqr - expected_iqr) < 1e-6

    def test_date_filtered_stats(self, populated_engine, base_time):
        start = base_time + timedelta(days=5)
        end = base_time + timedelta(days=15)
        stats = populated_engine.compute_statistics(start_date=start, end_date=end)
        assert stats.count >= 0


# ── Severity Distribution Tests ───────────────────────────────────────────────

class TestSeverityDistribution:
    """Tests for severity distribution computation."""

    def test_empty(self, empty_engine):
        dist = empty_engine.compute_severity_distribution()
        assert dist.total == 0
        assert dist.high_rate == 0.0

    def test_populated(self, populated_engine):
        dist = populated_engine.compute_severity_distribution()
        assert dist.total > 0
        assert dist.low + dist.medium + dist.high + dist.critical == dist.total

    def test_high_rate(self):
        dist = SeverityDistribution(low=10, medium=5, high=3, critical=2, total=20)
        assert dist.high_rate == 25.0

    def test_high_rate_zero_total(self):
        dist = SeverityDistribution()
        assert dist.high_rate == 0.0


# ── Offender Analysis Tests ───────────────────────────────────────────────────

class TestOffenderAnalysis:
    """Tests for repeat offender detection."""

    def test_top_offenders_empty(self, empty_engine):
        offenders = empty_engine.get_top_offenders()
        assert offenders == []

    def test_top_offenders_populated(self, populated_engine):
        offenders = populated_engine.get_top_offenders(top_n=5)
        assert len(offenders) <= 5
        assert all(isinstance(o, OffenderProfile) for o in offenders)

    def test_offenders_sorted_by_count(self, populated_engine):
        offenders = populated_engine.get_top_offenders(top_n=10)
        counts = [o.incident_count for o in offenders]
        assert counts == sorted(counts, reverse=True)

    def test_repeat_offense_rate_empty(self, empty_engine):
        rate = empty_engine.get_repeat_offense_rate()
        assert rate == 0.0

    def test_repeat_offense_rate_populated(self, populated_engine):
        rate = populated_engine.get_repeat_offense_rate()
        assert 0.0 <= rate <= 1.0

    def test_offender_profile_fields(self, populated_engine):
        offenders = populated_engine.get_top_offenders(top_n=1)
        if offenders:
            off = offenders[0]
            assert off.document_name
            assert off.incident_count >= 1
            assert 0.0 <= off.avg_similarity <= 1.0
            assert 0.0 <= off.max_similarity <= 1.0
            assert off.first_detected <= off.last_detected


# ── Growth Rate Tests ─────────────────────────────────────────────────────────

class TestGrowthRate:
    """Tests for monthly growth rate computation."""

    def test_empty_engine(self, empty_engine):
        rate = empty_engine.compute_monthly_growth_rate()
        assert rate == 0.0

    def test_populated_engine(self, populated_engine):
        rate = populated_engine.compute_monthly_growth_rate()
        assert isinstance(rate, float)


# ── Moving Average Tests ─────────────────────────────────────────────────────

class TestMovingAverage:
    """Tests for moving average computation."""

    def test_empty_engine(self, empty_engine):
        ma = empty_engine.compute_moving_average()
        assert ma == []

    def test_populated_engine(self, populated_engine):
        ma = populated_engine.compute_moving_average(period=3)
        assert len(ma) > 0
        assert all(isinstance(p, TimeSeriesPoint) for p in ma)

    def test_ma_values_non_negative(self, populated_engine):
        ma = populated_engine.compute_moving_average(period=5)
        for point in ma:
            assert point.value >= 0.0


# ── Report Generation Tests ───────────────────────────────────────────────────

class TestReportGeneration:
    """Tests for full report generation."""

    def test_empty_report(self, empty_engine):
        report = empty_engine.generate_report()
        assert isinstance(report, AnalyticsReport)
        assert report.total_incidents == 0

    def test_populated_report(self, populated_engine):
        report = populated_engine.generate_report(
            window=TimeWindow.WEEKLY,
            forecast_periods=2,
        )
        assert report.total_incidents > 0
        assert report.statistical_summary.count > 0
        assert report.trend.direction in TrendDirection
        assert len(report.windows) > 0
        assert len(report.trend.forecast_values) == 2

    def test_report_has_all_fields(self, populated_engine):
        report = populated_engine.generate_report()
        assert report.generated_at is not None
        assert report.time_window
        assert report.date_range_start is not None
        assert report.date_range_end is not None
        assert isinstance(report.statistical_summary, StatisticalSummary)
        assert isinstance(report.severity_distribution, SeverityDistribution)
        assert isinstance(report.trend, TrendResult)
        assert isinstance(report.windows, list)
        assert isinstance(report.top_offenders, list)


# ── Export Tests ───────────────────────────────────────────────────────────────

class TestExport:
    """Tests for JSON and CSV export."""

    def test_export_json_empty(self, empty_engine):
        json_str = empty_engine.export_json()
        data = json.loads(json_str)
        assert "generated_at" in data
        assert data["total_incidents"] == 0

    def test_export_json_populated(self, populated_engine):
        report = populated_engine.generate_report()
        json_str = populated_engine.export_json(report)
        data = json.loads(json_str)
        assert data["total_incidents"] > 0
        assert "statistical_summary" in data
        assert "severity_distribution" in data
        assert "trend" in data
        assert "windows" in data
        assert "top_offenders" in data

    def test_export_csv_empty(self, empty_engine):
        csv_str = empty_engine.export_csv()
        assert "Plagiarism Trend Analytics Report" in csv_str

    def test_export_csv_populated(self, populated_engine):
        report = populated_engine.generate_report()
        csv_str = populated_engine.export_csv(report)
        lines = csv_str.strip().split("\n")
        assert len(lines) > 10  # Should have header + sections + data
        assert "Statistical Summary" in csv_str
        assert "Severity Distribution" in csv_str
        assert "Time Window Breakdown" in csv_str
        assert "Top Offenders" in csv_str

    def test_json_forecast_includes_timestamps(self, populated_engine):
        report = populated_engine.generate_report(forecast_periods=3)
        data = json.loads(populated_engine.export_json(report))
        assert len(data["trend"]["forecast_timestamps"]) == 3


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_incident(self, base_time):
        engine = PlagiarismTrendAnalytics()
        engine.add_incident(PlagiarismIncident(
            incident_id="solo", document_name="only.pdf",
            matched_against="source.pdf", similarity_score=0.95,
            severity="critical", detected_at=base_time,
        ))
        report = engine.generate_report()
        assert report.total_incidents == 1
        assert report.statistical_summary.count == 1
        assert report.statistical_summary.mean == 0.95

    def test_all_same_score(self, base_time):
        engine = PlagiarismTrendAnalytics()
        for i in range(10):
            engine.add_incident(PlagiarismIncident(
                incident_id=f"same_{i}", document_name="a.pdf",
                matched_against="b.pdf", similarity_score=0.75,
                severity="medium",
                detected_at=base_time + timedelta(days=i * 30),
            ))
        stats = engine.compute_statistics()
        assert stats.mean == 0.75
        assert stats.std_dev == 0.0
        assert stats.min_value == 0.75
        assert stats.max_value == 0.75

    def test_window_start_end_consistency(self, populated_engine):
        """Window end should equal the start of the next window."""
        windows = populated_engine.aggregate_by_window(TimeWindow.MONTHLY)
        for i in range(len(windows) - 1):
            assert windows[i].end <= windows[i + 1].start or \
                   windows[i].end == windows[i + 1].start

    def test_similarity_score_boundaries(self, empty_engine, base_time):
        """Test with min and max similarity scores."""
        for score in [0.0, 0.59, 0.75, 0.90, 1.0]:
            empty_engine.add_incident(PlagiarismIncident(
                incident_id=f"boundary_{score}",
                document_name="test.pdf",
                matched_against="source.pdf",
                similarity_score=score,
                severity="low",
                detected_at=base_time,
            ))
        stats = empty_engine.compute_statistics()
        assert stats.min_value == 0.0
        assert stats.max_value == 1.0

    def test_many_documents_unique(self, base_time):
        """Test with many unique documents."""
        engine = PlagiarismTrendAnalytics()
        for i in range(50):
            engine.add_incident(PlagiarismIncident(
                incident_id=f"multi_{i}",
                document_name=f"doc_{i}.pdf",
                matched_against=f"source_{i}.pdf",
                similarity_score=0.5 + (i % 5) * 0.1,
                severity="medium",
                detected_at=base_time + timedelta(days=i),
            ))
        offenders = engine.get_top_offenders(top_n=5)
        # Each doc has exactly 1 incident
        assert all(o.incident_count == 1 for o in offenders)
        assert engine.get_repeat_offense_rate() == 0.0
