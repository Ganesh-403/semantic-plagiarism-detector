"""
Tests for src.core.report_export_analytics
-------------------------------------------
Covers severity classification, trend aggregation, risk profiling,
rolling averages, anomaly detection, threshold sensitivity, and
the full analytics summary pipeline.
"""

from __future__ import annotations

import pytest

from src.core.report_export_analytics import (
    AnalyticsSummary,
    DocumentRiskProfile,
    SeverityBucket,
    TrendPoint,
    classify_severity,
    compute_document_risk_profiles,
    compute_flagged_rate,
    compute_rolling_averages,
    compute_severity_distribution,
    compute_trend_deltas,
    detect_scan_anomalies,
    generate_analytics_summary,
    threshold_sensitivity_analysis,
)


@pytest.fixture
def sample_scan_history():
    return [
        {"timestamp": "2025-08-01T10:00:00", "document_count": 10,
         "avg_similarity": 0.45, "max_similarity": 0.82, "flagged_count": 2, "threshold_used": 0.59},
        {"timestamp": "2025-08-01T14:00:00", "document_count": 15,
         "avg_similarity": 0.51, "max_similarity": 0.91, "flagged_count": 4, "threshold_used": 0.59},
        {"timestamp": "2025-08-02T09:00:00", "document_count": 8,
         "avg_similarity": 0.38, "max_similarity": 0.72, "flagged_count": 1, "threshold_used": 0.55},
        {"timestamp": "2025-08-03T11:00:00", "document_count": 20,
         "avg_similarity": 0.62, "max_similarity": 0.95, "flagged_count": 6, "threshold_used": 0.59},
        {"timestamp": "2025-08-10T09:00:00", "document_count": 12,
         "avg_similarity": 0.41, "max_similarity": 0.78, "flagged_count": 3, "threshold_used": 0.60},
    ]


@pytest.fixture
def sample_incidents():
    return [
        {"incident_id": "i1", "document_a": "doc_a.pdf", "document_b": "doc_b.pdf",
         "similarity_score": 0.95, "date_flagged": "2025-08-01T10:00:00"},
        {"incident_id": "i2", "document_a": "doc_a.pdf", "document_b": "doc_c.pdf",
         "similarity_score": 0.82, "date_flagged": "2025-08-02T09:00:00"},
        {"incident_id": "i3", "document_a": "doc_d.pdf", "document_b": "doc_e.pdf",
         "similarity_score": 0.77, "date_flagged": "2025-08-03T11:00:00"},
        {"incident_id": "i4", "document_a": "doc_f.pdf", "document_b": "doc_g.pdf",
         "similarity_score": 0.61, "date_flagged": "2025-08-04T08:00:00"},
        {"incident_id": "i5", "document_a": "doc_a.pdf", "document_b": "doc_h.pdf",
         "similarity_score": 0.53, "date_flagged": "2025-08-05T12:00:00"},
    ]


class TestClassifySeverity:
    def test_high(self):
        assert classify_severity(0.95) == "High"
        assert classify_severity(0.90) == "High"

    def test_medium(self):
        assert classify_severity(0.75) == "Medium"
        assert classify_severity(0.82) == "Medium"
        assert classify_severity(0.89) == "Medium"

    def test_low(self):
        assert classify_severity(0.50) == "Low"
        assert classify_severity(0.0) == "Low"

    def test_clamping(self):
        assert classify_severity(1.5) == "High"
        assert classify_severity(-0.5) == "Low"

    def test_string_input(self):
        assert classify_severity("0.80") == "Medium"


class TestTrendAggregation:
    def test_daily_merges_same_day(self, sample_scan_history):
        summary = generate_analytics_summary(sample_scan_history, [], 10)
        aug_01 = [t for t in summary.daily_trends if t.timestamp == "2025-08-01"]
        assert len(aug_01) == 1
        assert aug_01[0].document_count == 25
        assert aug_01[0].flagged_count == 6

    def test_weekly(self, sample_scan_history):
        summary = generate_analytics_summary(sample_scan_history, [], 10)
        assert len(summary.weekly_trends) >= 2

    def test_empty(self):
        summary = generate_analytics_summary([], [], 0)
        assert summary.daily_trends == []


class TestSeverityDistribution:
    def test_basic(self, sample_incidents):
        dist = compute_severity_distribution(sample_incidents)
        labels = {b.label for b in dist}
        assert "High" in labels
        assert "Medium" in labels

    def test_all_high(self):
        dist = compute_severity_distribution([{"similarity_score": 0.95}, {"similarity_score": 0.98}])
        assert len(dist) == 1 and dist[0].label == "High"

    def test_empty(self):
        assert compute_severity_distribution([]) == []

    def test_invalid_scores(self):
        dist = compute_severity_distribution([{"similarity_score": "bad"}, {"similarity_score": 0.80}])
        assert sum(b.count for b in dist) == 1


class TestDocumentRiskProfiles:
    def test_doc_a_in_3_incidents(self, sample_incidents):
        profiles = compute_document_risk_profiles(sample_incidents, top_n=5)
        doc_a = [p for p in profiles if p.filename == "doc_a.pdf"]
        assert len(doc_a) == 1 and doc_a[0].incident_count == 3

    def test_sorted_desc(self, sample_incidents):
        profiles = compute_document_risk_profiles(sample_incidents, top_n=10)
        for i in range(len(profiles) - 1):
            assert profiles[i].incident_count >= profiles[i + 1].incident_count

    def test_empty(self):
        assert compute_document_risk_profiles([]) == []


class TestFlaggedRate:
    def test_all_flagged(self, sample_scan_history):
        assert compute_flagged_rate(sample_scan_history) == 1.0

    def test_empty(self):
        assert compute_flagged_rate([]) == 0.0

    def test_none_flagged(self):
        assert compute_flagged_rate([{"flagged_count": 0}, {"flagged_count": 0}]) == 0.0


class TestRollingAverages:
    def test_basic(self):
        points = [
            TrendPoint("2025-08-01", 10, 0.40, 0.80, 2, 0.59),
            TrendPoint("2025-08-02", 12, 0.45, 0.85, 3, 0.59),
            TrendPoint("2025-08-03", 8, 0.50, 0.90, 1, 0.59),
        ]
        rolling = compute_rolling_averages(points, window=3)
        assert len(rolling) == 3
        assert rolling[0]["rolling_avg_similarity"] == 0.40
        assert rolling[2]["rolling_avg_similarity"] == round((0.40 + 0.45 + 0.50) / 3, 4)

    def test_empty(self):
        assert compute_rolling_averages([]) == []


class TestAnomalyDetection:
    def test_detect_spike(self):
        points = [
            TrendPoint("d1", 10, 0.40, 0.80, 2, 0.59),
            TrendPoint("d2", 10, 0.40, 0.80, 2, 0.59),
            TrendPoint("d3", 10, 0.40, 0.80, 2, 0.59),
            TrendPoint("d4", 10, 0.40, 0.80, 15, 0.59),
            TrendPoint("d5", 10, 0.40, 0.80, 2, 0.59),
        ]
        anomalies = detect_scan_anomalies(points, z_threshold=1.5)
        assert len(anomalies) >= 1 and anomalies[0]["direction"] == "spike"

    def test_no_anomalies(self):
        points = [TrendPoint(f"d{i}", 10, 0.40, 0.80, 2, 0.59) for i in range(5)]
        assert detect_scan_anomalies(points) == []

    def test_insufficient_data(self):
        points = [TrendPoint("d1", 10, 0.40, 0.80, 2, 0.59)]
        assert detect_scan_anomalies(points) == []


class TestTrendDeltas:
    def test_positive(self):
        points = [
            TrendPoint("d1", 10, 0.40, 0.80, 2, 0.59),
            TrendPoint("d2", 10, 0.50, 0.80, 5, 0.59),
        ]
        d = compute_trend_deltas(points)
        assert d["avg_similarity_delta"] > 0 and d["flagged_count_delta"] > 0

    def test_empty(self):
        d = compute_trend_deltas([])
        assert d["avg_similarity_delta"] == 0.0


class TestThresholdSensitivity:
    def test_monotonic_decrease(self, sample_incidents):
        analysis = threshold_sensitivity_analysis(sample_incidents, [])
        counts = [e["incident_count"] for e in analysis]
        assert counts == sorted(counts, reverse=True)

    def test_empty(self):
        analysis = threshold_sensitivity_analysis([], [])
        assert len(analysis) == 14
        assert all(e["incident_count"] == 0 for e in analysis)


class TestFullPipeline:
    def test_summary_fields(self, sample_scan_history, sample_incidents):
        summary = generate_analytics_summary(sample_scan_history, sample_incidents, 10)
        assert isinstance(summary, AnalyticsSummary)
        assert summary.total_scans == 5
        assert summary.total_incidents == 5
        assert summary.total_documents == 10
        assert 0.0 <= summary.flagged_rate <= 1.0

    def test_to_dict(self, sample_scan_history, sample_incidents):
        summary = generate_analytics_summary(sample_scan_history, sample_incidents, 10)
        d = summary.to_dict()
        assert "total_scans" in d and "severity_distribution" in d and "daily_trends" in d

    def test_empty_data(self):
        summary = generate_analytics_summary([], [], 0)
        assert summary.total_scans == 0 and summary.flagged_rate == 0.0
