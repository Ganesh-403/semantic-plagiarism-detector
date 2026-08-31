"""Tests for src.utils.trend_report_exporter."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.core.trend_tracker import (
    AlertSeverity,
    PlagiarismTrendTracker,
    ScanSnapshot,
    TrendAlert,
    TrendAnalysisResult,
    TrendDirection,
    TrendMetrics,
)
from src.utils.trend_report_exporter import (
    export_trend_html,
    export_trend_json,
    export_trend_markdown,
    get_chart_data,
)


def _make_result(n: int = 10) -> TrendAnalysisResult:
    """Create a TrendAnalysisResult for testing."""
    tracker = PlagiarismTrendTracker()
    snapshots = [
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
    tracker.load_snapshots(snapshots)
    return tracker.analyze()


# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------

class TestExportTrendJson:
    def test_export_creates_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.json")
            returned = export_trend_json(result, path)
            assert os.path.exists(returned)
            with open(returned) as f:
                data = json.load(f)
            assert "metrics" in data
            assert "alerts" in data

    def test_export_creates_parent_dirs(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subdir", "reports", "trend.json")
            export_trend_json(result, path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Export Markdown
# ---------------------------------------------------------------------------

class TestExportTrendMarkdown:
    def test_export_creates_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.md")
            export_trend_markdown(result, path)
            with open(path) as f:
                content = f.read()
            assert "Plagiarism Trend Analysis Report" in content
            assert "avg_similarity" in content

    def test_custom_title(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.md")
            export_trend_markdown(result, path, title="Custom Title")
            with open(path) as f:
                content = f.read()
            assert "Custom Title" in content

    def test_alerts_in_markdown(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.md")
            export_trend_markdown(result, path)
            with open(path) as f:
                content = f.read()
            if result.alerts:
                assert "Alerts" in content


# ---------------------------------------------------------------------------
# Export HTML
# ---------------------------------------------------------------------------

class TestExportTrendHtml:
    def test_export_creates_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.html")
            export_trend_html(result, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "avg_similarity" in content

    def test_html_contains_chart_data(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.html")
            export_trend_html(result, path)
            with open(path) as f:
                content = f.read()
            assert "chart-data" in content

    def test_html_with_no_alerts(self):
        tracker = PlagiarismTrendTracker()
        snapshots = [
            ScanSnapshot(f"2025-01-{i+1:02d}T00:00:00", 20, 0.50, 0.80, 3, 0.59)
            for i in range(5)
        ]
        tracker.load_snapshots(snapshots)
        result = tracker.analyze()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trend.html")
            export_trend_html(result, path)
            with open(path) as f:
                content = f.read()
            assert "No active alerts" in content


# ---------------------------------------------------------------------------
# get_chart_data
# ---------------------------------------------------------------------------

class TestGetChartData:
    def test_basic(self):
        result = _make_result()
        chart = get_chart_data(result)
        assert "avg_similarity" in chart
        assert "timestamps" in chart["avg_similarity"]
        assert "values" in chart["avg_similarity"]
        assert "moving_average" in chart["avg_similarity"]

    def test_empty_result(self):
        empty = TrendAnalysisResult(
            snapshots_analyzed=0,
            time_range_days=0,
            metrics={},
            alerts=[],
            moving_average_window=5,
        )
        chart = get_chart_data(empty)
        assert len(chart) == 0
