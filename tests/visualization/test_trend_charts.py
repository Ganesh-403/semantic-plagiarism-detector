"""Verify real trend-chart traces, metric units and static report exports."""

from datetime import datetime, timedelta

import pytest
from PIL import Image

from src.core.plagiarism_trends import (
    AnalyticsReport,
    OffenderProfile,
    SeverityDistribution,
    StatisticalSummary,
    TrendDirection,
    TrendResult,
    TrendWindow,
)
from src.visualization import trend_charts as charts


@pytest.fixture
def windows():
    start = datetime(2026, 1, 1)
    return [
        TrendWindow(
            start + timedelta(days=i),
            start + timedelta(days=i + 1),
            f"Day {i + 1}",
            count,
            0.5 + i / 10,
            0.8 + i / 20,
            SeverityDistribution(low=1, high=count - 1, total=count),
            unique_documents=i + 2,
        )
        for i, count in enumerate([2, 4, 6])
    ]


def test_timeline_forecast_dates_align_with_values_and_moving_average(windows):
    dates = [datetime(2026, 1, 4), datetime(2026, 1, 5)]
    fig = charts.create_incident_timeline(windows, [8.0, 10.0], dates)
    assert list(fig.data[0].y) == [2, 4, 6]
    assert list(fig.data[1].y) == [2, 3, 4]
    forecast = fig.data[2]
    assert list(forecast.x) == ["Day 3", "2026-01-04T00:00:00", "2026-01-05T00:00:00"]
    assert list(forecast.y) == [6, 8.0, 10.0]
    assert all(len(trace.x) == len(trace.y) for trace in fig.data)
    assert len(charts.create_incident_timeline(windows[:1]).data) == 1
    with pytest.raises(ValueError, match="timestamp"):
        charts.create_incident_timeline(windows, [8.0, 10.0], dates[:1])


def test_similarity_heatmap_and_severity_traces_preserve_source_data(windows):
    similarity = charts.create_similarity_trend_chart(windows)
    assert list(similarity.data[0].y) == [0.5, 0.6, 0.7]
    assert list(similarity.data[1].y) == pytest.approx([0.8, 0.85, 0.9])
    assert similarity.layout.shapes[0].y0 == 0.59
    heatmap = charts.create_window_comparison_heatmap(windows)
    assert list(heatmap.data[0].z[0]) == [2, 4, 6]
    assert list(heatmap.data[0].z[3]) == [2, 3, 4]
    severity = charts.create_severity_timeline(windows)
    assert [trace.name for trace in severity.data] == [
        "Critical",
        "High",
        "Medium",
        "Low",
    ]
    assert list(severity.data[1].y) == [1, 3, 5]
    assert all(trace.stackgroup == "one" for trace in severity.data)


def test_severity_and_offender_charts_use_counts_and_limit(windows):
    dist = SeverityDistribution(low=1, medium=0, high=3, critical=2, total=6)
    donut = charts.create_severity_donut(dist)
    assert list(donut.data[0].labels) == ["Low", "High", "Critical"]
    assert list(donut.data[0].values) == [1, 3, 2]
    offenders = [
        OffenderProfile("A" * 60, 4, 0.7, 0.9, windows[0].start, windows[-1].end),
        OffenderProfile("Second", 2, 0.5, 0.8, windows[0].start, windows[-1].end),
    ]
    bar = charts.create_offender_bar_chart(offenders, top_n=1)
    assert list(bar.data[0].y) == ["A" * 40]
    assert list(bar.data[0].x) == [4]
    assert not charts.create_offender_bar_chart(offenders, top_n=0).data
    assert not charts.create_severity_donut(SeverityDistribution(total=1)).data


@pytest.mark.parametrize(
    "function",
    [
        charts.create_incident_timeline,
        charts.create_similarity_trend_chart,
        charts.create_offender_bar_chart,
        charts.create_window_comparison_heatmap,
        charts.create_severity_timeline,
    ],
)
def test_empty_interactive_charts_explain_missing_data(function):
    fig = function([])
    assert not fig.data and len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text


def test_summary_metrics_preserve_percent_units_and_forecast(windows):
    trend = TrendResult(
        TrendDirection.INCREASING,
        2.0,
        0.0,
        0.9,
        0.01,
        99.0,
        [8.0],
        [datetime(2026, 1, 4)],
    )
    report = AnalyticsReport(
        datetime(2026, 1, 4),
        "daily",
        12,
        windows[0].start,
        windows[-1].end,
        StatisticalSummary(3, 0.6, 0.6, 0.1, 0.5, 0.7, 0.55, 0.65, 0.69, 0.1),
        SeverityDistribution(low=3, high=9, total=12),
        trend,
        windows,
        [],
        0.123,
        0.25,
    )
    values = charts.render_analytics_summary_metrics(report)
    assert values == {
        "total_incidents": 12,
        "avg_similarity": 0.6,
        "high_rate": 75.0,
        "repeat_offense_rate": 25.0,
        "monthly_growth": 12.3,
        "trend_direction": "increasing",
        "trend_confidence": 99.0,
        "date_range_start": "2026-01-01",
        "date_range_end": "2026-01-04",
        "forecast_next": 8.0,
    }
    card = charts.create_trend_summary_card(trend)
    assert card.data[0].value == 99.0 and list(card.data[0].gauge.axis.range) == [
        0,
        100,
    ]
    trend.forecast_values = []
    assert charts.render_analytics_summary_metrics(report)["forecast_next"] == 0


def test_optional_chart_libraries_fall_back_without_crashing(monkeypatch):
    monkeypatch.setattr(charts, "HAS_PLOTLY", False)
    for function in (
        charts.create_incident_timeline,
        charts.create_similarity_trend_chart,
        charts.create_offender_bar_chart,
        charts.create_window_comparison_heatmap,
        charts.create_severity_timeline,
    ):
        assert function([]) is None
    assert charts.create_severity_donut(SeverityDistribution()) is None
    assert charts.create_trend_summary_card(None) is None
    assert charts._empty_figure("No library") is None
    monkeypatch.setattr(charts, "HAS_MATPLOTLIB", False)
    assert charts.create_static_severity_pie(SeverityDistribution()) is None
    assert charts.create_static_incident_bar([]) is None


def test_static_exports_are_valid_pngs_and_release_figures(windows, tmp_path):
    before = charts.plt.get_fignums()
    for function, data, filename in [
        (charts.create_static_incident_bar, windows, "timeline.png"),
        (charts.create_static_incident_bar, windows[:1], "single.png"),
        (
            charts.create_static_severity_pie,
            SeverityDistribution(low=1, high=3, total=4),
            "severity.png",
        ),
    ]:
        target = str(tmp_path / filename)
        assert function(data, output_path=target, dpi=72) == target
        with Image.open(target) as image:
            assert image.format == "PNG" and min(image.size) > 200
            image.verify()
    assert charts.plt.get_fignums() == before
    assert charts.create_static_incident_bar([]) is None
    assert charts.create_static_severity_pie(SeverityDistribution()) is None
    assert charts.create_static_severity_pie(SeverityDistribution(total=1)) is None


@pytest.mark.parametrize("kind", ["bar", "pie"])
def test_failed_static_export_releases_figure(kind, windows, monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    def fail_save(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    before = charts.plt.get_fignums()
    with pytest.raises(OSError, match="Disk full"):
        if kind == "bar":
            charts.create_static_incident_bar(windows, str(tmp_path / "bad.png"))
        else:
            charts.create_static_severity_pie(
                SeverityDistribution(high=1, total=1), str(tmp_path / "bad.png")
            )
    assert charts.plt.get_fignums() == before


def test_moving_average_boundary_periods():
    assert charts._compute_moving_average([], 3) == []
    assert charts._compute_moving_average([2, 4, 6, 8], 2) == [2, 3, 5, 7]
    assert charts._compute_moving_average([2, 4], 1) == [2, 4]
    with pytest.raises(ValueError, match="positive"):
        charts._compute_moving_average([1], 0)
