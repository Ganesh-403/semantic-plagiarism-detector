"""
tests/visualization/test_analytics.py
-------------------------------------
Unit tests for the analytics visualization functions.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.analytics import (
    plot_severity_donut_chart,
    plot_similarity_boxplot,
    plot_similarity_percentiles,
)


def test_plot_similarity_percentiles_calculation():
    """Verify the 25th, 50th, 75th, and 90th percentiles are plotted correctly."""
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig = plot_similarity_percentiles(scores)

    expected = np.percentile(scores, [25, 50, 75, 90])
    assert list(fig.data[0].x) == pytest.approx(list(expected))
    assert list(fig.data[0].y) == ["25th", "50th (Median)", "75th", "90th"]


def test_plot_similarity_percentiles_returns_figure():
    """Test that the function returns a Plotly Figure."""
    fig = plot_similarity_percentiles([0.4, 0.6, 0.8])
    assert isinstance(fig, go.Figure)


def test_plot_similarity_percentiles_empty_scores():
    """Test that an empty score list returns an empty chart with a message."""
    fig = plot_similarity_percentiles([])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


def test_plot_similarity_percentiles_skips_invalid_scores():
    """Test that non-numeric scores are ignored during percentile calculation."""
    scores = [0.2, "not-a-number", None, 0.8]
    fig = plot_similarity_percentiles(scores)

    expected = np.percentile([0.2, 0.8], [25, 50, 75, 90])
    assert list(fig.data[0].x) == pytest.approx(list(expected))


def test_plot_severity_donut_chart_returns_figure():
    incidents = [{"severity": "High"}, {"severity": "Medium"}]
    fig = plot_severity_donut_chart(incidents)
    assert isinstance(fig, go.Figure)


def test_plot_severity_donut_chart_counts_correct():
    incidents = [
        {"severity": "High"},
        {"severity": "High"},
        {"severity": "Medium"},
        {"severity": "Low"},
        {"severity": "Low"},
        {"severity": "Low"},
    ]
    fig = plot_severity_donut_chart(incidents)

    pie_trace = fig.data[0]
    labels = list(pie_trace.labels)
    values = list(pie_trace.values)

    assert "High" in labels
    assert values[labels.index("High")] == 2

    assert "Medium" in labels
    assert values[labels.index("Medium")] == 1

    assert "Low" in labels
    assert values[labels.index("Low")] == 3


def test_plot_severity_donut_chart_donut_hole():
    incidents = [{"severity": "High"}]
    fig = plot_severity_donut_chart(incidents)
    pie_trace = fig.data[0]
    assert pie_trace.hole == 0.4


def test_plot_severity_donut_chart_colors():
    incidents = [
        {"severity": "High"},
        {"severity": "Medium"},
        {"severity": "Low"},
    ]
    fig = plot_severity_donut_chart(incidents)
    pie_trace = fig.data[0]
    labels = list(pie_trace.labels)
    colors = pie_trace.marker.colors

    expected_colors = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#10b981",
    }

    for i, label in enumerate(labels):
        assert colors[i] == expected_colors[label]


def test_plot_severity_donut_chart_empty_input():
    # Empty input shouldn't crash
    fig = plot_severity_donut_chart([])
    assert isinstance(fig, go.Figure)
    # Check if there's an annotation for empty data
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "No plagiarism incidents recorded"


def test_plot_similarity_boxplot_returns_figure():
    """Test that the function returns a Plotly Figure."""
    incidents = [{"assignment_title": "Essay", "similarity_score": 0.8}]
    fig = plot_similarity_boxplot(incidents)
    assert isinstance(fig, go.Figure)


def test_plot_similarity_boxplot_groups_by_assignment_title():
    """Test that one box trace is created per assignment title."""
    incidents = [
        {"assignment_title": "Essay 1", "similarity_score": 0.8},
        {"assignment_title": "Essay 1", "similarity_score": 0.6},
        {"assignment_title": "Essay 2", "similarity_score": 0.3},
    ]
    fig = plot_similarity_boxplot(incidents)

    assert len(fig.data) == 2

    trace_by_name = {trace.name: list(trace.y) for trace in fig.data}
    assert trace_by_name["Essay 1"] == [0.8, 0.6]
    assert trace_by_name["Essay 2"] == [0.3]


def test_plot_similarity_boxplot_empty_incidents():
    """Test that an empty incident list returns an empty chart with a message."""
    fig = plot_similarity_boxplot([])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


def test_plot_similarity_boxplot_skips_missing_scores():
    """Test that incidents without a similarity score are skipped."""
    incidents = [
        {"assignment_title": "Essay 1", "similarity_score": 0.7},
        {"assignment_title": "Essay 1"},
        {"assignment_title": "Essay 1", "similarity_score": None},
        {"assignment_title": "Essay 1", "similarity_score": "not-a-number"},
    ]
    fig = plot_similarity_boxplot(incidents)

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.7]


def test_plot_similarity_boxplot_fallback_keys():
    """Test that 'title' and 'similarity' fallback keys are honoured."""
    incidents = [
        {"title": "Essay 1", "similarity": 0.9},
        {"title": "Essay 1", "similarity_score": 0.5},
    ]
    fig = plot_similarity_boxplot(incidents)

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.9, 0.5]

from src.visualization.analytics import plot_similarity_histogram


def test_plot_similarity_histogram_returns_figure():
    scores = [0.1, 0.2, 0.35, 0.5, 0.55, 0.9]
    fig = plot_similarity_histogram(scores, n_bins=10)

    assert isinstance(fig, go.Figure)
    bar_trace = fig.data[0]
    assert sum(bar_trace.y) == len(scores)


def test_plot_similarity_histogram_uses_color_gradient():
    scores = [0.1, 0.1, 0.1, 0.8]
    fig = plot_similarity_histogram(scores, n_bins=10)

    bar_trace = fig.data[0]
    assert list(bar_trace.marker.color) == list(bar_trace.y)
    assert bar_trace.marker.colorscale is not None


def test_plot_similarity_histogram_empty_scores():
    fig = plot_similarity_histogram([])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1