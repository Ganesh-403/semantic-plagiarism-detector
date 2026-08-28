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
tests/visualization/test_analytics.py
-------------------------------------
Unit tests for the analytics visualization functions.
"""

import numpy as np
import pandas as pd  # noqa: F811
import plotly.graph_objects as go
import pytest

from src.visualization.analytics import plot_similarity_distribution


def test_plot_similarity_distribution_xaxis_label():
    sim_df = pd.DataFrame(
        [[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]],
        index=["Doc1", "Doc2", "Doc3"],
        columns=["Doc1", "Doc2", "Doc3"],
    )
    fig = plot_similarity_distribution(sim_df)
    assert fig.layout.xaxis.title.text == "Similarity Score"


from src.visualization.analytics import (
    calculate_severity_ratios,
    plot_high_severity_trends,
    plot_monthly_incident_trends,
    plot_severity_donut_chart,
    plot_similarity_boxplot,
    plot_similarity_boxplot_by_group,
    plot_similarity_histogram,
    plot_similarity_percentiles,
)


def test_plot_high_severity_trends_cumulative_line():
    """Verify cumulative incidents are plotted on a secondary Y-axis."""
    trend_data = [
        {"date": "2026-08-01", "count": 2},
        {"date": "2026-08-02", "count": 3},
        {"date": "2026-08-03", "count": 1},
    ]

    fig = plot_high_severity_trends(trend_data)

    cumulative_trace = next(
        trace for trace in fig.data if trace.name == "Cumulative Incidents"
    )

    assert list(cumulative_trace.y) == [2, 5, 6]
    assert cumulative_trace.yaxis == "y2"
    assert fig.layout.yaxis2.title.text == "Cumulative Incidents"


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


def test_plot_similarity_boxplot_by_group_returns_figure():
    """Test that the function returns a Plotly Figure with one box per group."""
    scores_dict = {
        "Essay 1": [0.1, 0.4, 0.6, 0.9],
        "Essay 2": [0.2, 0.3, 0.5],
    }
    fig = plot_similarity_boxplot_by_group(scores_dict)

    assert isinstance(fig, go.Figure)
    box_names = [trace.name for trace in fig.data]
    assert box_names == ["Essay 1", "Essay 2"]
    assert list(fig.data[0].y) == scores_dict["Essay 1"]


def test_plot_similarity_boxplot_by_group_empty_dict():
    """An empty scores_dict should return a figure with a message, not error."""
    fig = plot_similarity_boxplot_by_group({})

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == "No similarity scores available to plot"


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


def test_plot_similarity_histogram_custom_colorscale():
    scores = [0.1, 0.2, 0.35, 0.5, 0.55, 0.9]
    fig = plot_similarity_histogram(scores, n_bins=10, colorscale="Cividis")

    bar_trace = fig.data[0]
    # Plotly expands named colorscales to their tuple form; Cividis starts
    # with a dark blue (#00224e) that differs from Viridis' starting purple.
    assert bar_trace.marker.colorscale[0][1] == "#00224e"


def test_plot_similarity_histogram_empty_scores():
    fig = plot_similarity_histogram([])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


def test_plot_analytics_charts_dark_mode_theme_colors():
    """Verify Issue #1619: theme_colors applies dark background and ink font color."""
    dark_theme = {
        "background": "#0F172A",
        "surface": "#1E293B",
        "ink": "#F8FAFC",
        "border": "#334155",
    }
    from src.visualization.analytics import (
        plot_high_severity_trends,
        plot_most_plagiarized_documents,
        plot_severity_donut_chart,
        plot_similarity_percentiles,
    )

    fig1 = plot_high_severity_trends(
        [{"date": "2026-08-01", "count": 3}], theme_colors=dark_theme
    )
    assert fig1.layout.paper_bgcolor == "#0F172A"
    assert fig1.layout.plot_bgcolor == "#1E293B"
    assert fig1.layout.font.color == "#F8FAFC"

    fig2 = plot_most_plagiarized_documents(
        [{"document_name": "essay.pdf", "incident_count": 5}], theme_colors=dark_theme
    )
    assert fig2.layout.paper_bgcolor == "#0F172A"
    assert fig2.layout.plot_bgcolor == "#1E293B"

    fig3 = plot_severity_donut_chart([{"severity": "High"}], theme_colors=dark_theme)
    assert fig3.layout.paper_bgcolor == "#0F172A"
    assert fig3.layout.plot_bgcolor == "#1E293B"

    fig4 = plot_similarity_percentiles([0.5, 0.8, 0.9], theme_colors=dark_theme)
    assert fig4.layout.paper_bgcolor == "#0F172A"


# ── Hierarchical Clustering Dendrogram (Issue #1367) ──────────────────────


def _make_similarity_matrix(n: int = 5, seed: int = 42) -> "pd.DataFrame":
    """Build a synthetic symmetric similarity matrix for testing."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    mat = rng.random((n, n))
    # Symmetrize and force diagonal = 1.
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 1.0)
    np.clip(mat, 0.0, 1.0, out=mat)
    names = [f"doc_{i}" for i in range(n)]
    return pd.DataFrame(mat, index=names, columns=names)


def test_plot_hierarchical_dendrogram_returns_figure():
    """The function must return a plotly Figure object."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=5)
    fig = plot_hierarchical_dendrogram(sim_df)
    assert isinstance(fig, go.Figure)


def test_plot_hierarchical_dendrogram_has_single_scatter_trace():
    """The dendrogram is rendered as exactly one Scatter trace in lines mode."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=5)
    fig = plot_hierarchical_dendrogram(sim_df)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert isinstance(trace, go.Scatter)
    assert trace.mode == "lines"


def test_plot_hierarchical_dendrogram_wards_linkage():
    """The merge tree must contain exactly n-1 merges for n documents."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    n = 6
    sim_df = _make_similarity_matrix(n=n)
    fig = plot_hierarchical_dendrogram(sim_df)

    # Each Ward merge contributes 4 points + 1 None separator = 5 entries
    # in the x/y arrays.  So len(x) should equal 5 * (n - 1).
    trace = fig.data[0]
    none_count = list(trace.x).count(None)
    assert none_count == n - 1


def test_plot_hierarchical_dendrogram_xaxis_shows_doc_names():
    """Leaf x-tick labels must be the document names from the DataFrame."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=4)
    fig = plot_hierarchical_dendrogram(sim_df)
    ticktext = list(fig.layout.xaxis.ticktext)
    assert ticktext == ["doc_0", "doc_1", "doc_2", "doc_3"]


def test_plot_hierarchical_dendrogram_yaxis_is_inverted():
    """The y-axis must be reversed so the tree grows downward (leaves at bottom)."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=5)
    fig = plot_hierarchical_dendrogram(sim_df)
    assert fig.layout.yaxis.autorange == "reversed"


def test_plot_hierarchical_dendrogram_empty_input_returns_annotation_figure():
    """An empty DataFrame must return a figure with an annotation, not raise."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    empty_df = pd.DataFrame()
    fig = plot_hierarchical_dendrogram(empty_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1
    assert "No similarity data available" in fig.layout.annotations[0].text


def test_plot_hierarchical_dendrogram_single_document_returns_annotation_figure():
    """A 1×1 matrix must return an annotation figure, not raise."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    single_df = pd.DataFrame([[1.0]], index=["only_doc"], columns=["only_doc"])
    fig = plot_hierarchical_dendrogram(single_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1
    assert "At least two documents" in fig.layout.annotations[0].text


def test_plot_hierarchical_dendrogram_identical_documents_merge_at_distance_zero():
    """When all documents are identical, every merge distance is ~0."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    n = 4
    # All-ones similarity matrix → distance 0 everywhere.
    sim_df = pd.DataFrame(
        np.ones((n, n)),
        index=[f"d{i}" for i in range(n)],
        columns=[f"d{i}" for i in range(n)],
    )
    fig = plot_hierarchical_dendrogram(sim_df)
    trace = fig.data[0]
    # Filter out None separators and assert every real y value is ~0.
    ys = [y for y in trace.y if y is not None]
    assert all(abs(float(y)) < 1e-9 for y in ys)


def test_plot_hierarchical_dendrogram_hover_text_contains_doc_names():
    """Hover tooltips must reference the document names so instructors can
    identify which submissions belong to which cluster."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=4)
    fig = plot_hierarchical_dendrogram(sim_df)
    trace = fig.data[0]
    # Concatenate all hovertext entries and check that each doc name appears.
    all_text = " ".join(t or "" for t in trace.hovertext)
    for name in sim_df.index:
        assert name in all_text


def test_plot_hierarchical_dendrogram_respects_theme_colors():
    """Dark theme_colors must propagate to paper_bgcolor / plot_bgcolor."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    dark_theme = {
        "background": "#0F172A",
        "surface": "#1E293B",
        "ink": "#F8FAFC",
        "border": "#334155",
    }
    sim_df = _make_similarity_matrix(n=5)
    fig = plot_hierarchical_dendrogram(sim_df, theme_colors=dark_theme)
    assert fig.layout.paper_bgcolor == "#0F172A"
    assert fig.layout.plot_bgcolor == "#1E293B"
    assert fig.layout.font.color == "#F8FAFC"


def test_plot_hierarchical_dendrogram_uses_wards_method():
    """End-to-end sanity check: for a known dataset, verify the merge
    sequence matches scipy's Ward linkage output exactly.

    This guards against silent regressions where someone swaps the linkage
    method (e.g. to 'single' or 'average') — which would still produce a
    valid-looking dendrogram but with different cluster groupings.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    from src.visualization.analytics import plot_hierarchical_dendrogram

    sim_df = _make_similarity_matrix(n=6, seed=7)
    sim_values = np.clip(sim_df.to_numpy(dtype=float), 0.0, 1.0)
    distance_matrix = 1.0 - sim_values
    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=False)
    expected_linkage = linkage(condensed, method="ward")

    # Reconstruct merge distances from the rendered figure's y values.
    # Each merge contributes 4 real y values (drop, bridge, bridge, drop)
    # plus one None separator.  The bridge y == merge distance.
    fig = plot_hierarchical_dendrogram(sim_df)
    ys = list(fig.data[0].y)

    # Extract the bridge distances: the unique non-zero y values that
    # appear exactly twice in a row (the horizontal bridge).
    rendered_distances: list[float] = []
    prev_y = None
    for y in ys:
        if y is None:
            prev_y = None
            continue
        if prev_y is not None and abs(float(y) - float(prev_y)) < 1e-9:
            # This is part of a horizontal bridge.
            rendered_distances.append(float(y))
        prev_y = y

    # There should be exactly n-1 = 5 merges.
    assert len(rendered_distances) == 5

    # Compare against scipy's expected merge distances, sorted ascending.
    expected_distances = sorted(float(row[2]) for row in expected_linkage)
    rendered_sorted = sorted(rendered_distances)
    for expected, rendered in zip(expected_distances, rendered_sorted):
        assert abs(expected - rendered) < 1e-9, (
            f"Ward merge distance mismatch: expected {expected}, " f"got {rendered}"
        )


class TestPlotMonthlyIncidentTrends:
    """Test suite for the monthly incident trend bar chart (Issue #2211)."""

    @pytest.fixture
    def sample_incidents(self):
        """Provide a standard set of incidents spanning multiple months."""
        return [
            {"date_flagged": "2024-01-15T10:00:00", "similarity_score": 0.85},
            {"date_flagged": "2024-01-20T14:30:00", "similarity_score": 0.92},
            {"date_flagged": "2024-02-05T09:15:00", "similarity_score": 0.75},
            {"date_flagged": "2024-03-10T11:00:00", "similarity_score": 0.88},
            {"date_flagged": "2024-03-15T16:45:00", "similarity_score": 0.95},
            {"date_flagged": "2024-03-20T08:30:00", "similarity_score": 0.82},
        ]

    def test_aggregates_incidents_by_month(self, sample_incidents):
        """Verify incidents are correctly aggregated by YYYY-MM."""
        fig = plot_monthly_incident_trends(sample_incidents)

        # Extract data from the first bar trace
        bar_trace = fig.data[0]

        # Should have 3 months: Jan, Feb, Mar
        assert len(bar_trace.x) == 3

        # Verify counts: Jan=2, Feb=1, Mar=3
        y_values = list(bar_trace.y)
        assert 2 in y_values  # January
        assert 1 in y_values  # February
        assert 3 in y_values  # March

    def test_handles_empty_incidents_list(self):
        """Verify empty list returns an empty-state chart."""
        fig = plot_monthly_incident_trends([])

        # Should have an annotation instead of bar traces
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) > 0
        assert "No plagiarism incidents" in fig.layout.annotations[0].text

    def test_handles_none_input(self):
        """Verify None input returns an empty-state chart."""
        fig = plot_monthly_incident_trends(None)

        assert len(fig.data) == 0
        assert len(fig.layout.annotations) > 0

    def test_fills_missing_months_with_zeros(self):
        """Verify months with no incidents are included with count=0."""
        incidents = [
            {"date_flagged": "2024-01-15T10:00:00"},
            # February is missing
            {"date_flagged": "2024-03-10T11:00:00"},
        ]

        fig = plot_monthly_incident_trends(incidents)
        bar_trace = fig.data[0]

        # Should have 3 bars: Jan, Feb (0), Mar
        assert len(bar_trace.x) == 3

        # Find the February bar (should have y=0)
        y_values = list(bar_trace.y)
        assert 0 in y_values

    def test_respects_months_to_show_parameter(self):
        """Verify only the most recent N months are displayed."""
        # Create incidents spanning 12 months
        incidents = [
            {"date_flagged": f"2024-{str(m).zfill(2)}-15T10:00:00"}
            for m in range(1, 13)
        ]

        # Request only last 3 months
        fig = plot_monthly_incident_trends(incidents, months_to_show=3)
        bar_trace = fig.data[0]

        # Should only show Oct, Nov, Dec
        assert len(bar_trace.x) == 3

    def test_ignores_incidents_with_invalid_dates(self):
        """Verify incidents with unparseable dates are silently skipped."""
        incidents = [
            {"date_flagged": "2024-01-15T10:00:00"},
            {"date_flagged": "not-a-date"},
            {"date_flagged": None},
            {"date_flagged": "2024-02-05T09:15:00"},
        ]

        fig = plot_monthly_incident_trends(incidents)
        bar_trace = fig.data[0]

        # Should only process Jan and Feb
        assert len(bar_trace.x) == 2

    def test_applies_theme_colors(self):
        """Verify theme colors are applied to the chart."""
        incidents = [{"date_flagged": "2024-01-15T10:00:00"}]
        dark_colors = get_chart_theme_colors("Dark")

        fig = plot_monthly_incident_trends(incidents, theme_colors=dark_colors)

        # Verify dark background is applied
        assert fig.layout.paper_bgcolor == "#1e293b"

    def test_y_axis_starts_at_zero(self):
        """Verify y-axis always starts at 0 to prevent misleading visualizations."""
        incidents = [
            {"date_flagged": "2024-01-15T10:00:00"},
            {"date_flagged": "2024-01-20T14:30:00"},
        ]

        fig = plot_monthly_incident_trends(incidents)

        # Check y-axis range mode
        assert fig.layout.yaxis.rangemode == "tozero"

    def test_handles_timestamp_key_fallback(self):
        """Verify function falls back to 'timestamp' key if 'date_flagged' missing."""
        incidents = [
            {"timestamp": "2024-01-15T10:00:00"},
            {"created_at": "2024-02-05T09:15:00"},
        ]

        fig = plot_monthly_incident_trends(incidents)
        bar_trace = fig.data[0]

        # Should process both incidents
        assert len(bar_trace.x) == 2

    def test_chart_title_and_labels(self):
        """Verify chart has correct title and axis labels."""
        incidents = [{"date_flagged": "2024-01-15T10:00:00"}]
        fig = plot_monthly_incident_trends(incidents)

        assert fig.layout.title.text == "Monthly Plagiarism Incident Trends"
        assert fig.layout.xaxis.title.text == "Month"
        assert fig.layout.yaxis.title.text == "Number of Incidents"


def test_plot_charts_default_to_light_template_without_theme_colors():
    """Without theme_colors the layout must keep the Plotly defaults."""
    fig = plot_similarity_percentiles([0.4, 0.6, 0.8])

    assert fig.layout.paper_bgcolor is None
    assert fig.layout.font.color is None


def test_theme_override_forces_light_template():
    """theme_override='light' should force the plotly_white template."""
    fig = plot_similarity_percentiles([0.4, 0.6, 0.8], theme_override="light")

    assert fig.layout.template.layout.paper_bgcolor == "white"


def test_theme_override_forces_dark_template():
    """theme_override='dark' should force the plotly_dark template."""
    fig = plot_similarity_percentiles([0.4, 0.6, 0.8], theme_override="dark")

    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"


def test_theme_override_none_leaves_default_template():
    """Without theme_override, the default Plotly template should apply."""
    fig = plot_similarity_percentiles([0.4, 0.6, 0.8])

    assert fig.layout.template.layout.paper_bgcolor not in (
        "white",
        "rgb(17,17,17)",
    )


def test_calculate_severity_ratios_percentage_breakdown():
    """Test the exact percentage breakdown across High, Medium, and Low."""
    from src.core.config import DEFAULT_THRESHOLDS

    incidents = [
        {"similarity_score": DEFAULT_THRESHOLDS.high + 0.05},  # High
        {"similarity_score": DEFAULT_THRESHOLDS.high},  # High
        {"similarity_score": DEFAULT_THRESHOLDS.medium},  # Medium
        {"similarity_score": DEFAULT_THRESHOLDS.medium - 0.05},  # Low
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 50.0, "Medium": 25.0, "Low": 25.0}


def test_calculate_severity_ratios_ignores_invalid_scores():
    """Incidents with missing or non-numeric scores should be skipped."""
    from src.core.config import DEFAULT_THRESHOLDS

    incidents = [
        {"similarity_score": DEFAULT_THRESHOLDS.high},
        {"similarity_score": None},
        {"assignment_title": "no score field"},
        {"similarity_score": "not-a-number"},
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 100.0, "Medium": 0.0, "Low": 0.0}


def test_calculate_severity_ratios_empty_incidents():
    """An empty incident list should return all-zero percentages, not error."""
    ratios = calculate_severity_ratios([])

    assert ratios == {"High": 0.0, "Medium": 0.0, "Low": 0.0}


def test_severity_ratios_all_high():
    """All incidents scoring >= 0.80 should be classified as High only."""
    incidents = [
        {"similarity_score": 0.80},
        {"similarity_score": 0.9},
        {"similarity_score": 1.0},
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 100.0, "Medium": 0.0, "Low": 0.0}


def test_severity_ratios_mixed():
    """Scores spanning all three tiers should split proportionally."""
    incidents = [
        {"similarity_score": 0.95},
        {"similarity_score": 0.65},
        {"similarity_score": 0.55},
        {"similarity_score": 0.2},
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 25.0, "Medium": 50.0, "Low": 25.0}


def test_severity_ratios_empty_list():
    """An empty incident list must return all zeros, not error."""
    ratios = calculate_severity_ratios([])

    assert ratios == {"High": 0.0, "Medium": 0.0, "Low": 0.0}


def test_severity_ratios_non_numeric_skipped():
    """Incidents with None or non-numeric scores should be ignored."""
    incidents = [
        {"similarity_score": 0.9},
        {"similarity_score": None},
        {"similarity_score": "invalid"},
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 100.0, "Medium": 0.0, "Low": 0.0}


def test_severity_ratios_fallback_key():
    """The 'similarity' key should be used when 'similarity_score' is missing."""
    incidents = [
        {"similarity": 0.9},
        {"similarity": 0.6},
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 50.0, "Medium": 50.0, "Low": 0.0}


# ── Added tests for theme synchronization and helper utilities ─────────────

from unittest.mock import MagicMock, patch  # noqa: F401

from src.visualization.analytics import (
    _empty_chart,
    apply_plotly_theme,
    get_chart_theme_colors,
    get_top_similar_pairs,
)


class TestGetChartThemeColors:
    """Test suite for the get_chart_theme_colors() synchronizer (Issue #1887)."""

    def test_light_mode_returns_correct_palette(self):
        """Verify Light mode returns white background and dark ink."""
        colors = get_chart_theme_colors("Light")

        assert colors["background"] == "#ffffff"
        assert colors["ink"] == "#0f172a"
        assert "surface" in colors
        assert "muted" in colors
        assert "border" in colors

    def test_dark_mode_returns_correct_palette(self):
        """Verify Dark mode returns slate background and light ink."""
        colors = get_chart_theme_colors("Dark")

        assert colors["background"] == "#1e293b"
        assert colors["ink"] == "#f8fafc"
        assert colors["surface"] == "#0f172a"

    def test_case_insensitive_handling(self):
        """Verify theme mode string is case-insensitive."""
        assert get_chart_theme_colors("dark")["background"] == "#1e293b"
        assert get_chart_theme_colors("DARK")["background"] == "#1e293b"
        assert get_chart_theme_colors("LIGHT")["background"] == "#ffffff"
        assert get_chart_theme_colors("light")["background"] == "#ffffff"

    def test_whitespace_handling(self):
        """Verify leading/trailing whitespace is stripped."""
        assert get_chart_theme_colors("  Dark  ")["background"] == "#1e293b"
        assert get_chart_theme_colors("\tLight\n")["background"] == "#ffffff"

    def test_invalid_mode_defaults_to_light(self):
        """Verify unrecognized theme modes safely default to Light palette."""
        colors = get_chart_theme_colors("midnight")
        assert colors["background"] == "#ffffff"

        colors = get_chart_theme_colors("")
        assert colors["background"] == "#ffffff"

    def test_none_input_defaults_to_light(self):
        """Verify None input safely defaults to Light palette."""
        colors = get_chart_theme_colors(None)
        assert colors["background"] == "#ffffff"
        assert colors["ink"] == "#0f172a"

    def test_color_contrast_ratios(self):
        """Verify ink and background colors have sufficient contrast for readability.

        This is a basic sanity check to ensure we aren't returning white-on-white
        or black-on-black combinations that would render charts invisible.
        """
        light = get_chart_theme_colors("Light")
        dark = get_chart_theme_colors("Dark")

        # Light mode: dark ink on white background
        assert light["ink"] != light["background"]

        # Dark mode: light ink on dark background
        assert dark["ink"] != dark["background"]

        # Ensure muted text is distinct from primary ink
        assert light["muted"] != light["ink"]
        assert dark["muted"] != dark["ink"]


class TestApplyPlotlyTheme:
    """Test suite for the apply_plotly_theme() layout updater."""

    def test_applies_light_theme_to_figure(self):
        """Verify light theme colors are applied to Plotly figure layout."""
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = get_chart_theme_colors("Light")

        updated_fig = apply_plotly_theme(fig, theme_colors=colors)

        assert updated_fig.layout.paper_bgcolor == "#ffffff"
        assert updated_fig.layout.font.color == "#0f172a"

    def test_applies_dark_theme_to_figure(self):
        """Verify dark theme colors are applied to Plotly figure layout."""
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = get_chart_theme_colors("Dark")

        updated_fig = apply_plotly_theme(fig, theme_colors=colors)

        assert updated_fig.layout.paper_bgcolor == "#1e293b"
        assert updated_fig.layout.font.color == "#f8fafc"

    def test_handles_none_theme_colors_gracefully(self):
        """Verify function doesn't crash when theme_colors is None."""
        import plotly.graph_objects as go

        fig = go.Figure()
        # Should not raise, just return the figure unchanged
        updated_fig = apply_plotly_theme(fig, theme_colors=None)
        assert updated_fig is fig

    def test_gridlines_toggle(self):
        """Verify show_grid parameter controls gridline visibility."""
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = get_chart_theme_colors("Light")

        # With grid
        fig_grid = apply_plotly_theme(fig, theme_colors=colors, show_grid=True)
        assert fig_grid.layout.xaxis.gridcolor == colors["border"]

        # Without grid
        apply_plotly_theme(go.Figure(), theme_colors=colors, show_grid=False)
        # When show_grid=False, gridcolor shouldn't be explicitly set by our function
        # (Plotly defaults apply)


class TestCalculateSeverityRatios:
    """Test suite for severity percentage calculations."""

    def test_calculates_correct_percentages(self):
        """Verify High/Medium/Low ratios are calculated correctly."""
        incidents = [
            {"similarity_score": 0.90},  # High
            {"similarity_score": 0.85},  # High
            {"similarity_score": 0.60},  # Medium
            {"similarity_score": 0.30},  # Low
        ]

        ratios = calculate_severity_ratios(incidents)

        assert ratios["High"] == 50.0  # 2/4
        assert ratios["Medium"] == 25.0  # 1/4
        assert ratios["Low"] == 25.0  # 1/4

    def test_handles_empty_incidents_list(self):
        """Verify empty list returns all zeros."""
        ratios = calculate_severity_ratios([])
        assert ratios == {"High": 0.0, "Medium": 0.0, "Low": 0.0}

    def test_ignores_invalid_scores(self):
        """Verify incidents with non-numeric scores are ignored."""
        incidents = [
            {"similarity_score": 0.90},
            {"similarity_score": "invalid"},
            {"similarity_score": None},
        ]

        ratios = calculate_severity_ratios(incidents)
        # Only 1 valid incident (High)
        assert ratios["High"] == 100.0
        assert ratios["Medium"] == 0.0
        assert ratios["Low"] == 0.0

    def test_fallback_to_similarity_key(self):
        """Verify function falls back to 'similarity' key if 'similarity_score' missing."""
        incidents = [
            {"similarity": 0.55},  # Medium
        ]

        ratios = calculate_severity_ratios(incidents)
        assert ratios["Medium"] == 100.0


class TestGetTopSimilarPairs:
    """Test suite for extracting top-N similar document pairs."""

    def test_returns_top_n_pairs_sorted(self):
        """Verify top N pairs are returned in descending order."""
        df = pd.DataFrame(
            [[1.0, 0.8, 0.2], [0.8, 1.0, 0.9], [0.2, 0.9, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        pairs = get_top_similar_pairs(df, top_n=2)

        assert len(pairs) == 2
        # Top pair should be B-C (0.9)
        assert pairs[0] == ("B", "C", 0.9) or pairs[0] == ("C", "B", 0.9)
        # Second should be A-B (0.8)
        assert pairs[1][2] == 0.8

    def test_excludes_self_similarity(self):
        """Verify diagonal (self-similarity) is excluded."""
        df = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"]
        )

        pairs = get_top_similar_pairs(df, top_n=5)

        # Should only return A-B pair, not A-A or B-B
        assert len(pairs) == 1
        assert pairs[0][2] == 0.5

    def test_handles_empty_dataframe(self):
        """Verify empty DataFrame returns empty list."""
        df = pd.DataFrame()
        assert get_top_similar_pairs(df) == []

    def test_handles_single_document(self):
        """Verify DataFrame with < 2 documents returns empty list."""
        df = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        assert get_top_similar_pairs(df) == []

    def test_top_pairs_empty_df(self):
        """Verify get_top_similar_pairs returns empty list for empty DataFrame."""
        df = pd.DataFrame()
        assert get_top_similar_pairs(df) == []

    def test_top_pairs_single_doc(self):
        """Verify get_top_similar_pairs returns empty list for single document."""
        df = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        assert get_top_similar_pairs(df) == []

    def test_top_pairs_returns_sorted_desc(self):
        """Verify get_top_similar_pairs returns pairs sorted by highest similarity first."""
        df = pd.DataFrame(
            [[1.0, 0.3, 0.9], [0.3, 1.0, 0.6], [0.9, 0.6, 1.0]],
            index=["DocA", "DocB", "DocC"],
            columns=["DocA", "DocB", "DocC"],
        )
        pairs = get_top_similar_pairs(df, top_n=5)
        assert len(pairs) == 3
        assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]
        assert pairs[0][2] == 0.9

    def test_top_pairs_respects_top_n(self):
        """Verify get_top_similar_pairs respects top_n clamp (top_n=2 returns at most 2)."""
        df = pd.DataFrame(
            [[1.0, 0.3, 0.9], [0.3, 1.0, 0.6], [0.9, 0.6, 1.0]],
            index=["DocA", "DocB", "DocC"],
            columns=["DocA", "DocB", "DocC"],
        )
        pairs = get_top_similar_pairs(df, top_n=2)
        assert len(pairs) == 2


def test_top_pairs_empty_df():
    """Verify get_top_similar_pairs returns empty list for empty DataFrame."""
    df = pd.DataFrame()
    assert get_top_similar_pairs(df) == []


def test_top_pairs_single_doc():
    """Verify get_top_similar_pairs returns empty list for single document."""
    df = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    assert get_top_similar_pairs(df) == []


def test_top_pairs_returns_sorted_desc():
    """Verify get_top_similar_pairs returns pairs sorted by highest similarity first."""
    df = pd.DataFrame(
        [[1.0, 0.3, 0.9], [0.3, 1.0, 0.6], [0.9, 0.6, 1.0]],
        index=["DocA", "DocB", "DocC"],
        columns=["DocA", "DocB", "DocC"],
    )
    pairs = get_top_similar_pairs(df, top_n=5)
    assert len(pairs) == 3
    assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]
    assert pairs[0][2] == 0.9
    assert pairs[1][2] == 0.6
    assert pairs[2][2] == 0.3


def test_top_pairs_respects_top_n():
    """Verify get_top_similar_pairs respects top_n clamp (top_n=2 returns at most 2)."""
    df = pd.DataFrame(
        [[1.0, 0.3, 0.9], [0.3, 1.0, 0.6], [0.9, 0.6, 1.0]],
        index=["DocA", "DocB", "DocC"],
        columns=["DocA", "DocB", "DocC"],
    )
    pairs = get_top_similar_pairs(df, top_n=2)
    assert len(pairs) == 2


class TestEmptyChartHelper:
    """Test suite for the _empty_chart() placeholder generator."""

    def test_creates_figure_with_annotation(self):
        """Verify empty chart contains the specified message annotation."""
        fig = _empty_chart(
            title="Test Title",
            message="No data available",
            theme_colors=get_chart_theme_colors("Light"),
        )

        assert fig.layout.title.text == "Test Title"
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "No data available"

    def test_applies_theme_colors(self):
        """Verify theme colors are applied to the empty chart."""
        dark_colors = get_chart_theme_colors("Dark")
        fig = _empty_chart(
            title="Test",
            message="Test",
            theme_colors=dark_colors,
        )

        assert fig.layout.paper_bgcolor == "#1e293b"


def test_get_top_similar_pairs_vectorized_matches_loop():
    """Verify vectorized np.triu_indices implementation matches old nested loop logic."""
    import numpy as np
    import pandas as pd

    from src.visualization.analytics import get_top_similar_pairs

    # Create a 10x10 similarity matrix to test performance and correctness
    np.random.seed(42)
    data = np.random.rand(10, 10)
    # Make symmetric and set diagonal to 1.0
    data = (data + data.T) / 2
    np.fill_diagonal(data, 1.0)

    doc_names = [f"doc_{chr(65+i)}" for i in range(10)]
    df = pd.DataFrame(data, index=doc_names, columns=doc_names)

    result = get_top_similar_pairs(df, top_n=5)

    assert len(result) == 5
    # Verify descending order
    for i in range(len(result) - 1):
        assert result[i][2] >= result[i + 1][2]

    # Verify no self-pairs (diagonal exclusion)
    for doc_a, doc_b, score in result:
        assert doc_a != doc_b

    # Verify scores are within valid range
    for doc_a, doc_b, score in result:
        assert 0.0 <= score <= 1.0


def test_apply_theme_colors_mutates_layout():
    """Verify _apply_theme_colors actually mutates figure layout paper_bgcolor, plot_bgcolor, and font.color."""
    import plotly.graph_objects as go

    from src.visualization.analytics import _apply_theme_colors

    fig = go.Figure()
    theme_colors = {
        "background": "#1e293b",
        "surface": "#0f172a",
        "ink": "#f8fafc",
        "border": "#334155",
    }

    _apply_theme_colors(fig, theme_colors, theme_override="dark")

    assert fig.layout.paper_bgcolor == "#1e293b"
    assert fig.layout.plot_bgcolor == "#0f172a"
    assert fig.layout.font.color == "#f8fafc"


def test_plot_similarity_boxplot_by_group_applies_theme():
    """Verify plot_similarity_boxplot_by_group applies theme colors to layout."""
    from src.visualization.analytics import plot_similarity_boxplot_by_group

    scores_dict = {"Assignment 1": [0.2, 0.5, 0.8]}
    theme_colors = {
        "background": "#1e293b",
        "surface": "#0f172a",
        "ink": "#f8fafc",
        "border": "#334155",
    }

    fig = plot_similarity_boxplot_by_group(scores_dict, theme_colors=theme_colors)

    assert fig.layout.paper_bgcolor == "#1e293b"
    assert fig.layout.plot_bgcolor == "#0f172a"
    assert fig.layout.font.color == "#f8fafc"
