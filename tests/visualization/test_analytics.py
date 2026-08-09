"""
tests/visualization/test_analytics.py
-------------------------------------
Unit tests for the analytics visualization functions.
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pytest

from src.visualization.analytics import (
    calculate_severity_ratios,
    plot_severity_donut_chart,
    plot_similarity_boxplot,
    plot_similarity_boxplot_by_group,
    plot_similarity_histogram,
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

    fig3 = plot_severity_donut_chart(
        [{"severity": "High"}], theme_colors=dark_theme
    )
    assert fig3.layout.paper_bgcolor == "#0F172A"
    assert fig3.layout.plot_bgcolor == "#1E293B"

    fig4 = plot_similarity_percentiles(
        [0.5, 0.8, 0.9], theme_colors=dark_theme
    )
    assert fig4.layout.paper_bgcolor == "#0F172A"



# ── Hierarchical Clustering Dendrogram (Issue #1367) ──────────────────────


def _make_similarity_matrix(
    n: int = 5, seed: int = 42
) -> pd.DataFrame:
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
    assert (
        "No similarity data available"
        in fig.layout.annotations[0].text
    )


def test_plot_hierarchical_dendrogram_single_document_returns_annotation_figure():
    """A 1×1 matrix must return an annotation figure, not raise."""
    from src.visualization.analytics import plot_hierarchical_dendrogram

    single_df = pd.DataFrame(
        [[1.0]], index=["only_doc"], columns=["only_doc"]
    )
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
            f"Ward merge distance mismatch: expected {expected}, "
            f"got {rendered}"
        )


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
    incidents = [
        {"similarity_score": 0.9},   # High
        {"similarity_score": 0.85},  # High
        {"similarity_score": 0.6},   # Medium
        {"similarity_score": 0.3},   # Low
    ]
    ratios = calculate_severity_ratios(incidents)

    assert ratios == {"High": 50.0, "Medium": 25.0, "Low": 25.0}


def test_calculate_severity_ratios_ignores_invalid_scores():
    """Incidents with missing or non-numeric scores should be skipped."""
    incidents = [
        {"similarity_score": 0.9},
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