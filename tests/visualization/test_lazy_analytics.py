from unittest.mock import Mock

from src.visualization.analytics import (
    build_visualization_lazily,
)

import pandas as pd

from src.visualization.analytics import get_top_similar_pairs


def test_get_top_similar_pairs():
    similarity_df = pd.DataFrame(
        [
            [1.0, 0.95, 0.60],
            [0.95, 1.0, 0.82],
            [0.60, 0.82, 1.0],
        ],
        columns=["Doc A", "Doc B", "Doc C"],
        index=["Doc A", "Doc B", "Doc C"],
    )

    pairs = get_top_similar_pairs(similarity_df)

    assert len(pairs) == 3
    assert pairs[0] == ("Doc A", "Doc B", 0.95)
    assert pairs[1] == ("Doc B", "Doc C", 0.82)
    assert pairs[2] == ("Doc A", "Doc C", 0.60)
def test_get_top_similar_pairs_empty_dataframe():
    similarity_df = pd.DataFrame()

    assert get_top_similar_pairs(similarity_df) == []
def test_factory_is_not_called_when_visualization_is_disabled():
    factory = Mock(return_value="figure")

    result = build_visualization_lazily(False, factory)

    assert result is None
    factory.assert_not_called()


def test_factory_is_called_once_when_visualization_is_enabled():
    figure = object()
    factory = Mock(return_value=figure)

    result = build_visualization_lazily(True, factory)

    assert result is figure
    factory.assert_called_once_with()


def test_factory_exception_is_not_hidden():
    factory = Mock(side_effect=RuntimeError("render failed"))

    try:
        build_visualization_lazily(True, factory)
    except RuntimeError as exc:
        assert str(exc) == "render failed"
    else:
        raise AssertionError("Expected RuntimeError")

# ---------------------------------------------------------------------------
# Issue #1060 — Axis titles for plot_similarity_distribution
# ---------------------------------------------------------------------------

from src.visualization.analytics import plot_similarity_distribution


def test_plot_similarity_distribution_axis_titles():
    """Verify that the X and Y axis titles match the issue #1060 spec."""
    sim_matrix = pd.DataFrame(
        [
            [1.0, 0.85, 0.40],
            [0.85, 1.0, 0.60],
            [0.40, 0.60, 1.0],
        ],
        columns=["Doc A", "Doc B", "Doc C"],
        index=["Doc A", "Doc B", "Doc C"],
    )

    fig = plot_similarity_distribution(sim_matrix)

    assert fig.layout.xaxis.title.text == "Similarity Score Range (%)"
    assert fig.layout.yaxis.title.text == "Number of Document Pairs"


def test_plot_similarity_distribution_empty_axis_titles():
    """Verify that the empty-state chart also has the correct axis titles."""
    empty_df = pd.DataFrame()
    fig = plot_similarity_distribution(empty_df)

    assert fig.layout.xaxis.title.text == "Similarity Score Range (%)"
    assert fig.layout.yaxis.title.text == "Number of Document Pairs"


def test_plot_similarity_distribution_custom_title():
    """Verify that a custom title is still applied alongside the axis titles."""
    sim_matrix = pd.DataFrame(
        [
            [1.0, 0.50],
            [0.50, 1.0],
        ],
        columns=["Doc A", "Doc B"],
        index=["Doc A", "Doc B"],
    )

    fig = plot_similarity_distribution(sim_matrix, title="My Custom Title")

    assert fig.layout.title.text == "My Custom Title"
    assert fig.layout.xaxis.title.text == "Similarity Score Range (%)"
    assert fig.layout.yaxis.title.text == "Number of Document Pairs"
