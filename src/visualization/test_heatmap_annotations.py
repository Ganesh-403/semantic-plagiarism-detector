"""
tests/visualization/test_heatmap_annotations.py

Minimal tests for the show_annotations toggle on
plot_similarity_heatmap_plotly() (Issue #1795).

Covers:
    - show_annotations=True  -> annotations present
    - show_annotations=False -> annotations absent
    - matrices > 15x15 auto-disable annotations when show_annotations
      is left unset (default / backward-compatible behavior)
    - an explicit show_annotations=True on a >15x15 matrix overrides
      the auto-disable
    - omitting show_annotations entirely on a small matrix behaves
      exactly as it did before this change (annotations shown)
"""

import numpy as np
import pandas as pd

from src.visualization.heatmap import plot_similarity_heatmap_plotly


def _square_similarity_df(n: int) -> pd.DataFrame:
    """Build an n x n symmetric similarity matrix with a unit diagonal."""
    rng = np.random.default_rng(seed=42)
    values = rng.uniform(0.0, 1.0, size=(n, n))
    sym = (values + values.T) / 2
    np.fill_diagonal(sym, 1.0)
    labels = [f"doc_{i}.txt" for i in range(n)]
    return pd.DataFrame(sym, index=labels, columns=labels)


def test_show_annotations_true_renders_cell_text():
    df = _square_similarity_df(4)
    fig = plot_similarity_heatmap_plotly(df, show_annotations=True)
    assert len(fig.layout.annotations) == 4 * 4


def test_show_annotations_false_renders_no_cell_text():
    df = _square_similarity_df(4)
    fig = plot_similarity_heatmap_plotly(df, show_annotations=False)
    assert len(fig.layout.annotations) == 0


def test_default_unset_matches_previous_behavior_on_small_matrix():
    """Backward compatibility: omitting show_annotations on a matrix
    <= 15x15 must behave exactly as before this change (annotations on).
    """
    df = _square_similarity_df(5)
    fig = plot_similarity_heatmap_plotly(df)
    assert len(fig.layout.annotations) == 5 * 5


def test_large_matrix_auto_disables_annotations_when_unset():
    df = _square_similarity_df(16)
    fig = plot_similarity_heatmap_plotly(df)
    assert len(fig.layout.annotations) == 0


def test_large_matrix_explicit_override_keeps_annotations():
    """A caller explicitly asking for annotations on a >15x15 matrix
    must get them; only the *unset* default auto-disables."""
    df = _square_similarity_df(16)
    fig = plot_similarity_heatmap_plotly(df, show_annotations=True)
    assert len(fig.layout.annotations) == 16 * 16


def test_large_matrix_explicit_false_still_disables():
    df = _square_similarity_df(20)
    fig = plot_similarity_heatmap_plotly(df, show_annotations=False)
    assert len(fig.layout.annotations) == 0


def test_boundary_15x15_keeps_default_annotations():
    """15x15 is inclusive of the "on" side of the auto threshold."""
    df = _square_similarity_df(15)
    fig = plot_similarity_heatmap_plotly(df)
    assert len(fig.layout.annotations) == 15 * 15
