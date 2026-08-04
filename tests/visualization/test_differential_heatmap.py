"""
test_differential_heatmap.py
--------------------------------
Exhaustive unit test suite for similarity matrix differential heatmap visualizer (#1369).
Validates delta calculation (matrix_a - matrix_b), diverging colormaps (Coolwarm/RdBu),
alignment on common documents, hover annotations, class tag filters, and static Matplotlib fallback.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from src.visualization.heatmap import (
    plot_differential_heatmap,
    plot_differential_heatmap_matplotlib,
)


def _build_test_matrix(values: list[list[float]], doc_names: list[str]) -> pd.DataFrame:
    """Utility helper to build symmetric DataFrame similarity matrices."""
    return pd.DataFrame(values, index=doc_names, columns=doc_names)


def test_differential_heatmap_delta_computation_correctness():
    """Verify that delta matrix is precisely (matrix_a - matrix_b)."""
    names = ["doc1.txt", "doc2.txt", "doc3.txt"]
    val_a = [
        [1.00, 0.90, 0.30],
        [0.90, 1.00, 0.80],
        [0.30, 0.80, 1.00],
    ]
    val_b = [
        [1.00, 0.50, 0.40],
        [0.50, 1.00, 0.20],
        [0.40, 0.20, 1.00],
    ]
    df_a = _build_test_matrix(val_a, names)
    df_b = _build_test_matrix(val_b, names)

    fig = plot_differential_heatmap(df_a, df_b, label_a="Lexical", label_b="Vector")
    heatmap = fig.data[0]

    # Delta for doc1-doc2 = 0.90 - 0.50 = +0.40
    # Delta for doc1-doc3 = 0.30 - 0.40 = -0.10
    # Delta for doc2-doc3 = 0.80 - 0.20 = +0.60
    assert heatmap.z[0][1] == pytest.approx(0.40, abs=1e-5)
    assert heatmap.z[0][2] == pytest.approx(-0.10, abs=1e-5)
    assert heatmap.z[1][2] == pytest.approx(0.60, abs=1e-5)


def test_differential_heatmap_diverging_color_bounds():
    """Verify zmin and zmax are set symmetrically around zmid=0.0."""
    names = ["docA", "docB"]
    df_a = _build_test_matrix([[1.0, 0.95], [0.95, 1.0]], names)
    df_b = _build_test_matrix([[1.0, 0.15], [0.15, 1.0]], names)

    fig = plot_differential_heatmap(df_a, df_b, colorscale="Coolwarm")
    trace = fig.data[0]

    assert trace.zmid == 0.0
    assert trace.zmax == pytest.approx(0.80, abs=1e-5)
    assert trace.zmin == pytest.approx(-0.80, abs=1e-5)
    assert trace.colorscale is not None


def test_differential_heatmap_document_alignment_subset():
    """Verify automatic alignment when matrices have overlapping document subsets."""
    names_a = ["doc1", "doc2", "doc3"]
    names_b = ["doc2", "doc3", "doc4"]

    df_a = _build_test_matrix([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]], names_a)
    df_b = _build_test_matrix([[1.0, 0.4, 0.2], [0.4, 1.0, 0.6], [0.2, 0.6, 1.0]], names_b)

    fig = plot_differential_heatmap(df_a, df_b)
    trace = fig.data[0]

    assert list(trace.x) == ["doc2", "doc3"]
    assert list(trace.y) == ["doc2", "doc3"]


def test_differential_heatmap_hover_template_formatting():
    """Verify hover text contains algorithm labels and formatted score values."""
    names = ["doc1", "doc2"]
    df_a = _build_test_matrix([[1.0, 0.75], [0.75, 1.0]], names)
    df_b = _build_test_matrix([[1.0, 0.25], [0.25, 1.0]], names)

    fig = plot_differential_heatmap(
        df_a, df_b, label_a="Model X", label_b="Model Y"
    )

    hover_text = fig.data[0].hovertext[0][1]
    assert "<b>Model X:</b> 0.75" in hover_text
    assert "<b>Model Y:</b> 0.25" in hover_text
    assert "<b>Delta (Model X - Model Y):</b> +0.50" in hover_text


def test_differential_heatmap_class_tag_filtering():
    """Verify class_tag parameter filters rows and columns appropriately."""
    names = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    df_a = _build_test_matrix([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]], names)
    df_b = _build_test_matrix([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]], names)

    doc_class_map = {
        "doc1.pdf": "CS101",
        "doc2.pdf": "CS101",
        "doc3.pdf": "ENG202",
    }

    fig = plot_differential_heatmap(
        df_a, df_b, class_tag="CS101", doc_class_map=doc_class_map
    )

    trace = fig.data[0]
    assert list(trace.x) == ["doc1.pdf", "doc2.pdf"]


def test_differential_heatmap_theme_palette_styling():
    """Verify theme colors set paper and plot background appropriately."""
    names = ["doc1", "doc2"]
    df_a = _build_test_matrix([[1.0, 0.8], [0.8, 1.0]], names)
    df_b = _build_test_matrix([[1.0, 0.4], [0.4, 1.0]], names)

    custom_theme = {
        "background": "#0F172A",
        "surface": "#1E293B",
        "ink": "#F8FAFC",
    }

    fig = plot_differential_heatmap(df_a, df_b, theme_colors=custom_theme)

    assert fig.layout.paper_bgcolor == "#0F172A"
    assert fig.layout.plot_bgcolor == "#0F172A"
    assert fig.layout.font.color == "#F8FAFC"


def test_differential_heatmap_matplotlib_rendering():
    """Verify static Matplotlib implementation creates non-empty Figure with correct title."""
    names = ["doc1", "doc2", "doc3"]
    df_a = _build_test_matrix([[1.0, 0.9, 0.4], [0.9, 1.0, 0.7], [0.4, 0.7, 1.0]], names)
    df_b = _build_test_matrix([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]], names)

    fig = plot_differential_heatmap_matplotlib(
        df_a, df_b, title="Static Matplotlib Differential Heatmap"
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Static Matplotlib Differential Heatmap"
    plt.close(fig)
