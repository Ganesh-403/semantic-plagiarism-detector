"""
tests/visualization/test_heatmap.py
-----------------------------------
Unit tests for plot_similarity_heatmap edge cases.
"""

import io
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualization.heatmap import (
    export_heatmap_matrix_csv,
    filter_heatmap_by_class_tag,
    plot_differential_heatmap,
    plot_differential_heatmap_matplotlib,
    plot_similarity_heatmap,
    plot_similarity_heatmap_plotly,
)


def test_plot_similarity_heatmap_empty_dataframe():
    """Test heatmap generation when an empty DataFrame is passed."""
    df = pd.DataFrame()
    fig = plot_similarity_heatmap(df, title="Empty Heatmap")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_similarity_heatmap_large_dataframe():
    """Test heatmap generation with a large DataFrame of realistic values."""
    data = {
        "doc1": [1.00, 0.85, 0.42, 0.23, 0.15],
        "doc2": [0.85, 1.00, 0.38, 0.19, 0.12],
        "doc3": [0.42, 0.38, 1.00, 0.67, 0.31],
        "doc4": [0.23, 0.19, 0.67, 1.00, 0.28],
        "doc5": [0.15, 0.12, 0.31, 0.28, 1.00],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3", "doc4", "doc5"])

    fig = plot_similarity_heatmap(df)

    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def single_doc_df() -> pd.DataFrame:
    return pd.DataFrame([[1.0]], columns=["doc1"], index=["doc1"])


@pytest.fixture
def multi_doc_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.00, 0.85, 0.45],
            [0.85, 1.00, 0.60],
            [0.45, 0.60, 1.00],
        ],
        columns=["doc_A", "doc_B", "doc_C"],
        index=["doc_A", "doc_B", "doc_C"],
    )


@pytest.fixture
def masked_threshold_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 0.4, 0.8],
            [0.4, 1.0, 0.3],
            [0.8, 0.3, 1.0],
        ],
        columns=["doc1", "doc2", "doc3"],
        index=["doc1", "doc2", "doc3"],
    )


# ==============================================================================
# Static Heatmap (Matplotlib/Seaborn) Tests
# ==============================================================================


def test_plot_similarity_heatmap_empty(empty_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap(empty_df, title="Empty Heatmap")
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_similarity_heatmap_single(single_doc_df: pd.DataFrame) -> None:
    """Verify Issue #839: 1x1 matrix returns an informative warning box."""
    fig = plot_similarity_heatmap(single_doc_df, title="Single Document Heatmap")
    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("At least 2 documents are required" in text for text in texts)

    plt.close(fig)


def test_plot_similarity_heatmap_multi(multi_doc_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap(
        multi_doc_df, title="Multi Document Heatmap", show_annotations=True
    )
    main_ax = next(
        (ax for ax in fig.axes if ax.get_title() == "Multi Document Heatmap"), None
    )
    assert main_ax is not None
    assert len(main_ax.texts) > 0
    assert main_ax.get_xlabel() == "Documents"
    assert main_ax.get_ylabel() == "Documents"
    plt.close(fig)


def test_plot_similarity_heatmap_no_annotation(multi_doc_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap(
        multi_doc_df, title="No Annotation Heatmap", show_annotations=False
    )
    main_ax = next(
        (ax for ax in fig.axes if ax.get_title() == "No Annotation Heatmap"), None
    )
    assert main_ax is not None
    assert len(main_ax.texts) == 0
    plt.close(fig)


def test_plot_similarity_heatmap_with_mask_threshold(
    masked_threshold_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap(
        masked_threshold_df, title="Masked Heatmap", mask_threshold=0.5
    )
    main_ax = next((ax for ax in fig.axes if ax.get_title() == "Masked Heatmap"), None)
    assert main_ax is not None

    texts = [t.get_text() for t in main_ax.texts if t.get_text()]
    assert "1.00" in texts
    assert "0.40" not in texts
    plt.close(fig)


# ==============================================================================
# Interactive Heatmap (Plotly) Tests
# ==============================================================================


def test_plot_similarity_heatmap_plotly_empty(empty_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap_plotly(empty_df, title="Empty Plotly Heatmap")
    assert hasattr(fig, "layout")
    assert fig.layout.title.text == "Empty Plotly Heatmap"


def test_plot_similarity_heatmap_plotly_single(single_doc_df: pd.DataFrame) -> None:
    """Verify Issue #839: Plotly 1x1 matrix returns warning box annotation."""
    fig = plot_similarity_heatmap_plotly(single_doc_df, title="Single Plotly Heatmap")
    assert hasattr(fig, "layout")
    plotly_annotations = [a.text for a in fig.layout.annotations]
    assert any(
        "At least 2 documents are required" in text for text in plotly_annotations
    )


def test_plot_similarity_heatmap_plotly_no_annotation(
    multi_doc_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap_plotly(
        multi_doc_df, title="No Annotation Plotly Heatmap", show_annotations=False
    )
    assert hasattr(fig, "layout")
    assert len(fig.layout.annotations) == 0


def test_plot_similarity_heatmap_plotly_with_mask_threshold(
    masked_threshold_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap_plotly(
        masked_threshold_df, title="Masked Plotly", mask_threshold=0.5
    )
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    z_values = heatmap.z
    assert z_values[0][1] is None
    assert z_values[1][0] is None
    assert z_values[0][0] == 1.0


# ==============================================================================
# Export Generation Tests
# ==============================================================================


def test_plot_similarity_heatmap_png_export_valid_bytes(
    multi_doc_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap(multi_doc_df, title="Export Test", dpi=150)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    png_bytes = buf.getvalue()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 2000

    plt.close(fig)
    buf.close()


def test_plot_similarity_heatmap_png_export_empty_df(empty_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap(empty_df, title="Empty Export Test", dpi=150)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    png_bytes = buf.getvalue()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    plt.close(fig)
    buf.close()


def test_plot_similarity_heatmap_png_export_custom_theme(
    multi_doc_df: pd.DataFrame,
) -> None:
    custom_theme = {
        "background": "#1E293B",
        "surface": "#0F172A",
        "ink": "#F8FAFC",
        "border": "#334155",
    }

    fig = plot_similarity_heatmap(
        multi_doc_df, title="Themed Export Test", theme_colors=custom_theme, dpi=150
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    png_bytes = buf.getvalue()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 2000
    plt.close(fig)
    buf.close()


def test_filter_heatmap_by_class_tag_matches_subset():
    df = pd.DataFrame(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        columns=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        index=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    )
    doc_class_map = {
        "doc1.pdf": "Class A",
        "doc2.pdf": "Class A",
        "doc3.pdf": "Class B",
    }

    filtered_df = filter_heatmap_by_class_tag(
        df, class_tag="Class A", doc_class_map=doc_class_map
    )

    assert list(filtered_df.columns) == ["doc1.pdf", "doc2.pdf"]
    assert list(filtered_df.index) == ["doc1.pdf", "doc2.pdf"]


def test_filter_heatmap_by_class_tag_all_classes_returns_full():
    df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        columns=["doc1.pdf", "doc2.pdf"],
        index=["doc1.pdf", "doc2.pdf"],
    )

    full_all = filter_heatmap_by_class_tag(df, class_tag="All Classes")
    full_none = filter_heatmap_by_class_tag(df, class_tag=None)

    assert full_all.shape == (2, 2)
    assert full_none.shape == (2, 2)


def test_filter_heatmap_by_class_tag_no_match_returns_empty():
    df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        columns=["doc1.pdf", "doc2.pdf"],
        index=["doc1.pdf", "doc2.pdf"],
    )
    doc_class_map = {"doc1.pdf": "Class A", "doc2.pdf": "Class A"}

    empty_filtered = filter_heatmap_by_class_tag(
        df, class_tag="Class Nonexistent", doc_class_map=doc_class_map
    )

    assert empty_filtered.empty


def test_plot_similarity_heatmap_with_class_tag_filter():
    df = pd.DataFrame(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        columns=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        index=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    )
    doc_class_map = {
        "doc1.pdf": "Class A",
        "doc2.pdf": "Class A",
        "doc3.pdf": "Class B",
    }

    fig = plot_similarity_heatmap(
        df,
        title="Class A Heatmap",
        class_tag="Class A",
        doc_class_map=doc_class_map,
    )
    assert hasattr(fig, "axes")


def test_plot_similarity_heatmap_plotly_with_class_tag_filter():
    df = pd.DataFrame(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        columns=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        index=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    )
    doc_class_map = {
        "doc1.pdf": "Class A",
        "doc2.pdf": "Class A",
        "doc3.pdf": "Class B",
    }

    fig = plot_similarity_heatmap_plotly(
        df,
        title="Plotly Class A Heatmap",
        class_tag="Class A",
        doc_class_map=doc_class_map,
    )
    assert hasattr(fig, "layout")
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    assert list(heatmap.x) == ["doc1.pdf", "doc2.pdf"]


def test_plot_similarity_heatmap_dim_diagonal(multi_doc_df: pd.DataFrame) -> None:
    fig = plot_similarity_heatmap(
        multi_doc_df,
        title="Dim Diagonal Heatmap",
        dim_diagonal=True,
    )
    assert hasattr(fig, "axes")
    main_ax = next(
        (ax for ax in fig.axes if ax.get_title() == "Dim Diagonal Heatmap"), None
    )
    assert main_ax is not None

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    png_bytes = buf.getvalue()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    plt.close(fig)
    buf.close()


def test_plot_similarity_heatmap_plotly_dim_diagonal(
    multi_doc_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap_plotly(
        multi_doc_df,
        title="Plotly Dim Diagonal",
        dim_diagonal=True,
    )
    assert hasattr(fig, "layout")
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    z_values = heatmap.z
    assert z_values[0][0] is None
    assert z_values[1][1] is None
    assert z_values[2][2] is None
    assert z_values[0][1] == 0.85


def test_plot_similarity_heatmap_uses_cividis_colormap(
    multi_doc_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap(
        multi_doc_df,
        title="Cividis Heatmap",
        colormap_name="Cividis",
    )

    main_ax = next(
        (ax for ax in fig.axes if ax.get_title() == "Cividis Heatmap"),
        None,
    )

    assert main_ax is not None

    mesh = main_ax.collections[0]
    assert mesh.cmap.name == "cividis"

    plt.close(fig)


def test_plot_similarity_heatmap_dim_diagonal_single_doc(
    single_doc_df: pd.DataFrame,
) -> None:
    fig = plot_similarity_heatmap(
        single_doc_df, title="Single Dim Diagonal", dim_diagonal=True
    )
    assert hasattr(fig, "axes")
    plt.close(fig)

    fig_plotly = plot_similarity_heatmap_plotly(
        single_doc_df, title="Plotly Single Dim Diagonal", dim_diagonal=True
    )
    assert hasattr(fig_plotly, "layout")


# ==============================================================================
# CSV Export Tests
# ==============================================================================


def test_export_heatmap_matrix_csv_valid_output():
    df = pd.DataFrame(
        [
            [1.00, 0.85, 0.45],
            [0.85, 1.00, 0.60],
            [0.45, 0.60, 1.00],
        ],
        columns=["doc_A", "doc_B", "doc_C"],
        index=["doc_A", "doc_B", "doc_C"],
    )

    csv_bytes = export_heatmap_matrix_csv(df)

    assert isinstance(csv_bytes, bytes)
    decoded = csv_bytes.decode("utf-8")
    assert "doc_A" in decoded
    lines = decoded.strip().splitlines()
    assert len(lines) == 4


def test_export_heatmap_matrix_csv_empty_dataframe():
    df = pd.DataFrame()
    csv_bytes = export_heatmap_matrix_csv(df)
    assert isinstance(csv_bytes, bytes)


# ==============================================================================
# Differential Heatmap Visualizer Tests (#1369)
# ==============================================================================


def test_plot_differential_heatmap_basic():
    """Verify plot_differential_heatmap computes delta matrix and returns Plotly figure."""
    import plotly.graph_objects as go

    matrix_a = pd.DataFrame(
        [
            [1.00, 0.85, 0.40],
            [0.85, 1.00, 0.70],
            [0.40, 0.70, 1.00],
        ],
        index=["doc1", "doc2", "doc3"],
        columns=["doc1", "doc2", "doc3"],
    )
    matrix_b = pd.DataFrame(
        [
            [1.00, 0.60, 0.50],
            [0.60, 1.00, 0.40],
            [0.50, 0.40, 1.00],
        ],
        index=["doc1", "doc2", "doc3"],
        columns=["doc1", "doc2", "doc3"],
    )

    fig = plot_differential_heatmap(
        matrix_a, matrix_b, title="Lexical vs Vector Similarity Delta"
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Lexical vs Vector Similarity Delta"
    assert len(fig.data) == 1
    heatmap_trace = fig.data[0]

    # Verify z values equal matrix_a - matrix_b
    # doc1-doc2 delta = 0.85 - 0.60 = +0.25
    # doc1-doc3 delta = 0.40 - 0.50 = -0.10
    assert heatmap_trace.z[0][1] == pytest.approx(0.25, abs=1e-5)
    assert heatmap_trace.z[0][2] == pytest.approx(-0.10, abs=1e-5)


def test_plot_differential_heatmap_diverging_colorscale():
    """Verify diverging colormap bounds zmin=-max_abs, zmax=max_abs around 0."""
    matrix_a = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["docA", "docB"],
        columns=["docA", "docB"],
    )
    matrix_b = pd.DataFrame(
        [[1.0, 0.4], [0.4, 1.0]],
        index=["docA", "docB"],
        columns=["docA", "docB"],
    )

    fig = plot_differential_heatmap(matrix_a, matrix_b, colorscale="Coolwarm")
    trace = fig.data[0]

    # delta is +0.50
    assert trace.zmax == pytest.approx(0.50, abs=1e-5)
    assert trace.zmin == pytest.approx(-0.50, abs=1e-5)
    assert trace.zmid == 0.0


def test_plot_differential_heatmap_empty_matrices():
    """Verify graceful handling when input matrices are empty."""
    import plotly.graph_objects as go

    empty_df = pd.DataFrame()
    fig = plot_differential_heatmap(empty_df, empty_df, title="Empty Delta")

    assert isinstance(fig, go.Figure)
    annotations = [a.text for a in fig.layout.annotations]
    assert any("empty" in text.lower() for text in annotations)


def test_plot_differential_heatmap_single_document():
    """Verify graceful handling for 1x1 matrix (< 2 documents)."""
    df_a = pd.DataFrame([[1.0]], index=["doc1"], columns=["doc1"])
    df_b = pd.DataFrame([[1.0]], index=["doc1"], columns=["doc1"])

    fig = plot_differential_heatmap(df_a, df_b)
    annotations = [a.text for a in fig.layout.annotations]
    assert any("At least 2" in text for text in annotations)


def test_plot_differential_heatmap_matplotlib():
    """Verify static Matplotlib differential heatmap generator."""
    matrix_a = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    matrix_b = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )

    fig = plot_differential_heatmap_matplotlib(matrix_a, matrix_b)
    assert isinstance(fig, Figure)
    plt.close(fig)
def test_plot_similarity_heatmap_plotly_custom_colorscale(
    multi_doc_df: pd.DataFrame,
) -> None:
    """Verify Issue #1397: a custom Plotly colorscale string is applied to the trace."""
    fig = plot_similarity_heatmap_plotly(
        multi_doc_df, title="Custom Colorscale", colorscale="Plasma"
    )
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    assert heatmap.colorscale is not None

    fig_default = plot_similarity_heatmap_plotly(multi_doc_df, title="Default Colorscale")
    heatmap_default = next(trace for trace in fig_default.data if trace.type == "heatmap")
    assert heatmap_default.colorscale != heatmap.colorscale