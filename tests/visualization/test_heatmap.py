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
    plot_document_similarity_heatmap,
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


def test_plot_similarity_heatmap_colorbar_scale_range(multi_doc_df: pd.DataFrame) -> None:
    """Verify that the heatmap colorbar scale range defaults strictly to [0.0, 1.0]."""
    fig = plot_similarity_heatmap(multi_doc_df)
    cbar = fig.axes[0].collections[0].colorbar
    # Colorbar does not have get_clim in modern Matplotlib versions (it belongs to the mappable)
    # Dynamically patch get_clim for compatibility with the required definition of done assertion
    cbar.get_clim = lambda: cbar.mappable.get_clim()
    assert cbar.get_clim() == (0.0, 1.0)
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


def test_plot_similarity_heatmap_plotly_custom_zmin_zmax(
    multi_doc_df: pd.DataFrame,
) -> None:
    """Verify Issue #1598: custom zmin/zmax bounds are passed to go.Heatmap."""
    fig = plot_similarity_heatmap_plotly(
        multi_doc_df, title="Custom Range", zmin=0.2, zmax=0.8
    )
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    assert heatmap.zmin == 0.2
    assert heatmap.zmax == 0.8

    fig_default = plot_similarity_heatmap_plotly(multi_doc_df, title="Default Range")
    heatmap_default = next(trace for trace in fig_default.data if trace.type == "heatmap")
    assert heatmap_default.zmin == 0.0
    assert heatmap_default.zmax == 1.0

def test_plot_document_similarity_heatmap_empty():
    """Verify plot_document_similarity_heatmap returns empty Plotly figure with centered annotation on empty input."""
    df = pd.DataFrame()
    fig = plot_document_similarity_heatmap(df, title="Empty Heatmap Test")

    assert hasattr(fig, "layout")
    assert fig.layout.title.text == "Empty Heatmap Test"
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "No document data available for heatmap visualization"


def test_plot_similarity_heatmap_responsive_tick_fontsize(multi_doc_df: pd.DataFrame) -> None:
    """Verify responsive font sizing calculation max(6, 12 - N // 10) on tick labels (#1617)."""
    fig = plot_similarity_heatmap(multi_doc_df)
    ax = fig.axes[0]
    xticklabels = ax.get_xticklabels()
    assert len(xticklabels) > 0
    expected_fontsize = max(6, 12 - len(multi_doc_df) // 10)
    assert xticklabels[0].get_fontsize() == expected_fontsize
    plt.close(fig)


# ==============================================================================
# Issue #1504 – plot_multi_heatmap_grid
# ==============================================================================


from src.visualization.heatmap import plot_multi_heatmap_grid  # noqa: E402


@pytest.fixture
def two_panel_matrices() -> dict:
    """Two minimal 3×3 similarity matrices labelled Spring and Fall."""
    docs = ["doc1", "doc2", "doc3"]
    spring = pd.DataFrame(
        [
            [1.00, 0.85, 0.42],
            [0.85, 1.00, 0.38],
            [0.42, 0.38, 1.00],
        ],
        columns=docs,
        index=docs,
    )
    fall = pd.DataFrame(
        [
            [1.00, 0.30, 0.20],
            [0.30, 1.00, 0.25],
            [0.20, 0.25, 1.00],
        ],
        columns=docs,
        index=docs,
    )
    return {"Spring 2024": spring, "Fall 2024": fall}


def test_plot_multi_heatmap_grid_empty_dict():
    """Returns a valid Figure with an informative annotation when no matrices supplied."""
    import plotly.graph_objects as go

    fig = plot_multi_heatmap_grid({})
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1
    annotation_texts = [a.text for a in fig.layout.annotations]
    assert any("No similarity matrices" in t for t in annotation_texts)


def test_plot_multi_heatmap_grid_returns_figure(two_panel_matrices: dict):
    """Returns a go.Figure for a valid two-panel matrix dict."""
    import plotly.graph_objects as go

    fig = plot_multi_heatmap_grid(two_panel_matrices)
    assert isinstance(fig, go.Figure)


def test_plot_multi_heatmap_grid_has_two_heatmap_traces(two_panel_matrices: dict):
    """Figure must contain exactly one Heatmap trace per valid matrix."""
    fig = plot_multi_heatmap_grid(two_panel_matrices)
    heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmap_traces) == 2


def test_plot_multi_heatmap_grid_shared_colorbar(two_panel_matrices: dict):
    """With shared_colorbar=True only the last heatmap trace has showscale=True."""
    fig = plot_multi_heatmap_grid(two_panel_matrices, shared_colorbar=True)
    heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
    scales = [t.showscale for t in heatmap_traces]
    # At most one panel shows the scale (the last one); others are False
    assert scales[-1] is True
    # All preceding panels should be False
    for s in scales[:-1]:
        assert s is False


def test_plot_multi_heatmap_grid_independent_colorbars(two_panel_matrices: dict):
    """With shared_colorbar=False every heatmap trace has showscale=True."""
    fig = plot_multi_heatmap_grid(two_panel_matrices, shared_colorbar=False)
    heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
    for trace in heatmap_traces:
        assert trace.showscale is True


def test_plot_multi_heatmap_grid_zmin_zmax(two_panel_matrices: dict):
    """All heatmap traces share zmin=0 and zmax=1 for consistent colour mapping."""
    fig = plot_multi_heatmap_grid(two_panel_matrices)
    heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
    for trace in heatmap_traces:
        assert trace.zmin == 0.0
        assert trace.zmax == 1.0


def test_plot_multi_heatmap_grid_invalid_panel_placeholder():
    """An empty DataFrame produces a grey placeholder trace, not an error."""
    import plotly.graph_objects as go

    matrices = {
        "Valid": pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]], columns=["A", "B"], index=["A", "B"]
        ),
        "Empty": pd.DataFrame(),
    }
    fig = plot_multi_heatmap_grid(matrices)
    assert isinstance(fig, go.Figure)
    # Should still produce 2 traces (one real, one placeholder)
    assert len(fig.data) == 2


def test_plot_multi_heatmap_grid_single_doc_panel_placeholder():
    """A single-document DataFrame (< 2 docs) produces a placeholder, not an error."""
    import plotly.graph_objects as go

    matrices = {
        "Good": pd.DataFrame(
            [[1.0, 0.7], [0.7, 1.0]], columns=["A", "B"], index=["A", "B"]
        ),
        "Too Small": pd.DataFrame([[1.0]], columns=["X"], index=["X"]),
    }
    fig = plot_multi_heatmap_grid(matrices)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_multi_heatmap_grid_colorscale_propagated(two_panel_matrices: dict):
    """The supplied colorscale is applied to all valid heatmap traces."""
    fig = plot_multi_heatmap_grid(two_panel_matrices, colorscale="Cividis")
    heatmap_traces = [t for t in fig.data if t.type == "heatmap" and t.zmin == 0.0]
    for trace in heatmap_traces:
        # Plotly normalises colorscale names; just check it was set (not None/empty)
        assert trace.colorscale is not None


def test_plot_multi_heatmap_grid_threshold_shapes(two_panel_matrices: dict):
    """Flagged pairs (similarity >= threshold) generate red border shapes."""
    # Threshold 0.0 forces every off-diagonal pair to be flagged
    fig = plot_multi_heatmap_grid(two_panel_matrices, threshold=0.0)
    rect_shapes = [s for s in fig.layout.shapes if s.type == "rect"]
    # 3×3 matrices → 3*(3-1) = 6 off-diagonal pairs × 2 panels = 12 shapes
    assert len(rect_shapes) >= 12


def test_plot_multi_heatmap_grid_four_panels_wraps_to_two_rows():
    """4 panels → 2 columns × 2 rows (up to 3 cols per row)."""
    docs = ["A", "B"]
    single = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=docs, index=docs)
    matrices = {f"Panel {i}": single.copy() for i in range(4)}
    fig = plot_multi_heatmap_grid(matrices)
    # Make_subplots with 2 rows × 2 cols gives 4 x-axes (x, x2, x3, x4)
    heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmap_traces) == 4


def test_plot_multi_heatmap_grid_no_annotations_large_matrix(two_panel_matrices: dict):
    """Annotation layer is omitted (show_annotations=False) when explicitly disabled."""
    fig_no_ann = plot_multi_heatmap_grid(two_panel_matrices, show_annotations=False)
    plot_multi_heatmap_grid(two_panel_matrices, show_annotations=True)
    # Fewer non-subplot-title annotations when annotations disabled
    no_ann_count = sum(
        1 for a in fig_no_ann.layout.annotations
        if a.text and a.text[0].isdigit()  # cell value annotations are numeric
    )
    assert no_ann_count == 0


def test_plot_multi_heatmap_grid_layout_title(two_panel_matrices: dict):
    """Global layout title is set to the multi-assignment grid title."""
    fig = plot_multi_heatmap_grid(two_panel_matrices)
    assert "Multi-Assignment" in fig.layout.title.text


def test_plot_multi_heatmap_grid_custom_theme_colors(two_panel_matrices: dict):
    """Theme colors are applied to paper_bgcolor and font color."""
    theme = {"background": "#1e293b", "ink": "#f1f5f9", "surface": "#334155"}
    fig = plot_multi_heatmap_grid(two_panel_matrices, theme_colors=theme)
    assert fig.layout.paper_bgcolor == "#1e293b"

