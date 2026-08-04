from pathlib import Path


APP_PATH = Path("app/streamlit_app.py")


def test_heatmap_is_gated_by_lazy_helper():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="load_similarity_heatmap"' in source
    assert "heatmap_fig = build_visualization_lazily(" in source


def test_network_is_gated_by_lazy_helper():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="load_plagiarism_network"' in source
    assert "network_fig = build_visualization_lazily(" in source


def test_analytics_charts_are_gated():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="load_high_severity_trends"' in source
    assert (
        'key="load_most_plagiarized_documents"'
        in source
    )
    assert 'key="load_similarity_distribution"' in source


def test_duplicate_eager_heatmap_render_is_removed():
    source = APP_PATH.read_text(encoding="utf-8")

    assert source.count(
        "heatmap_fig = plot_similarity_heatmap("
    ) == 0


def test_lazy_controls_use_collapsed_expanders():
    source = APP_PATH.read_text(encoding="utf-8")

    assert source.count("expanded=False") >= 5
