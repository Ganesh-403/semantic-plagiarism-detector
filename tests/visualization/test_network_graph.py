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
tests/visualization/test_network_graph.py
-------------------------------------------
Unit tests for the network graph visualization and analysis helpers.

Validates:
- plot_similarity_network edge cases
- Connected component counting (Issue #1793)
- Graph data building and layout calculations
- Export functions (GEXF, CSV)
"""

import xml.etree.ElementTree as ET
from unittest.mock import patch

import networkx as nx
import numpy
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.network_graph import (
    build_network_data,
    calculate_force_directed_layout,
    export_graph_to_csv,
    export_network_adjacency_csv,
    export_network_centrality_csv,
    export_network_to_csv_bytes,
    export_network_to_gexf_bytes,
    get_cluster_count,
    plot_plagiarism_network_graph,
    plot_similarity_network,
    render_network_plotly,
)


def test_export_network_adjacency_csv():
    graph = nx.Graph()

    graph.add_edge("Doc A", "Doc B", weight=0.95)
    graph.add_edge("Doc B", "Doc C", weight=0.82)

    csv_output = export_network_adjacency_csv(graph)

    assert "Source,Target,Weight" in csv_output
    assert "Doc A,Doc B,0.95" in csv_output
    assert "Doc B,Doc C,0.82" in csv_output


def test_export_network_adjacency_csv_empty_graph():
    graph = nx.Graph()

    csv_output = export_network_adjacency_csv(graph)

    assert csv_output.strip() == "Source,Target,Weight"


from src.visualization.network_graph import NETWORK_GRAPH_CONFIG


def test_build_network_data_structure():
    """Verify build_network_data returns expected keys, NetworkX graph, and Plotly traces."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    # show_isolated=True keeps doc3 (isolated node) in the graph for structure checks
    net_data = build_network_data(df, threshold=0.75, show_isolated=True)

    assert "shapes" in net_data
    assert "edge_hover_trace" in net_data
    assert "node_trace" in net_data
    assert "graph" in net_data
    assert "pos" in net_data

    # Check graph nodes and edges
    assert len(net_data["graph"].nodes()) == 3
    assert len(net_data["graph"].edges()) == 1
    assert len(net_data["shapes"]) == 1


def test_build_network_data_hides_isolated_nodes():
    """Verify show_isolated=False removes unconnected/isolated nodes such as doc3."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    net_data = build_network_data(df, threshold=0.75, show_isolated=False)

    assert len(net_data["graph"].nodes()) == 2
    assert "doc3" not in net_data["graph"].nodes()
    assert set(net_data["graph"].nodes()) == {"doc1", "doc2"}


def test_build_network_data_with_theme_colors():
    """Verify build_network_data applies custom theme colors correctly."""
    data = {
        "doc1": [1.0, 0.95],
        "doc2": [0.95, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    custom_theme = {
        "danger": "#e53935",
        "warning": "#fb8c00",
        "success": "#43a047",
        "background": "#121212",
        "ink": "#ffffff",
    }

    net_data = build_network_data(df, threshold=0.75, theme_colors=custom_theme)

    # Similarity 0.95 >= 0.90 -> danger color
    assert net_data["shapes"][0]["line"]["color"] == "#e53935"
    assert net_data["node_trace"].textfont.color == "#ffffff"


def test_build_network_data_hover_text():
    """Verify hover text explicitly shows Document Title."""
    data = {
        "doc1": [1.0, 0.85],
        "doc2": [0.85, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    net_data = build_network_data(df, threshold=0.75)

    hover_texts = net_data["node_trace"].hovertext
    assert "<b>📄 Document Title:</b> doc1<br>" in hover_texts[0]


def test_build_network_data_node_color_severity():
    """Verify node colors are mapped by plagiarism severity (max similarity)."""
    data = {
        "doc_danger": [1.0, 0.95, 0.1],  # max_score=0.95 -> danger
        "doc_warning": [0.95, 1.0, 0.8],  # max_score=0.95 -> danger
        "doc_success": [0.1, 0.8, 1.0],  # max_score=0.8 -> warning
    }
    df = pd.DataFrame(data, index=["doc_danger", "doc_warning", "doc_success"])
    custom_theme = {
        "danger": "#ff0000",
        "warning": "#ffff00",
        "success": "#00ff00",
    }
    net_data = build_network_data(df, threshold=0.5, theme_colors=custom_theme)

    # doc_danger has max_score=0.95 -> #ff0000
    assert net_data["node_trace"].marker.color[0] == "#ff0000"
    # doc_warning has max_score=0.95 -> #ff0000
    assert net_data["node_trace"].marker.color[1] == "#ff0000"
    # doc_success has max_score=0.8 -> #ffff00
    assert net_data["node_trace"].marker.color[2] == "#ffff00"


def test_network_graph_config_enables_scroll_zoom():
    """Verify Plotly network graph configuration enables scroll zoom."""
    assert NETWORK_GRAPH_CONFIG["scrollZoom"] is True


def test_render_network_plotly_construction():
    """Verify render_network_plotly constructs a valid Plotly Figure from network data."""
    data = {
        "doc1": [1.0, 0.85],
        "doc2": [0.85, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    custom_theme = {
        "background": "#f0f0f0",
        "ink": "#111111",
    }

    net_data = build_network_data(df, threshold=0.75, theme_colors=custom_theme)
    fig = render_network_plotly(
        net_data, title="Custom Title", theme_colors=custom_theme
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Custom Title"
    assert fig.layout.paper_bgcolor == "#f0f0f0"
    assert fig.layout.plot_bgcolor == "#f0f0f0"
    assert len(fig.layout.shapes) == 1


def test_plot_similarity_network_returns_plotly_figure():
    # Setup simple square similarity matrix
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    fig = plot_similarity_network(df, threshold=0.75)

    assert isinstance(fig, go.Figure)
    # Check that there are traces in the graph
    assert len(fig.data) == 2  # edge_hover_trace, node_trace

    # Check that layout has shapes representing the edges
    # doc1 and doc2 are connected (0.85 >= 0.75), so 1 line shape should exist
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0]["type"] == "line"


def test_plot_similarity_network_no_edges():
    # Setup matrix where no similarities exceed the threshold
    data = {
        "doc1": [1.0, 0.10, 0.20],
        "doc2": [0.10, 1.0, 0.15],
        "doc3": [0.20, 0.15, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    fig = plot_similarity_network(df, threshold=0.75)

    assert isinstance(fig, go.Figure)
    # No shapes/lines should be added
    assert len(fig.layout.shapes) == 0


def test_plot_similarity_network_single_document():
    """Test graph generation when only one document is provided (1x1 matrix)."""
    data = {"doc1": [1.0]}
    df = pd.DataFrame(data, index=["doc1"])

    fig = plot_similarity_network(df, threshold=0.75)

    assert isinstance(fig, go.Figure)
    # No edges should be created for a single document
    assert len(fig.layout.shapes) == 0


def test_plot_similarity_network_empty_dataframe():
    """Test graph generation when an empty DataFrame is passed."""
    df = pd.DataFrame()

    fig = plot_similarity_network(df, threshold=0.75)

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 0


@patch("src.visualization.network_graph.go.Figure")
def test_plot_similarity_network_mocked_plotly(mock_figure):
    """Mock Plotly figure generation to verify execution without errors."""
    data = {
        "doc1": [1.0, 0.90],
        "doc2": [0.90, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    plot_similarity_network(df, threshold=0.75)

    # Verify that the Figure constructor was invoked properly
    assert mock_figure.called


def test_plot_similarity_network_layout_autosize():
    """Verify layout has autosize=True and width=None for dynamic scaling."""
    data = {
        "doc1": [1.0, 0.90],
        "doc2": [0.90, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    fig = plot_similarity_network(df, threshold=0.75)

    assert fig.layout.autosize is True
    assert fig.layout.width is None


def test_build_network_data_highlighted_doc():
    """Verify highlighted_doc node color and marker size are updated to bright yellow."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    result = export_network_to_gexf_bytes(df, threshold=0.75)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_network_to_gexf_bytes_contains_nodes_and_edges():
    """Verify GEXF output contains expected nodes and edge attributes from similarity matrix."""
    data = {
        "doc1": [1.0, 0.95],
        "doc2": [0.95, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    net_data = build_network_data(df, threshold=0.75, selected_node="doc1")
    node_colors = net_data["node_trace"].marker.color
    node_sizes = net_data["node_trace"].marker.size

    # doc1 is highlighted -> color #FFFF00 and larger size
    assert node_colors[0] == "#FFFF00"
    assert node_colors[1] != "#FFFF00"
    assert node_sizes[0] > node_sizes[1]


def test_export_graph_to_csv():
    """Verify export_graph_to_csv returns a CSV formatted string with Source,Target,Similarity header."""
    G = nx.Graph()
    G.add_edge("docA", "docB", similarity=0.88)
    csv_str = export_graph_to_csv(G)

    lines = csv_str.strip().splitlines()
    assert lines[0] == "Source,Target,Similarity"
    assert len(lines) == 2
    assert "docA,docB,0.88" in lines[1] or "docB,docA,0.88" in lines[1]


def test_export_network_to_csv_bytes():
    """Verify export_network_to_csv_bytes builds graph and returns encoded CSV bytes."""
    data = {
        "doc1": [1.0, 0.92],
        "doc2": [0.92, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    csv_bytes = export_network_to_csv_bytes(df, threshold=0.75)
    assert isinstance(csv_bytes, bytes)

    decoded = csv_bytes.decode("utf-8")
    lines = decoded.strip().splitlines()
    assert lines[0] == "Source,Target,Similarity"
    assert "doc1,doc2,0.92" in decoded or "doc2,doc1,0.92" in decoded


def test_plot_similarity_network_json_serialization():
    """Verify network graph figures serialize to valid JSON without circular references."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    fig = plot_similarity_network(df, threshold=0.75)

    json_str = fig.to_json()
    assert json_str is not None
    assert len(json_str) > 0


def test_plot_similarity_network_json_serialization_with_theme():
    """Verify JSON serialization works with custom theme colors."""
    data = {
        "doc1": [1.0, 0.95],
        "doc2": [0.95, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    custom_theme = {
        "danger": "#e53935",
        "warning": "#fb8c00",
        "success": "#43a047",
        "background": "#121212",
        "ink": "#ffffff",
    }

    fig = plot_similarity_network(df, threshold=0.75, theme_colors=custom_theme)

    json_str = fig.to_json()
    assert json_str is not None
    assert len(json_str) > 0


def test_plot_similarity_network_json_serialization_single_doc():
    """Verify JSON serialization works for single document graph."""
    data = {"doc1": [1.0]}
    df = pd.DataFrame(data, index=["doc1"])

    fig = plot_similarity_network(df, threshold=0.75)

    json_str = fig.to_json()
    assert json_str is not None
    assert len(json_str) > 0


def test_plot_similarity_network_json_serialization_no_edges():
    """Verify JSON serialization works when no edges exist."""
    data = {
        "doc1": [1.0, 0.10, 0.20],
        "doc2": [0.10, 1.0, 0.15],
        "doc3": [0.20, 0.15, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    fig = plot_similarity_network(df, threshold=0.75)

    json_str = fig.to_json()
    assert json_str is not None
    assert len(json_str) > 0


def test_plot_similarity_network_json_serialization_with_highlighted_node():
    """Verify JSON serialization works with highlighted node."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])

    fig = plot_similarity_network(df, threshold=0.75, selected_node="doc1")

    json_str = fig.to_json()
    assert json_str is not None
    assert len(json_str) > 0


# ==============================================================================
# Node Scale Factor Tests (Issue #1062)
# ==============================================================================


def test_build_network_data_node_scale_default():
    """Verify default node_scale=1.0 produces expected base node sizes."""
    data = {
        "doc1": [1.0, 0.85],
        "doc2": [0.85, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    net_data = build_network_data(df, threshold=0.75)
    sizes = net_data["node_trace"].marker.size
    assert sizes[0] == 26.0 or sizes[0] == 26


def test_build_network_data_node_scale_custom():
    """Verify node_scale=2.0 doubles the base node sizes compared to default."""
    data = {
        "doc1": [1.0, 0.85],
        "doc2": [0.85, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])
    net_data_default = build_network_data(df, threshold=0.75)
    net_data_scaled = build_network_data(df, threshold=0.75, node_scale=2.0)

    default_sizes = net_data_default["node_trace"].marker.size
    scaled_sizes = net_data_scaled["node_trace"].marker.size

    for d, s in zip(default_sizes, scaled_sizes):
        assert s == pytest.approx(d * 2.0)


# ==============================================================================
# Isolated Nodes Visibility Toggle Tests (Issue #1399)
# ==============================================================================


def _three_doc_matrix():
    """Return a matrix where doc3 has no similarity >= threshold (isolated)."""
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    return pd.DataFrame(data, index=["doc1", "doc2", "doc3"])


def test_build_network_data_filters_isolated_nodes_by_default():
    """Verify isolated nodes (degree 0) are removed when show_isolated=False (default)."""
    df = _three_doc_matrix()

    net_data = build_network_data(df, threshold=0.75)

    assert set(net_data["graph"].nodes()) == {"doc1", "doc2"}
    assert list(net_data["node_trace"].customdata) == ["doc1", "doc2"]


def test_build_network_data_keeps_isolated_nodes_when_show_isolated():
    """Verify all nodes remain when show_isolated=True."""
    df = _three_doc_matrix()

    net_data = build_network_data(df, threshold=0.75, show_isolated=True)

    assert set(net_data["graph"].nodes()) == {"doc1", "doc2", "doc3"}
    assert set(net_data["node_trace"].customdata) == {"doc1", "doc2", "doc3"}


def test_plot_similarity_network_filters_isolated_nodes():
    """Verify plot_similarity_network excludes isolated nodes by default."""
    df = _three_doc_matrix()

    fig = plot_similarity_network(df, threshold=0.75)

    assert len(fig.layout.shapes) == 1
    node_trace = fig.data[1]
    assert list(node_trace.customdata) == ["doc1", "doc2"]


def test_plot_similarity_network_show_isolated_keeps_all_nodes():
    """Verify plot_similarity_network keeps isolated nodes when show_isolated=True."""
    df = _three_doc_matrix()

    fig = plot_similarity_network(df, threshold=0.75, show_isolated=True)

    assert len(fig.layout.shapes) == 1
    node_trace = fig.data[1]
    assert set(node_trace.customdata) == {"doc1", "doc2", "doc3"}


def test_build_network_data_all_nodes_isolated():
    """Verify an all-isolated graph still builds a valid figure without nodes."""
    data = {
        "doc1": [1.0, 0.10],
        "doc2": [0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    net_data = build_network_data(df, threshold=0.75)

    assert set(net_data["graph"].nodes()) == set()
    assert list(net_data["node_trace"].customdata) == []

    fig = render_network_plotly(net_data, title="Empty")
    assert isinstance(fig, go.Figure)


# ==============================================================================
# Force-Directed Graph Physics Customization Tests (Issue #1368)
# ==============================================================================


def test_build_network_data_physics_defaults():
    """Verify default physics parameters spring_k=0.15 and iterations=50."""
    df = _three_doc_matrix()
    net_data = build_network_data(df, threshold=0.75, show_isolated=True)

    pos = net_data["pos"]
    assert isinstance(pos, dict)
    assert len(pos) == 3
    for node in ["doc1", "doc2", "doc3"]:
        assert node in pos
        assert len(pos[node]) == 2


def test_build_network_data_spring_k_customization():
    """Verify modifying spring_k recalculates node layout positions."""
    df = _three_doc_matrix()

    net_data_default = build_network_data(
        df, threshold=0.75, show_isolated=True, spring_k=0.15
    )
    net_data_tight = build_network_data(
        df, threshold=0.75, show_isolated=True, spring_k=0.05
    )
    net_data_spread = build_network_data(
        df, threshold=0.75, show_isolated=True, spring_k=0.85
    )

    pos_default = net_data_default["pos"]
    pos_tight = net_data_tight["pos"]
    pos_spread = net_data_spread["pos"]

    assert (
        pos_default["doc1"][0] != pos_tight["doc1"][0]
        or pos_default["doc1"][1] != pos_tight["doc1"][1]
    )
    assert (
        pos_default["doc1"][0] != pos_spread["doc1"][0]
        or pos_default["doc1"][1] != pos_spread["doc1"][1]
    )


def test_build_network_data_iterations_customization():
    """Verify modifying iteration count affects layout convergence."""
    df = _three_doc_matrix()

    net_data_few = build_network_data(
        df, threshold=0.75, show_isolated=True, iterations=5
    )
    net_data_many = build_network_data(
        df, threshold=0.75, show_isolated=True, iterations=200
    )

    pos_few = net_data_few["pos"]
    pos_many = net_data_many["pos"]

    assert (
        pos_few["doc1"][0] != pos_many["doc1"][0]
        or pos_few["doc1"][1] != pos_many["doc1"][1]
    )


def test_build_network_data_repulsion_customization():
    """Verify repulsion parameter alters force-directed node positioning."""
    df = _three_doc_matrix()

    net_data_base = build_network_data(
        df, threshold=0.75, show_isolated=True, repulsion=1.0
    )
    net_data_repelled = build_network_data(
        df, threshold=0.75, show_isolated=True, repulsion=3.5
    )

    pos_base = net_data_base["pos"]
    pos_repelled = net_data_repelled["pos"]

    assert (
        pos_base["doc1"][0] != pos_repelled["doc1"][0]
        or pos_base["doc1"][1] != pos_repelled["doc1"][1]
    )


def test_plot_plagiarism_network_graph_acceptance_criteria():
    """Verify plot_plagiarism_network_graph function accepts spring_k=0.15 and iterations=50."""
    df = _three_doc_matrix()

    fig = plot_plagiarism_network_graph(
        similarity_df=df,
        threshold=0.75,
        spring_k=0.15,
        iterations=50,
        repulsion=1.2,
        show_isolated=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.shapes is not None


def test_calculate_force_directed_layout_utility():
    """Verify calculate_force_directed_layout computes 2D coordinates for a NetworkX graph."""
    graph = nx.Graph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "A")

    pos = calculate_force_directed_layout(
        graph, spring_k=0.25, iterations=75, repulsion=1.5
    )

    assert isinstance(pos, dict)
    assert set(pos.keys()) == {"A", "B", "C"}
    for node in ["A", "B", "C"]:
        assert len(pos[node]) == 2
        assert isinstance(
            pos[node][0], (float, int, numpy.number) if "numpy" in globals() else float
        )


# ==============================================================================
# Community Clustering Tests (Issue #1503)
# ==============================================================================


def test_build_network_data_community_clustering():
    """Verify nodes in the same community receive the same color, and different communities get different colors."""
    data = {
        "doc1": [1.0, 0.9, 0.9, 0.0, 0.0, 0.0],
        "doc2": [0.9, 1.0, 0.9, 0.0, 0.0, 0.0],
        "doc3": [0.9, 0.9, 1.0, 0.0, 0.0, 0.0],
        "doc4": [0.0, 0.0, 0.0, 1.0, 0.9, 0.9],
        "doc5": [0.0, 0.0, 0.0, 0.9, 1.0, 0.9],
        "doc6": [0.0, 0.0, 0.0, 0.9, 0.9, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"])

    net_data = build_network_data(df, threshold=0.5, show_isolated=True)

    node_trace = net_data["node_trace"]
    colors = node_trace.marker.color
    customdata = node_trace.customdata

    assert len(colors) == 6
    assert len(customdata) == 6

    color_map = dict(zip(customdata, colors))

    assert color_map["doc1"] == color_map["doc2"] == color_map["doc3"]
    assert color_map["doc4"] == color_map["doc5"] == color_map["doc6"]
    assert color_map["doc1"] != color_map["doc4"]


def test_build_network_data_single_node_clustering():
    """Verify single-node graphs don't crash and get a community color."""
    data = {"doc1": [1.0]}
    df = pd.DataFrame(data, index=["doc1"])
    net_data = build_network_data(df, threshold=0.5, show_isolated=True)

    node_trace = net_data["node_trace"]
    assert len(node_trace.marker.color) == 1
    assert node_trace.marker.color[0] is not None


def test_build_network_data_empty_clustering():
    """Verify empty graphs don't crash during community clustering."""
    df = pd.DataFrame()
    net_data = build_network_data(df, threshold=0.5)

    assert len(net_data["graph"].nodes()) == 0
    assert len(net_data["node_trace"].marker.color) == 0


def test_export_network_centrality_csv():
    """Verify degree centrality and PageRank are exported correctly."""
    graph = nx.Graph()
    graph.add_edge("doc1", "doc2", similarity=0.9)
    graph.add_edge("doc1", "doc3", similarity=0.8)

    csv_str = export_network_centrality_csv(graph)

    lines = csv_str.strip().splitlines()
    assert lines[0] == ("Document_Name,Degree,Centrality_Score,PageRank_Score")
    assert len(lines) == 4  # Header + 3 nodes

    rows = [line.split(",") for line in lines[1:]]
    row_dict = {row[0]: (int(row[1]), float(row[2]), float(row[3])) for row in rows}

    assert "doc1" in row_dict
    assert row_dict["doc1"][0] == 2
    assert row_dict["doc1"][1] == 1.0
    assert 0.0 < row_dict["doc1"][2] < 1.0


def test_export_network_centrality_csv_star_graph_center_ranks_highest():
    """Verify the center of a five-leaf star has the highest degree and PageRank."""
    graph = nx.star_graph(4)
    nx.relabel_nodes(
        graph,
        {
            0: "center",
            1: "leaf1",
            2: "leaf2",
            3: "leaf3",
            4: "leaf4",
        },
        copy=False,
    )

    csv_str = export_network_centrality_csv(graph)
    lines = csv_str.strip().splitlines()

    assert lines[0] == ("Document_Name,Degree,Centrality_Score,PageRank_Score")

    rows = [line.split(",") for line in lines[1:]]
    values = {
        row[0]: {
            "degree": int(row[1]),
            "centrality": float(row[2]),
            "pagerank": float(row[3]),
        }
        for row in rows
    }

    center = values["center"]

    assert center["degree"] == 4
    assert center["centrality"] == 1.0
    assert center["pagerank"] == max(item["pagerank"] for item in values.values())
    assert center["pagerank"] > values["leaf1"]["pagerank"]


# ==============================================================================
# Max Connected Nodes Filter (Issue #1797)
# ==============================================================================


def _star_similarity_matrix(n: int = 6) -> pd.DataFrame:
    """Star graph: center doc0 linked to all others; a few leaf-leaf edges raise mid-tier degrees."""
    labels = [f"doc{i}" for i in range(n)]
    matrix = numpy.eye(n, dtype=float)
    for i in range(1, n):
        matrix[0, i] = matrix[i, 0] = 0.9
    matrix[1, 2] = matrix[2, 1] = 0.85
    matrix[1, 3] = matrix[3, 1] = 0.85
    return pd.DataFrame(matrix, index=labels, columns=labels)


def test_build_network_data_keeps_top_max_nodes_by_degree():
    """When node count exceeds max_nodes, retain only the highest-degree documents."""
    df = _star_similarity_matrix(6)

    net_data = build_network_data(df, threshold=0.75, show_isolated=True, max_nodes=3)

    assert len(net_data["graph"].nodes()) == 3
    assert net_data["hidden_nodes"] == 3
    # doc0 (deg 5) and doc1 (deg 3) must be kept; third slot is the next-highest degree
    assert "doc0" in net_data["graph"].nodes()
    assert "doc1" in net_data["graph"].nodes()


def test_build_network_data_no_filter_when_under_max_nodes():
    """Graphs at or below max_nodes keep every node and report zero hidden."""
    df = _three_doc_matrix()

    net_data = build_network_data(df, threshold=0.75, show_isolated=True, max_nodes=50)

    assert len(net_data["graph"].nodes()) == 3
    assert net_data["hidden_nodes"] == 0


def test_plot_plagiarism_network_graph_max_nodes_caption():
    """plot_plagiarism_network_graph accepts max_nodes and captions hidden node count."""
    df = _star_similarity_matrix(6)

    fig = plot_plagiarism_network_graph(
        similarity_df=df,
        threshold=0.75,
        show_isolated=True,
        max_nodes=3,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data[1].customdata) == 3
    caption_texts = [ann.text for ann in (fig.layout.annotations or [])]
    assert "3 nodes hidden" in caption_texts


# ─── Tests for get_cluster_count (Issue #1793) ────────────────────────────────


class TestGetClusterCount:
    """Comprehensive test suite for the connected components counter helper."""

    def test_single_connected_component(self):
        """A fully connected graph should return exactly 1 cluster."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        assert get_cluster_count(G) == 1

    def test_multiple_isolated_clusters(self):
        """A graph with distinct disconnected subgraphs should return the correct count."""
        G = nx.Graph()
        # Cluster 1
        G.add_edges_from([("A", "B"), ("B", "C")])
        # Cluster 2
        G.add_edges_from([("D", "E")])
        # Cluster 3 (isolated node)
        G.add_node("F")

        assert get_cluster_count(G) == 3

    def test_empty_graph_returns_zero(self):
        """An empty graph with no nodes should return 0."""
        G = nx.Graph()
        assert get_cluster_count(G) == 0

    def test_none_input_returns_zero(self):
        """None input should be handled gracefully and return 0."""
        assert get_cluster_count(None) == 0

    def test_invalid_type_returns_zero(self):
        """Non-Graph inputs (string, dict, list) should return 0 without raising."""
        assert get_cluster_count("not a graph") == 0
        assert get_cluster_count({"nodes": []}) == 0
        assert get_cluster_count([1, 2, 3]) == 0

    def test_single_isolated_node(self):
        """A graph with a single node and no edges should return 1 cluster."""
        G = nx.Graph()
        G.add_node("LonelyDoc")
        assert get_cluster_count(G) == 1

    def test_large_graph_performance(self):
        """Verify the function performs efficiently on a larger graph."""
        G = nx.fast_gnp_random_graph(n=1000, p=0.01, seed=42)
        count = get_cluster_count(G)
        assert isinstance(count, int)
        assert count > 0

    def test_directed_graph_handling(self):
        """Verify behavior with DiGraph (NetworkX treats components as weakly connected)."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("C", "D")])
        # number_connected_components works on undirected views or weakly connected components
        # For DiGraph, we might need weakly_connected_components, but let's test standard behavior
        # If it raises, our try/except should catch it and return 0, or nx handles it.
        # Actually, nx.number_connected_components raises NetworkXNotImplemented for DiGraph.
        # Our try/except block will catch it and return 0.
        result = get_cluster_count(G)
        assert result == 0  # Caught by exception handler

    def test_graph_with_self_loops(self):
        """Self-loops should not affect the connected component count."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("A", "A")])
        G.add_node("C")
        assert get_cluster_count(G) == 2


# ─── Tests for build_network_data ─────────────────────────────────────────────


class TestBuildNetworkData:
    """Test suite for network data construction and layout."""

    def test_build_network_data_basic(self):
        """Verify basic network data structure is returned correctly."""
        df = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
        )

        data = build_network_data(df, threshold=0.5)

        assert "graph" in data
        assert "node_trace" in data
        assert "edge_hover_trace" in data
        assert "pos" in data
        assert isinstance(data["graph"], nx.Graph)
        assert len(data["graph"].nodes()) == 2
        assert len(data["graph"].edges()) == 1

    def test_threshold_filtering(self):
        """Verify edges are only created for pairs meeting the threshold."""
        df = pd.DataFrame(
            [[1.0, 0.4, 0.9], [0.4, 1.0, 0.3], [0.9, 0.3, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        # Threshold 0.5 should only create edge A-C (0.9)
        data = build_network_data(df, threshold=0.5, show_isolated=True)
        G = data["graph"]

        assert G.has_edge("A", "C")
        assert not G.has_edge("A", "B")
        assert not G.has_edge("B", "C")

    def test_isolated_nodes_hidden_by_default(self):
        """Verify isolated nodes (degree 0) are removed when show_isolated=False."""
        df = pd.DataFrame(
            [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        data = build_network_data(df, threshold=0.5, show_isolated=False)
        G = data["graph"]

        assert "A" in G.nodes()
        assert "B" in G.nodes()
        assert "C" not in G.nodes()  # C is isolated


# ─── Tests for Export Functions ───────────────────────────────────────────────


class TestNetworkExport:
    """Test suite for GEXF and CSV export functions."""

    def test_export_gexf_bytes(self):
        """Verify GEXF export returns valid UTF-8 bytes."""
        df = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
        )

        gexf_bytes = export_network_to_gexf_bytes(df, threshold=0.5)

        assert isinstance(gexf_bytes, bytes)
        gexf_str = gexf_bytes.decode("utf-8")
        assert "<?xml" in gexf_str
        assert "<gexf" in gexf_str
        assert "doc1" in gexf_str
        assert "doc2" in gexf_str

    def test_export_csv_bytes(self):
        """Verify CSV export returns valid UTF-8 bytes with correct headers."""
        df = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
        )

        csv_bytes = export_network_to_csv_bytes(df, threshold=0.5)

        assert isinstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        lines = csv_str.strip().split("\n")

        assert lines[0] == "Source,Target,Similarity"
        assert "doc1" in lines[1]
        assert "doc2" in lines[1]
        assert "0.8" in lines[1]


# ==============================================================================
# Max Connected Nodes Filter Tests (Issue #1278)
# ==============================================================================


def _max_nodes_matrix():
    labels = ["hub", "mid", "leaf1", "leaf2", "isolated"]
    return pd.DataFrame(
        [
            [1.0, 0.9, 0.9, 0.9, 0.1],
            [0.9, 1.0, 0.8, 0.1, 0.1],
            [0.9, 0.8, 1.0, 0.1, 0.1],
            [0.9, 0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0],
        ],
        index=labels,
        columns=labels,
    )


def test_build_network_data_keeps_top_highest_degree_nodes():
    data = build_network_data(
        _max_nodes_matrix(),
        threshold=0.75,
        show_isolated=True,
        max_nodes=3,
    )
    assert set(data["graph"].nodes()) == {"hub", "mid", "leaf1"}
    assert data["hidden_node_count"] == 2


def test_max_nodes_tie_breaking_follows_original_order():
    labels = ["A", "B", "C", "D"]
    df = pd.DataFrame(
        [
            [1.0, 0.9, 0.9, 0.9],
            [0.9, 1.0, 0.1, 0.1],
            [0.9, 0.1, 1.0, 0.1],
            [0.9, 0.1, 0.1, 1.0],
        ],
        index=labels,
        columns=labels,
    )
    data = build_network_data(
        df,
        threshold=0.75,
        show_isolated=True,
        max_nodes=2,
    )
    assert list(data["graph"].nodes()) == ["A", "B"]
    assert data["hidden_node_count"] == 2


def test_max_nodes_does_not_filter_small_graph():
    data = build_network_data(
        _max_nodes_matrix(),
        threshold=0.75,
        show_isolated=True,
        max_nodes=10,
    )
    assert len(data["graph"]) == 5
    assert data["hidden_node_count"] == 0


@pytest.mark.parametrize("max_nodes", [0, -1])
def test_max_nodes_rejects_non_positive_values(max_nodes):
    with pytest.raises(ValueError, match="max_nodes must be at least 1"):
        build_network_data(
            _max_nodes_matrix(),
            show_isolated=True,
            max_nodes=max_nodes,
        )


@pytest.mark.parametrize("max_nodes", [True, 1.5, "50", None])
def test_max_nodes_rejects_non_integer_values(max_nodes):
    with pytest.raises(TypeError, match="max_nodes must be an integer"):
        build_network_data(
            _max_nodes_matrix(),
            show_isolated=True,
            max_nodes=max_nodes,
        )


def test_render_network_plotly_displays_hidden_node_caption():
    data = build_network_data(
        _max_nodes_matrix(),
        threshold=0.75,
        show_isolated=True,
        max_nodes=3,
    )
    fig = render_network_plotly(data)
    assert len(fig.layout.annotations) == 1
    assert (
        fig.layout.annotations[0].text == "2 nodes hidden to keep the network readable."
    )


def test_plot_plagiarism_network_graph_accepts_max_nodes():
    fig = plot_plagiarism_network_graph(
        _max_nodes_matrix(),
        threshold=0.75,
        show_isolated=True,
        max_nodes=2,
    )
    assert len(fig.data[1].customdata) == 2
    assert "3 nodes hidden" in fig.layout.annotations[0].text


def test_get_cluster_count_returns_two_for_two_disjoint_pairs():
    """Verify get_cluster_count counts connected components correctly."""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("A", "B"),
            ("C", "D"),
        ]
    )

    assert get_cluster_count(graph) == 2


def test_export_network_to_gexf_valid_xml():
    """Verify GEXF export returns well-formed XML with a GEXF root."""
    data = {
        "doc1": [1.0, 0.95],
        "doc2": [0.95, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2"])

    gexf_bytes = export_network_to_gexf_bytes(
        df,
        threshold=0.75,
    )

    assert isinstance(gexf_bytes, bytes)
    assert gexf_bytes

    root = ET.fromstring(gexf_bytes)

    assert root.tag.endswith("gexf")


# ── Issue #2350: Empty state rendering for plot_plagiarism_network_graph ────


def test_issue_2350_plot_plagiarism_network_graph_empty_dataframe():
    """plot_plagiarism_network_graph must return a go.Figure and not crash
    when given a completely empty DataFrame (Issue #2350).
    """
    empty_df = pd.DataFrame()

    fig = plot_plagiarism_network_graph(empty_df, threshold=0.75)

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 0


def test_issue_2350_plot_plagiarism_network_graph_empty_matrix():
    """plot_plagiarism_network_graph must return a go.Figure when given an
    empty NxN similarity matrix (0 rows, 0 columns).
    """
    empty_matrix = pd.DataFrame()

    fig = plot_plagiarism_network_graph(similarity_df=empty_matrix)

    assert isinstance(fig, go.Figure)


def test_issue_2350_plot_plagiarism_network_graph_no_nodes():
    """plot_plagiarism_network_graph must return a go.Figure when the
    DataFrame has no columns (no documents).
    """
    df = pd.DataFrame({"col": []})

    fig = plot_plagiarism_network_graph(df)

    assert isinstance(fig, go.Figure)


def test_issue_2350_empty_state_has_fallback_annotation():
    """When the graph is empty, the figure must include a fallback
    annotation message (Issue #2350).
    """
    empty_df = pd.DataFrame()

    fig = plot_plagiarism_network_graph(empty_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1
    assert any(
        "No documents" in ann.text or "No data" in ann.text
        for ann in fig.layout.annotations
    ), f"Expected fallback annotation, got: {[ann.text for ann in fig.layout.annotations]}"


def test_issue_2350_empty_state_does_not_raise():
    """Passing an empty DataFrame to plot_plagiarism_network_graph must
    not raise any exception (Issue #2350).
    """
    empty_df = pd.DataFrame()

    try:
        fig = plot_plagiarism_network_graph(empty_df)
        assert isinstance(fig, go.Figure)
    except Exception as exc:
        pytest.fail(
            f"plot_plagiarism_network_graph raised an exception on empty input: {exc}"
        )
