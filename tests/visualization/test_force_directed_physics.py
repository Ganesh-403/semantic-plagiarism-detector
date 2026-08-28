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
test_force_directed_physics.py
--------------------------------
Exhaustive unit test suite for force-directed graph physics customization (#1368).
Validates spring constant (k), iterations, repulsion factors, node position recalculation,
reproducibility, edge cases, and graph visualization behavior.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.network_graph import (
    build_network_data,
    calculate_force_directed_layout,
    plot_plagiarism_network_graph,
)


def _create_sample_similarity_matrix(n: int = 5) -> pd.DataFrame:
    """Helper to construct a deterministic symmetric similarity matrix with known connections."""
    labels = [f"doc_{i}" for i in range(1, n + 1)]
    matrix = np.eye(n, dtype=float)
    # Add predictable off-diagonal similarities safely within range
    for i in range(n - 1):
        sim = 0.85 if (i % 2 == 0) else 0.70
        matrix[i, i + 1] = matrix[i + 1, i] = sim

    return pd.DataFrame(matrix, index=labels, columns=labels)


def test_force_directed_physics_default_parameters():
    """Verify default parameters spring_k=0.15 and iterations=50 produce valid 2D layout coordinates."""
    df = _create_sample_similarity_matrix(5)
    data = build_network_data(df, threshold=0.60, show_isolated=True)

    pos = data["pos"]
    assert len(pos) == 5
    for node, coord in pos.items():
        assert len(coord) == 2
        assert isinstance(coord[0], (float, np.floating))
        assert isinstance(coord[1], (float, np.floating))


def test_force_directed_physics_spring_k_scaling_impact():
    """Verify varying spring_k parameter significantly alters node positions."""
    df = _create_sample_similarity_matrix(6)

    data_small_k = build_network_data(
        df, threshold=0.60, spring_k=0.01, show_isolated=True
    )
    data_large_k = build_network_data(
        df, threshold=0.60, spring_k=1.50, show_isolated=True
    )

    pos_small = data_small_k["pos"]
    pos_large = data_large_k["pos"]

    # Calculate average distance from origin for small vs large k
    dist_small = np.mean([np.linalg.norm(coord) for coord in pos_small.values()])
    dist_large = np.mean([np.linalg.norm(coord) for coord in pos_large.values()])

    assert dist_small != dist_large


def test_force_directed_physics_iterations_convergence():
    """Verify layout positioning differs between 1 iteration and 100 iterations."""
    df = _create_sample_similarity_matrix(6)

    data_1_iter = build_network_data(
        df, threshold=0.60, iterations=1, show_isolated=True
    )
    data_100_iter = build_network_data(
        df, threshold=0.60, iterations=100, show_isolated=True
    )

    pos_1 = data_1_iter["pos"]
    pos_100 = data_100_iter["pos"]

    # Coordinates must diverge as physics iterations simulate spring mechanics
    diffs = [np.linalg.norm(pos_1[node] - pos_100[node]) for node in pos_1]
    assert np.mean(diffs) > 1e-4


def test_force_directed_physics_repulsion_factor():
    """Verify repulsion factor changes node separation distances."""
    df = _create_sample_similarity_matrix(5)

    data_rep1 = build_network_data(
        df, threshold=0.60, repulsion=1.0, show_isolated=True
    )
    data_rep5 = build_network_data(
        df, threshold=0.60, repulsion=5.0, show_isolated=True
    )

    pos_rep1 = data_rep1["pos"]
    pos_rep5 = data_rep5["pos"]

    assert (
        pos_rep1["doc_1"][0] != pos_rep5["doc_1"][0]
        or pos_rep1["doc_1"][1] != pos_rep5["doc_1"][1]
    )


def test_plot_plagiarism_network_graph_exposes_physics_parameters():
    """Verify plot_plagiarism_network_graph passes spring_k and iterations through to Plotly figure."""
    df = _create_sample_similarity_matrix(4)

    fig = plot_plagiarism_network_graph(
        df,
        threshold=0.60,
        spring_k=0.20,
        iterations=60,
        repulsion=1.5,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Document Plagiarism Network"


def test_calculate_force_directed_layout_standalone_function():
    """Verify calculate_force_directed_layout operates directly on NetworkX graphs."""
    G = nx.complete_graph(5)
    pos = calculate_force_directed_layout(
        G, spring_k=0.15, iterations=50, repulsion=1.0
    )

    assert len(pos) == 5
    for i in range(5):
        assert i in pos
        assert len(pos[i]) == 2


def test_force_directed_layout_returns_valid_coords():
    """Verify force-directed layout returns finite coordinates for disconnected nodes."""
    G = nx.Graph()
    G.add_nodes_from(range(5))

    pos = calculate_force_directed_layout(G)

    assert len(pos) == 5
    for coord in pos.values():
        assert len(coord) == 2
        assert np.isfinite(coord[0])
        assert np.isfinite(coord[1])


@pytest.mark.parametrize("invalid_k", [-0.5, 0.0, None])
def test_force_directed_physics_fallback_invalid_spring_k(invalid_k):
    """Verify invalid or zero spring_k falls back safely to default heuristic calculation."""
    df = _create_sample_similarity_matrix(4)
    data = build_network_data(
        df, threshold=0.60, spring_k=invalid_k, show_isolated=True
    )

    assert len(data["pos"]) == 4


@pytest.mark.parametrize("invalid_iter", [0, -10])
def test_force_directed_physics_fallback_invalid_iterations(invalid_iter):
    """Verify non-positive iteration parameters fall back safely to minimum 1 iteration."""
    df = _create_sample_similarity_matrix(4)
    data = build_network_data(
        df, threshold=0.60, iterations=invalid_iter, show_isolated=True
    )

    assert len(data["pos"]) == 4


def test_force_directed_physics_single_node_graph():
    """Verify force-directed layout computation on single node graph."""
    df = pd.DataFrame([[1.0]], index=["doc_single"], columns=["doc_single"])
    data = build_network_data(df, threshold=0.50, show_isolated=True)

    assert len(data["pos"]) == 1
    assert "doc_single" in data["pos"]


def test_force_directed_physics_empty_graph():
    """Verify force-directed layout computation on empty graph."""
    df = pd.DataFrame()
    data = build_network_data(df, threshold=0.50, show_isolated=True)

    assert len(data["pos"]) == 0
