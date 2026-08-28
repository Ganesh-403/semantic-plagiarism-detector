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

from pathlib import Path

SOURCE = Path("src/visualization/network_graph.py")
TESTS = Path("tests/visualization/test_network_graph.py")


def test_plot_api_has_required_default():
    source = SOURCE.read_text(encoding="utf-8")
    section = source[source.index("def plot_plagiarism_network_graph(") :]
    signature = section[: section.index(") -> go.Figure:")]
    assert "max_nodes: int = 50" in signature


def test_degree_filter_and_hidden_caption_are_implemented():
    source = SOURCE.read_text(encoding="utf-8")
    assert "-G.degree(node)" in source
    assert "hidden_node_count" in source
    assert "fig.add_annotation(" in source


def test_required_regression_tests_exist():
    tests = TESTS.read_text(encoding="utf-8")
    assert "test_build_network_data_keeps_top_highest_degree_nodes" in tests
    assert "test_render_network_plotly_displays_hidden_node_caption" in tests
