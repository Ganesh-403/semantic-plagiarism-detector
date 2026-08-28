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
    assert 'key="load_most_plagiarized_documents"' in source
    assert 'key="load_similarity_distribution"' in source


def test_duplicate_eager_heatmap_render_is_removed():
    source = APP_PATH.read_text(encoding="utf-8")

    assert source.count("heatmap_fig = plot_similarity_heatmap(") == 0


def test_lazy_controls_use_collapsed_expanders():
    source = APP_PATH.read_text(encoding="utf-8")

    assert source.count("expanded=False") >= 5
