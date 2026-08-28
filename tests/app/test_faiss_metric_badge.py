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

"""Unit tests for FAISS distance metric badge helper functions."""

from unittest.mock import MagicMock, patch

import faiss

from app.components.faiss_results import (
    get_faiss_metric_label,
    render_faiss_metric_badge,
)


def test_get_faiss_metric_label_uninitialized():
    """Verify fallback to 'Default' when FAISS index is None."""
    assert get_faiss_metric_label(None) == "Default"


def test_get_faiss_metric_label_inner_product():
    """Verify detection of Inner Product metric."""
    mock_index = MagicMock()
    mock_index.metric_type = faiss.METRIC_INNER_PRODUCT
    assert get_faiss_metric_label(mock_index) == "Inner Product (Cosine)"


def test_get_faiss_metric_label_l2():
    """Verify detection of L2 metric."""
    mock_index = MagicMock()
    mock_index.metric_type = faiss.METRIC_L2
    assert get_faiss_metric_label(mock_index) == "L2 (Euclidean)"


def test_render_faiss_metric_badge():
    """Verify rendering of metric badge in sidebar caption."""
    with patch("streamlit.sidebar.caption") as mock_caption:
        badge_text = render_faiss_metric_badge(None)
        assert badge_text == "Metric: Default"
        mock_caption.assert_called_once_with("🎯 Metric: Default")
