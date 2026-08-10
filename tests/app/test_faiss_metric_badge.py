"""Unit tests for FAISS distance metric badge helper functions."""

from unittest.mock import MagicMock, patch
import faiss
from app.components.faiss_results import get_faiss_metric_label, render_faiss_metric_badge


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
