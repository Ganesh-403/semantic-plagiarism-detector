"""Unit tests for storage quota progress bar component in app/components/storage_quota.py."""

from unittest.mock import patch

from app.components.storage_quota import (
    get_total_corpus_storage_bytes,
    render_storage_quota_progress,
)


def test_get_total_corpus_storage_bytes():
    """Verify calculation of total corpus storage bytes."""
    with patch("app.components.storage_quota.calculate_storage_usage") as mock_calc:
        mock_calc.return_value = {"total_bytes": 1288490188}  # ~1.2 GB
        total = get_total_corpus_storage_bytes()
        assert total >= 1288490188


def test_render_storage_quota_progress():
    """Verify st.progress and caption rendered with 10GB limit."""
    with (
        patch("app.components.storage_quota.calculate_storage_usage") as mock_calc,
        patch("streamlit.progress") as mock_progress,
        patch("streamlit.caption") as mock_caption,
    ):
        mock_calc.return_value = {"total_bytes": 1288490188}  # ~1.2 GB
        res = render_storage_quota_progress(limit_gb=10.0)

        assert res["total_gb"] > 1.1
        assert res["limit_gb"] == 10.0
        assert "Storage Used: 1.2 GB / 10.0 GB" in res["caption"]
        mock_progress.assert_called_once()
        mock_caption.assert_called_once()
