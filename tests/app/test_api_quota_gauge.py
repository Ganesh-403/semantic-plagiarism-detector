"""
tests/app/test_api_quota_gauge.py
---------------------------------
Unit and integration tests for Collapsible API Rate Limit Usage Gauge Component.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import streamlit as st

from app.components.api_quota_gauge import render_api_quota_gauge

APP_PATH = Path("app/streamlit_app.py")


def test_api_quota_gauge_integration_in_app():
    """Verify render_api_quota_gauge is imported and called in app/streamlit_app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "from app.components.api_quota_gauge import render_api_quota_gauge" in source
    assert "render_api_quota_gauge()" in source


@patch("streamlit.sidebar.expander")
@patch("streamlit.progress")
@patch("streamlit.caption")
@patch("streamlit.markdown")
def test_render_api_quota_gauge_default(
    mock_markdown, mock_caption, mock_progress, mock_expander
):
    """Verify render_api_quota_gauge behavior with default/normal values (<90%)."""
    # Mock expander as context manager
    mock_expander_context = MagicMock()
    mock_expander.return_value = mock_expander_context

    with patch.dict(st.session_state, {"api_quota_consumed": 850, "api_quota_limit": 1000}):
        render_api_quota_gauge()

        # Check expander was opened with correct title
        mock_expander.assert_called_once_with("📊 API Quota Usage", expanded=True)

        # Check st.progress called with 0.85
        mock_progress.assert_called_once_with(0.85)

        # Check caption rendered correctly
        mock_caption.assert_called_once_with("API Quota: 850 / 1000 requests (85%)")

        # Check style override not called (as 85% is < 90%)
        mock_markdown.assert_not_called()


@patch("streamlit.sidebar.expander")
@patch("streamlit.progress")
@patch("streamlit.caption")
@patch("streamlit.markdown")
def test_render_api_quota_gauge_red_color_exceeds_90(
    mock_markdown, mock_caption, mock_progress, mock_expander
):
    """Verify progress bar turns red when quota exceeds 90%."""
    mock_expander_context = MagicMock()
    mock_expander.return_value = mock_expander_context

    with patch.dict(st.session_state, {"api_quota_consumed": 950, "api_quota_limit": 1000}):
        render_api_quota_gauge()

        # Check progress called with 0.95
        mock_progress.assert_called_once_with(0.95)

        # Check caption rendered correctly
        mock_caption.assert_called_once_with("API Quota: 950 / 1000 requests (95%)")

        # Check styling override called to turn progress bar color to red
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        assert "background-color: #EF4444 !important;" in args[0]
