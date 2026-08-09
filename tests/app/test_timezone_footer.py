"""Unit tests for render_timezone_footer in app/theme.py."""

from unittest.mock import patch
from app.theme import render_timezone_footer


def test_render_timezone_footer():
    """Verify that render_timezone_footer displays server UTC time and timezone label."""
    with patch("streamlit.sidebar.caption") as mock_caption:
        caption_text = render_timezone_footer()
        assert "Server Time:" in caption_text
        assert "UTC" in caption_text
        mock_caption.assert_called_once()
        args, _ = mock_caption.call_args
        assert "Server Time:" in args[0]
        assert "UTC" in args[0]
