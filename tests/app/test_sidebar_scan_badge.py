"""
tests/app/test_sidebar_scan_badge.py
------------------------------------
Unit tests for Total Scan Count Badge in Sidebar Header (#1725).
"""

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_sidebar_scan_badge_exists():
    """Verify that Total Scans Processed badge is added to the sidebar header."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Total Scans Processed:" in source
    assert "get_upload_count()" in source
