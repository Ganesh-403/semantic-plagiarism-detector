"""
tests/app/test_db_schema_status_ui.py
-------------------------------------
Unit tests for Refresh Database Schema Status Button in System Settings (#1729).
"""

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_db_schema_status_ui_elements():
    """Verify that Check Database Schema button and message logic exist in settings tab."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Check Database Schema" in source
    assert "get_user_version" in source
    assert 'db_schema_status_msg' in source
    assert 'Corpus Schema: v' in source
    assert 'Auth Schema: v' in source
    assert 'st.toast(' in source
