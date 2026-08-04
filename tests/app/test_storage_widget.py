"""Integration/UI tests for Storage Space Used widget in app/streamlit_app.py."""

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_storage_widget_imports():
    """Verify calculate_storage_usage is imported in app/streamlit_app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "from src.utils.storage_metrics import calculate_storage_usage" in source


def test_storage_widget_inside_admin_sidebar_block():
    """Verify Storage Space Used widget is placed inside the admin user block."""
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'if user_role == "admin":' in source
    assert '### 💾 Storage Space Used' in source
    assert 'label="Total Storage Used"' in source
    assert 'calculate_storage_usage()' in source

    admin_pos = source.index('if user_role == "admin":')
    storage_widget_pos = source.index('### 💾 Storage Space Used')
    assert storage_widget_pos > admin_pos
