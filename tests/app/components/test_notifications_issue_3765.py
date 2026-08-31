"""Tests for show_notification (Issue #3765)."""

from unittest.mock import patch

from app.components.notifications import show_notification


@patch("app.components.notifications.st")
def test_show_notification_uses_toast_with_icons(mock_st):
    cases = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
    }
    for kind, icon in cases.items():
        mock_st.reset_mock()
        show_notification(f"msg-{kind}", type=kind)
        mock_st.toast.assert_called_once_with(f"msg-{kind}", icon=icon)
        mock_st.info.assert_not_called()


@patch("app.components.notifications.st")
def test_show_notification_defaults_to_info(mock_st):
    show_notification("hello")
    mock_st.toast.assert_called_once_with("hello", icon="ℹ️")


@patch("app.components.notifications.st")
def test_show_notification_falls_back_to_info(mock_st):
    mock_st.toast.side_effect = Exception("toast unavailable")
    show_notification("saved", type="success")
    mock_st.info.assert_called_once_with("saved")
