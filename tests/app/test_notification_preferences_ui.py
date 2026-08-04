from pathlib import Path


APP_PATH = Path("app/streamlit_app.py")


def test_settings_render_notification_toggles():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "### 🔔 Notification Preferences" in source
    assert '"📧 Email notifications"' in source
    assert '"🔗 Webhook notifications"' in source


def test_settings_load_persisted_preferences():
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        "persisted_notifications = "
        "get_notification_preferences("
    ) in source
    assert (
        'st.session_state["email_notifications_toggle"]'
        in source
    )
    assert (
        'st.session_state["webhook_notifications_toggle"]'
        in source
    )


def test_settings_save_preferences_to_database():
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        "update_notification_preferences("
        in source
    )
    assert (
        'key="save_notification_preferences"'
        in source
    )
    assert (
        "Notification preferences saved."
        in source
    )


def test_notification_section_is_available_before_admin_only_settings():
    source = APP_PATH.read_text(encoding="utf-8")

    notification_position = source.index(
        "### 🔔 Notification Preferences"
    )
    admin_position = source.index(
        "### ⚙️ Advanced Configuration",
        notification_position,
    )

    assert notification_position < admin_position
