from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from src.db import auth

PAGE = Path(__file__).resolve().parents[2] / "app/pages/2_Settings.py"


@pytest.fixture
def settings(mock_db):
    auth.add_user("preferences_user", "TestPassword123!", "teacher")
    at = AppTest.from_file(str(PAGE))
    at.session_state["authenticated"] = True
    at.session_state["username"] = "preferences_user"
    at.session_state["role"] = "teacher"
    return at


def test_settings_load_and_save_preferences(settings):
    auth.update_notification_preferences("preferences_user", False, True)
    settings.run()
    assert not settings.exception
    assert [toggle.value for toggle in settings.toggle] == [False, True]
    settings.toggle[0].set_value(True)
    settings.toggle[1].set_value(False)
    settings.button(key="save_notification_preferences").click().run()
    assert not settings.exception
    assert auth.get_notification_preferences("preferences_user") == {
        "email_notifications": True, "webhook_notifications": False,
    }
    assert any("saved" in item.value for item in settings.success)


def test_preferences_available_without_admin_controls(settings):
    settings.run()
    assert len(settings.toggle) == 2
    assert not settings.slider
    assert not settings.download_button


def test_signed_out_user_cannot_change_preferences():
    at = AppTest.from_file(str(PAGE)).run()
    assert not at.exception
    assert not at.toggle
    assert not at.button


def test_admin_can_access_preferences_and_configuration(settings):
    settings.session_state["role"] = "admin"
    settings.run()
    assert not settings.exception
    assert len(settings.toggle) == 2
    assert len(settings.slider) == 5
