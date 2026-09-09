"""Exercise the recovery forms through real token creation and consumption."""

import threading

import pytest
from streamlit.testing.v1 import AppTest

from src.db import auth
from src.security import password_recovery

EMAIL = "learner@example.com"
OLD_PASSWORD = "Recovery-Old-Password!984"  # pragma: allowlist secret
NEW_PASSWORD = "Recovery-New-Password!985"  # pragma: allowlist secret


@pytest.fixture
def recovery_form(mock_db, monkeypatch):
    auth.add_user(EMAIL, OLD_PASSWORD, role="teacher")
    messages = []
    finished = threading.Event()
    original = password_recovery.request_password_reset

    def request(email):
        try:
            original(email)
        finally:
            finished.set()

    monkeypatch.setattr(password_recovery, "mail_is_configured", lambda: True)
    monkeypatch.setattr(password_recovery, "send_recovery_email", lambda email, token=None: messages.append((email, token)))
    monkeypatch.setattr(password_recovery, "request_password_reset", request)
    at = AppTest.from_string(
        "from app.components.password_management import render_password_recovery\nrender_password_recovery()",
        default_timeout=30,
    ).run()
    assert not at.exception
    return at, messages, finished


def fill(at, label, value):
    next(widget for widget in at.text_input if widget.label == label).set_value(value)


def submit(at, label):
    next(button for button in at.button if button.label == label).click().run()
    assert not at.exception


def reset(at, token, password, confirmation):
    fill(at, "Reset token", token)
    fill(at, "Recovery new password", password)
    fill(at, "Confirm recovery password", confirmation)
    submit(at, "Reset password")


def test_recovery_request_and_single_use_reset_round_trip(recovery_form):
    at, messages, finished = recovery_form
    fill(at, "Account email", EMAIL)
    submit(at, "Email reset instructions")
    assert finished.wait(5), "Recovery worker did not finish"
    assert [item.value for item in at.info] == [password_recovery.RESET_MESSAGE]
    assert len(messages) == 1 and messages[0][0] == EMAIL
    token = messages[0][1]

    # A repeated request in the same browser still receives the generic response.
    fill(at, "Account email", EMAIL)
    submit(at, "Email reset instructions")
    assert len(messages) == 1
    reset(at, token, NEW_PASSWORD, "different")
    assert any("do not match" in item.value for item in at.error)
    reset(at, token, "weak", "weak")
    assert at.error and not at.success
    reset(at, token, NEW_PASSWORD, NEW_PASSWORD)
    assert any("Password updated" in item.value for item in at.success)
    assert auth.verify_user(EMAIL, NEW_PASSWORD)
    assert messages[-1] == (EMAIL, None)
    reset(at, token, NEW_PASSWORD, NEW_PASSWORD)
    assert any("invalid or expired" in item.value for item in at.error)


def test_invalid_email_is_rejected_before_starting_worker(recovery_form):
    at, messages, finished = recovery_form
    fill(at, "Account email", "not-an-email")
    submit(at, "Email reset instructions")
    assert any("valid email" in item.value for item in at.error)
    assert not messages and not finished.is_set()


def test_unknown_account_has_same_generic_response(recovery_form):
    at, messages, finished = recovery_form
    fill(at, "Account email", "unknown@example.com")
    submit(at, "Email reset instructions")
    assert finished.wait(5)
    assert [item.value for item in at.info] == [password_recovery.RESET_MESSAGE]
    assert not messages and not at.error


def test_unconfigured_mail_directs_user_to_administrator(monkeypatch):
    monkeypatch.setattr(password_recovery, "mail_is_configured", lambda: False)
    at = AppTest.from_string(
        "from app.components.password_management import render_password_recovery\nrender_password_recovery()",
        default_timeout=30,
    ).run()
    assert not at.exception
    assert any("Contact your administrator" in item.value for item in at.info)
    assert all(button.label != "Email reset instructions" for button in at.button)
