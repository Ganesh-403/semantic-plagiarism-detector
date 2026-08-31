"""
test_sso_csrf_state_validation_issue_2581.py
---------------------------------------------
Unit test suite for Issue #2581:
Validates that OAuth SSO state parameters are generated, stored server-side in the database,
and verified upon callback return to protect against Login CSRF attacks.
"""

from unittest.mock import patch
import pytest

from src.db.auth import init_db, validate_sso_state, store_sso_state
from src.utils.sso import (
    get_google_auth_url,
    get_github_auth_url,
    verify_sso_state,
    exchange_google_code,
    exchange_github_code,
)


@pytest.fixture(autouse=True)
def setup_auth_db(tmp_path, monkeypatch):
    """Setup clean isolated auth DB for state tracking tests."""
    db_file = tmp_path / "test_sso_auth.db"
    monkeypatch.setenv("AUTH_DB_PATH", str(db_file))
    init_db()


def test_google_auth_url_stores_state(monkeypatch):
    """Verify get_google_auth_url generates and stores state server-side."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-google-client-id")
    url, state, state_data = get_google_auth_url()

    assert f"state={state}" in url
    assert state.startswith("google_")
    # Verify state was stored server-side in DB and can be validated
    assert verify_sso_state(state) is True
    # Verify state is invalidated after single use (replay protection)
    assert verify_sso_state(state) is False


def test_github_auth_url_stores_state(monkeypatch):
    """Verify get_github_auth_url generates and stores state server-side."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test-github-client-id")
    url, state = get_github_auth_url()

    assert f"state={state}" in url
    assert state.startswith("github_")
    # Verify state was stored server-side in DB and can be validated
    assert verify_sso_state(state) is True
    # Verify state is invalidated after single use (replay protection)
    assert verify_sso_state(state) is False


def test_verify_sso_state_rejects_unregistered_state():
    """Verify verify_sso_state rejects forged or unregistered state parameters."""
    forged_state = "google_forged_csrf_state_12345"
    assert verify_sso_state(forged_state) is False
    assert verify_sso_state("") is False
    assert verify_sso_state(None) is False


def test_exchange_google_code_rejects_invalid_csrf_state():
    """Verify exchange_google_code rejects requests with invalid or forged state."""
    profile, error = exchange_google_code(code="dummy_code", state="invalid_state_123")
    assert profile is None
    assert "CSRF protection failed" in error


def test_exchange_github_code_rejects_invalid_csrf_state():
    """Verify exchange_github_code rejects requests with invalid or forged state."""
    profile, error = exchange_github_code(code="dummy_code", state="invalid_state_123")
    assert profile is None
    assert "CSRF protection failed" in error


@patch("src.db.auth.log_security_event")
def test_verify_sso_state_logs_security_event_on_failure(mock_log_security_event):
    """Verify log_security_event is called with SSO_CSRF_REJECTED on state verification failure."""
    forged_state = "google_forged_csrf_state_12345"
    assert verify_sso_state(forged_state) is False

    mock_log_security_event.assert_called_with(
        "SSO_CSRF_REJECTED",
        username="anonymous",
        details=f"Invalid state: {forged_state}",
    )

