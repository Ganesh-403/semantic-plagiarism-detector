# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""End-to-end integration test for the mock Google OAuth SSO login flow.

Simulates a complete OAuth login cycle:
    1. Generate the authorization URL and CSRF state (store_sso_state).
    2. Mock the provider's token exchange and userinfo endpoints.
    3. Exchange the authorization code for a user profile (state is
       validated/consumed as part of this step).
    4. Create/log in the local user session from the returned profile.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.db.auth import get_or_create_sso_user, init_db, validate_sso_state
from src.utils.sso import exchange_google_code, get_google_auth_url


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Uses the mock_db fixture from conftest.py to isolate DB operations."""
    init_db()
    yield


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_full_google_oauth_login_cycle(mock_post, mock_get, monkeypatch):
    """Simulate a full OAuth cycle: auth URL -> state -> token exchange -> session."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    # 1. Generate the authorization URL; this also stores the state parameter.
    auth_url, state = get_google_auth_url()
    assert "dummy_client_id" in auth_url
    assert state.startswith("google_")

    # State must be valid (stored, unused) before the callback is handled.
    assert validate_sso_state(state) is True
    # ``validate_sso_state`` consumes the state, so a second check must fail.
    assert validate_sso_state(state) is False

    # Since the state above was consumed by the assertion, generate a fresh
    # one to drive the rest of the flow exactly as the real callback would.
    auth_url, state = get_google_auth_url()

    # 2. Mock the provider's token exchange endpoint.
    mock_post.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {"access_token": "mock_access_token"},
    )

    # 3. Mock the provider's userinfo endpoint.
    mock_get.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {
            "email": "student@example.com",
            "name": "Test Student",
            "picture": "https://example.com/avatar.png",
        },
    )

    # 4. Exchange the authorization code for a user profile. This validates
    #    and consumes the state parameter (CSRF protection).
    profile, error = exchange_google_code("mock_auth_code", state=state)

    assert error is None
    assert profile is not None
    assert profile.email == "student@example.com"
    assert profile.username == "student"
    assert profile.name == "Test Student"
    assert profile.avatar == "https://example.com/avatar.png"

    mock_post.assert_called_once()
    mock_get.assert_called_once()

    # The state parameter must now be consumed and rejected on reuse
    # (protects against replay of the OAuth callback).
    assert validate_sso_state(state) is False

    # 5. Create the local user session/account from the SSO profile.
    role = get_or_create_sso_user(profile.email)
    assert role is not None


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_oauth_cycle_rejects_invalid_state(mock_post, mock_get, monkeypatch):
    """A callback with an unknown/forged state must be rejected before any network call."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    profile, error = exchange_google_code("mock_auth_code", state="forged_state_value")

    assert profile is None
    assert "Invalid or expired SSO state parameter" in error
    mock_post.assert_not_called()
    mock_get.assert_not_called()
