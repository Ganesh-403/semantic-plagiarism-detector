from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.sso import (
    exchange_github_code,
    exchange_google_code,
    get_github_auth_url,
    get_google_auth_url,
)


def test_get_google_auth_url_missing_client_id(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    with pytest.raises(
        ValueError, match="GOOGLE_CLIENT_ID environment variable is not configured"
    ):
        get_google_auth_url()


def test_get_google_auth_url_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_google_client_id")
    url, state = get_google_auth_url()
    assert "dummy_google_client_id" in url
    assert state.startswith("google_")


def test_exchange_google_code_missing_client_id(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(
        ValueError, match="GOOGLE_CLIENT_ID environment variable is not configured"
    ):
        exchange_google_code("dummy_code")


def test_exchange_google_code_missing_client_secret(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    with pytest.raises(
        ValueError, match="GOOGLE_CLIENT_SECRET environment variable is not configured"
    ):
        exchange_google_code("dummy_code")


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_google_code_success(mock_post, mock_get, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "google_token_123"}

    mock_get.return_value.ok = True
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "email": "user@example.com",
        "name": "Test User",
    }

    user_data, error_msg = exchange_google_code("valid_code")
    assert user_data == {"email": "user@example.com", "name": "Test User"}
    assert error_msg is None
    mock_post.assert_called_once()
    mock_get.assert_called_once()


@patch("src.utils.sso.requests.post")
def test_exchange_google_code_unauthorized(mock_post, monkeypatch):
    """Test Google OAuth returns 4xx error message when authorization code is rejected."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = 401

    user_data, error_msg = exchange_google_code("bad_code")

    assert user_data is None
    assert error_msg == "Invalid or expired SSO authorization code"
    mock_post.assert_called_once()


def test_get_github_auth_url_missing_client_id(monkeypatch):
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    with pytest.raises(
        ValueError, match="GITHUB_CLIENT_ID environment variable is not configured"
    ):
        get_github_auth_url()


def test_get_github_auth_url_success(monkeypatch):
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_github_client_id")
    url, state = get_github_auth_url()
    assert "dummy_github_client_id" in url
    assert state.startswith("github_")


def test_exchange_github_code_missing_client_id(monkeypatch):
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(
        ValueError, match="GITHUB_CLIENT_ID environment variable is not configured"
    ):
        exchange_github_code("dummy_code")


def test_exchange_github_code_missing_client_secret(monkeypatch):
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("GITHUB_CLIENT_SECRET", raising=False)
    with pytest.raises(
        ValueError, match="GITHUB_CLIENT_SECRET environment variable is not configured"
    ):
        exchange_github_code("dummy_code")


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_github_code_success(mock_post, mock_get, monkeypatch):
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}

    mock_get.return_value.ok = True
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "login": "octocat",
        "email": "octocat@github.com",
    }

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data == {"login": "octocat", "email": "octocat@github.com"}
    assert error_msg is None
    mock_post.assert_called_once()
    mock_get.assert_called_once()


@patch("src.utils.sso.requests.post")
def test_exchange_github_code_4xx_unauthorized(mock_post, monkeypatch):
    """Test GitHub OAuth returns 4xx error message when authorization code is rejected."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = 400

    user_data, error_msg = exchange_github_code("bad_code")

    assert user_data is None
    assert error_msg == "Invalid or expired SSO authorization code"
    mock_post.assert_called_once()


@patch("src.utils.sso.requests.post")
def test_exchange_github_code_json_error_response(mock_post, monkeypatch):
    """Test GitHub OAuth returning 200 OK with error payload."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "error": "bad_verification_code",
        "error_description": "The code passed is incorrect or expired.",
    }

    user_data, error_msg = exchange_github_code("bad_code")

    assert user_data is None
    assert error_msg == "Invalid or expired SSO authorization code"


@patch("src.utils.sso.requests.post")
def test_oauth_token_exchange_timeout(mock_post, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    mock_post.side_effect = requests.Timeout()

    user_data, error_msg = exchange_google_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs.get("timeout") == 10


@patch("src.utils.sso.requests.post")
def test_github_oauth_token_exchange_timeout(mock_post, monkeypatch):
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.side_effect = requests.Timeout()

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs.get("timeout") == 10


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_oauth_user_request_timeout(mock_post, mock_get, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "google_token_123"}
    mock_get.side_effect = requests.Timeout()

    user_data, error_msg = exchange_google_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."

    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs.get("timeout") == 10


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_github_oauth_user_request_timeout(mock_post, mock_get, monkeypatch):
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}
    mock_get.side_effect = requests.Timeout()

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."

    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs.get("timeout") == 10


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_github_code_email_fallback_success(mock_post, mock_get, monkeypatch):
    """Test GitHub email fallback when user profile does not contain email."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}

    # First call: GET /user (returns profile without email)
    user_response = MagicMock()
    user_response.ok = True
    user_response.status_code = 200
    user_response.json.return_value = {"login": "octocat", "email": None}

    # Second call: GET /user/emails (returns list of emails)
    emails_response = MagicMock()
    emails_response.ok = True
    emails_response.status_code = 200
    emails_response.json.return_value = [
        {"email": "secondary@github.com", "primary": False, "verified": True},
        {"email": "primary@github.com", "primary": True, "verified": True},
    ]

    mock_get.side_effect = [user_response, emails_response]

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data == {"login": "octocat", "email": "primary@github.com"}
    assert error_msg is None
    assert mock_get.call_count == 2


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_github_code_email_fallback_timeout(mock_post, mock_get, monkeypatch):
    """Test GitHub email fallback when /user/emails request times out."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}

    user_response = MagicMock()
    user_response.ok = True
    user_response.status_code = 200
    user_response.json.return_value = {"login": "octocat", "email": None}

    mock_get.side_effect = [user_response, requests.Timeout()]

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."
    assert mock_get.call_count == 2


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_github_code_filters_noreply_profile_email(
    mock_post, mock_get, monkeypatch
):
    """Test that users.noreply.github.com email in user profile is filtered out and we fallback."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}

    # /user returns noreply email
    user_response = MagicMock()
    user_response.ok = True
    user_response.status_code = 200
    user_response.json.return_value = {
        "login": "octocat",
        "email": "12345+octocat@users.noreply.github.com",
    }

    # /user/emails returns list with real verified email
    emails_response = MagicMock()
    emails_response.ok = True
    emails_response.status_code = 200
    emails_response.json.return_value = [
        {
            "email": "12345+octocat@users.noreply.github.com",
            "primary": True,
            "verified": True,
        },
        {"email": "octocat@github.com", "primary": False, "verified": True},
    ]

    mock_get.side_effect = [user_response, emails_response]

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data == {"login": "octocat", "email": "octocat@github.com"}
    assert error_msg is None


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_github_code_rejects_login_with_no_public_email(
    mock_post, mock_get, monkeypatch
):
    """Test that login is rejected with ValueError if no valid public verified email is found."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "github_token_123"}

    # Profile email is missing
    user_response = MagicMock()
    user_response.ok = True
    user_response.status_code = 200
    user_response.json.return_value = {"login": "octocat", "email": None}

    # /user/emails returns only unverified emails or noreply emails
    emails_response = MagicMock()
    emails_response.ok = True
    emails_response.status_code = 200
    emails_response.json.return_value = [
        {
            "email": "12345+octocat@users.noreply.github.com",
            "primary": True,
            "verified": True,
        },
        {"email": "unverified@github.com", "primary": False, "verified": False},
    ]

    mock_get.side_effect = [user_response, emails_response]

    with pytest.raises(
        ValueError, match="GitHub login failed: A verified public email is required"
    ):
        exchange_github_code("valid_code")
