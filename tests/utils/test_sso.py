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

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.sso import (
    SSOUserProfile,
    exchange_azure_code,
    exchange_github_code,
    exchange_google_code,
    get_azure_auth_url,
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
    assert "prompt=select_account" in url
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
        "picture": "https://example.com/avatar.png",
    }

    user_data, error_msg = exchange_google_code("valid_code")
    assert user_data == SSOUserProfile(
        email="user@example.com",
        username="user",
        name="Test User",
        avatar="https://example.com/avatar.png",
    )
    assert error_msg is None
    mock_post.assert_called_once()
    mock_get.assert_called_once()


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_google_code_sanitizes_username(mock_post, mock_get, monkeypatch):
    """Test that email with special characters (dots, pluses) produces a sanitized username."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "google_token_123"}

    mock_get.return_value.ok = True
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "email": "john.doe+class@school.edu",
        "name": "John Doe",
        "picture": "https://example.com/avatar.png",
    }

    user_data, error_msg = exchange_google_code("valid_code")
    assert user_data == SSOUserProfile(
        email="john.doe+class@school.edu",
        username="john_doe_class",
        name="John Doe",
        avatar="https://example.com/avatar.png",
    )
    assert error_msg is None


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
        "name": "The Octocat",
        "avatar_url": "https://example.com/octocat.png",
    }

    user_data, error_msg = exchange_github_code("valid_code")
    assert user_data == SSOUserProfile(
        email="octocat@github.com",
        username="octocat",
        name="The Octocat",
        avatar="https://example.com/octocat.png",
    )
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
    user_response.json.return_value = {
        "login": "octocat",
        "email": None,
        "name": "The Octocat",
        "avatar_url": "https://example.com/octocat.png",
    }

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
    assert user_data == SSOUserProfile(
        email="primary@github.com",
        username="octocat",
        name="The Octocat",
        avatar="https://example.com/octocat.png",
    )
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
        "name": "The Octocat",
        "avatar_url": "https://example.com/octocat.png",
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
    assert user_data == SSOUserProfile(
        email="octocat@github.com",
        username="octocat",
        name="The Octocat",
        avatar="https://example.com/octocat.png",
    )
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


def test_get_azure_auth_url_missing_client_id(monkeypatch):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    with pytest.raises(
        ValueError, match="AZURE_CLIENT_ID environment variable is not configured"
    ):
        get_azure_auth_url()


def test_get_azure_auth_url_success(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_azure_client_id")
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    url, state = get_azure_auth_url()
    assert "dummy_azure_client_id" in url
    assert "login.microsoftonline.com/common/oauth2/v2.0/authorize" in url
    assert state.startswith("azure_")


def test_get_azure_auth_url_custom_tenant(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_azure_client_id")
    monkeypatch.setenv("AZURE_TENANT_ID", "contoso-tenant-id")
    url, state = get_azure_auth_url()
    assert "login.microsoftonline.com/contoso-tenant-id/oauth2/v2.0/authorize" in url


def test_exchange_azure_code_missing_client_id(monkeypatch):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(
        ValueError, match="AZURE_CLIENT_ID environment variable is not configured"
    ):
        exchange_azure_code("dummy_code")


def test_exchange_azure_code_missing_client_secret(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    with pytest.raises(
        ValueError, match="AZURE_CLIENT_SECRET environment variable is not configured"
    ):
        exchange_azure_code("dummy_code")


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_azure_code_success(mock_post, mock_get, monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy_secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "custom_tenant")

    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "azure_token_123"}

    mock_get.return_value.ok = True
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "mail": "john.doe@university.edu",
        "displayName": "John Doe",
        "userPrincipalName": "john.doe@university.edu",
    }

    user_data, error_msg = exchange_azure_code("valid_code")
    assert user_data == SSOUserProfile(
        email="john.doe@university.edu",
        username="john_doe",
        name="John Doe",
        avatar="",
    )
    assert error_msg is None
    mock_post.assert_called_once()
    mock_get.assert_called_once()


@patch("src.utils.sso.requests.post")
def test_azure_oauth_token_exchange_timeout(mock_post, monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy_secret")

    mock_post.side_effect = requests.Timeout()

    user_data, error_msg = exchange_azure_code("valid_code")
    assert user_data is None
    assert error_msg == "SSO provider timed out. Please try again."
