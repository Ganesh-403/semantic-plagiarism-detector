"""
test_pkce_google_oauth_issue_3453.py
------------------------------------
Unit tests for Issue #3453: PKCE (Proof Key for Code Exchange) for Google OAuth.

Validates:
  1. generate_pkce_pair produces a valid code_verifier and S256 code_challenge.
  2. get_google_auth_url includes code_challenge and code_challenge_method=S256.
  3. get_google_auth_url state_data contains code_verifier.
  4. exchange_google_code forwards code_verifier in the token exchange request.
"""

import base64
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from src.utils.sso import (
    exchange_google_code,
    generate_pkce_pair,
    get_google_auth_url,
)


# ---------------------------------------------------------------------------
# generate_pkce_pair tests
# ---------------------------------------------------------------------------


def test_generate_pkce_pair_returns_two_strings():
    """generate_pkce_pair must return (code_verifier, code_challenge) as strings."""
    verifier, challenge = generate_pkce_pair()
    assert isinstance(verifier, str)
    assert isinstance(challenge, str)


def test_generate_pkce_pair_verifier_length():
    """code_verifier produced by secrets.token_urlsafe(32) is 43 characters."""
    verifier, _ = generate_pkce_pair()
    assert len(verifier) == 43


def test_generate_pkce_pair_challenge_matches_verifier():
    """code_challenge must equal base64url(sha256(code_verifier)) without padding."""
    verifier, challenge = generate_pkce_pair()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_generate_pkce_pair_uniqueness():
    """Each call must produce a unique verifier."""
    pairs = [generate_pkce_pair() for _ in range(10)]
    verifiers = [v for v, _ in pairs]
    assert len(set(verifiers)) == 10


# ---------------------------------------------------------------------------
# get_google_auth_url PKCE tests
# ---------------------------------------------------------------------------


def test_get_google_auth_url_includes_pkce_params(monkeypatch):
    """Authorization URL must contain code_challenge and code_challenge_method=S256."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_client_id")
    url, state, state_data = get_google_auth_url()

    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url


def test_get_google_auth_url_state_data_has_code_verifier(monkeypatch):
    """state_data dict must include a code_verifier key for the caller to store."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_client_id")
    _, _, state_data = get_google_auth_url()

    assert "code_verifier" in state_data
    assert isinstance(state_data["code_verifier"], str)
    assert len(state_data["code_verifier"]) > 0


def test_get_google_auth_url_challenge_matches_verifier_in_state(monkeypatch):
    """The code_challenge in the URL must be the S256 hash of the code_verifier in state_data."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_client_id")
    url, _, state_data = get_google_auth_url()

    verifier = state_data["code_verifier"]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    # Extract code_challenge from URL query string
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert params["code_challenge"] == [expected_challenge]
    assert params["code_challenge_method"] == ["S256"]


# ---------------------------------------------------------------------------
# exchange_google_code PKCE tests
# ---------------------------------------------------------------------------


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_google_code_sends_code_verifier(mock_post, mock_get, monkeypatch):
    """When code_verifier is provided, it must be included in the token exchange POST data."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test_secret")

    mock_post.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {"access_token": "test_token"},
    )
    mock_get.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {
            "email": "pkce@example.com",
            "name": "PKCE User",
            "picture": "",
        },
    )

    exchange_google_code("auth_code", code_verifier="test_verifier_value")

    # Verify code_verifier was passed in the POST data
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["code_verifier"] == "test_verifier_value"


@patch("src.utils.sso.requests.get")
@patch("src.utils.sso.requests.post")
def test_exchange_google_code_omits_code_verifier_when_none(mock_post, mock_get, monkeypatch):
    """When code_verifier is None (default), it must NOT appear in the token exchange POST data."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test_secret")

    mock_post.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {"access_token": "test_token"},
    )
    mock_get.return_value = MagicMock(
        ok=True,
        status_code=200,
        json=lambda: {
            "email": "user@example.com",
            "name": "User",
            "picture": "",
        },
    )

    exchange_google_code("auth_code")

    _, kwargs = mock_post.call_args
    assert "code_verifier" not in kwargs["data"]
