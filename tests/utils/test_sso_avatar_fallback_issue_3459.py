"""
test_sso_avatar_fallback_issue_3459.py
--------------------------------------
Unit tests for Issue #3459: Fallback avatar generation for SSO users without pictures.

Validates:
  1. exchange_google_code returns a fallback ui-avatars.com URL when profile has no picture.
  2. exchange_github_code returns a fallback ui-avatars.com URL when profile has no avatar_url.
"""

import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# Stub out the heavy framework dependencies that cause import errors in this environment
ld = types.ModuleType("langdetect")
ld.DetectorFactory = type("DF", (object,), {})
ld.LangDetectException = Exception
ld.detect = lambda x: "en"
ld.detect_langs = lambda x: []
sys.modules["langdetect"] = ld

f = types.ModuleType("faiss")
f.Index = None
sys.modules["faiss"] = f

dt = types.ModuleType("deep_translator")
dt.GoogleTranslator = None
sys.modules["deep_translator"] = dt

from src.utils.sso import exchange_google_code, exchange_github_code, SSOUserProfile


@patch("src.utils.sso.verify_sso_state")
@patch("src.utils.sso._get_oauth_session")
def test_exchange_google_code_avatar_fallback(mock_session_factory, mock_verify, monkeypatch):
    """Verify that Google SSO login generates a fallback avatar when picture is empty/missing."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test_secret")
    mock_verify.return_value = True

    # Setup mock responses
    mock_session = MagicMock()
    mock_session_factory.return_value = mock_session

    mock_token_resp = MagicMock(status_code=200, ok=True)
    mock_token_resp.json.return_value = {"access_token": "google_token_123"}
    mock_session.post.return_value = mock_token_resp

    mock_user_resp = MagicMock(status_code=200, ok=True)
    mock_user_resp.json.return_value = {
        "email": "empty-avatar@gmail.com",
        "name": "Jane Doe",
        "picture": None,  # Missing picture
    }
    mock_session.get.return_value = mock_user_resp

    profile, error = exchange_google_code("valid_code", state="google_state")
    
    assert error is None
    assert isinstance(profile, SSOUserProfile)
    assert profile.avatar == "https://ui-avatars.com/api/?name=Jane%20Doe"


@patch("src.utils.sso.verify_sso_state")
@patch("src.utils.sso._get_oauth_session")
def test_exchange_github_code_avatar_fallback(mock_session_factory, mock_verify, monkeypatch):
    """Verify that GitHub SSO login generates a fallback avatar when avatar_url is empty/missing."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "test_secret")
    mock_verify.return_value = True

    # Setup mock responses
    mock_session = MagicMock()
    mock_session_factory.return_value = mock_session

    mock_token_resp = MagicMock(status_code=200, ok=True)
    mock_token_resp.json.return_value = {"access_token": "github_token_123"}
    mock_session.post.return_value = mock_token_resp

    mock_user_resp = MagicMock(status_code=200, ok=True)
    mock_user_resp.json.return_value = {
        "login": "janesmith",
        "email": "janesmith@github.com",
        "name": "Jane Smith",
        "avatar_url": None,  # Missing avatar_url
    }
    mock_session.get.return_value = mock_user_resp

    profile, error = exchange_github_code("valid_code", state="github_state")

    assert error is None
    assert isinstance(profile, SSOUserProfile)
    assert profile.avatar == "https://ui-avatars.com/api/?name=Jane%20Smith"
