"""
test_sso_custom_scopes_issue_3458.py
------------------------------------
Unit tests for Issue #3458: Support custom OAuth scopes via environment variables.

Validates:
  1. Default scopes are used for Google ("email profile") and GitHub ("user:email")
     when environment variables are not set.
  2. Custom scopes are parsed and used from GOOGLE_OAUTH_SCOPES and GITHUB_OAUTH_SCOPES
     when defined.
"""

from urllib.parse import urlparse, parse_qs
from src.utils.sso import get_google_auth_url, get_github_auth_url


def test_google_auth_url_default_scope(monkeypatch):
    """Verify Google OAuth URL uses default scope 'email profile' when env var is absent."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.delenv("GOOGLE_OAUTH_SCOPES", raising=False)
    
    url, _, _ = get_google_auth_url()
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    assert params["scope"] == ["email profile"]


def test_google_auth_url_custom_scope(monkeypatch):
    """Verify Google OAuth URL uses custom scope from GOOGLE_OAUTH_SCOPES."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setenv("GOOGLE_OAUTH_SCOPES", "openid email profile https://www.googleapis.com/auth/drive.readonly")
    
    url, _, _ = get_google_auth_url()
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    assert params["scope"] == ["openid email profile https://www.googleapis.com/auth/drive.readonly"]


def test_github_auth_url_default_scope(monkeypatch):
    """Verify GitHub OAuth URL uses default scope 'user:email' when env var is absent."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test-client")
    monkeypatch.delenv("GITHUB_OAUTH_SCOPES", raising=False)
    
    url, _, _ = get_github_auth_url()
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    assert params["scope"] == ["user:email"]


def test_github_auth_url_custom_scope(monkeypatch):
    """Verify GitHub OAuth URL uses custom scope from GITHUB_OAUTH_SCOPES."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("GITHUB_OAUTH_SCOPES", "read:user read:org")
    
    url, _, _ = get_github_auth_url()
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    assert params["scope"] == ["read:user read:org"]
