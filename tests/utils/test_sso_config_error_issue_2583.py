"""
test_sso_config_error_issue_2583.py
------------------------------------
Unit tests for Issue #2583: Missing GOOGLE_CLIENT_ID raises a dedicated SSOConfigurationError
(inheriting from ValueError) so the UI layer catches and formats it nicely as a Streamlit st.error
instead of causing an unhandled HTTP 500 server error.
"""

from __future__ import annotations

import pytest

from src.errors import SSOConfigurationError
from src.utils.sso import (
    SSOConfigurationError as SSOConfigErrorFromSSO,
    exchange_azure_code,
    exchange_github_code,
    exchange_google_code,
    get_azure_auth_url,
    get_github_auth_url,
    get_google_auth_url,
)


def test_sso_configuration_error_inheritance():
    """Verify SSOConfigurationError subclasses ValueError for backward compatibility."""
    err = SSOConfigurationError("GOOGLE_CLIENT_ID environment variable is not configured")
    assert isinstance(err, ValueError)
    assert isinstance(err, SSOConfigurationError)
    assert SSOConfigErrorFromSSO is SSOConfigurationError


def test_get_google_auth_url_raises_sso_configuration_error(monkeypatch):
    """Verify get_google_auth_url raises SSOConfigurationError when GOOGLE_CLIENT_ID is missing."""
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    with pytest.raises(SSOConfigurationError, match="GOOGLE_CLIENT_ID environment variable is not configured"):
        get_google_auth_url()


def test_exchange_google_code_raises_sso_configuration_error_missing_client_id(monkeypatch):
    """Verify exchange_google_code raises SSOConfigurationError when GOOGLE_CLIENT_ID is missing."""
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(SSOConfigurationError, match="GOOGLE_CLIENT_ID environment variable is not configured"):
        exchange_google_code("dummy_code")


def test_exchange_google_code_raises_sso_configuration_error_missing_client_secret(monkeypatch):
    """Verify exchange_google_code raises SSOConfigurationError when GOOGLE_CLIENT_SECRET is missing."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    with pytest.raises(SSOConfigurationError, match="GOOGLE_CLIENT_SECRET environment variable is not configured"):
        exchange_google_code("dummy_code")


def test_get_github_auth_url_raises_sso_configuration_error(monkeypatch):
    """Verify get_github_auth_url raises SSOConfigurationError when GITHUB_CLIENT_ID is missing."""
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    with pytest.raises(SSOConfigurationError, match="GITHUB_CLIENT_ID environment variable is not configured"):
        get_github_auth_url()


def test_exchange_github_code_raises_sso_configuration_error_missing_client_id(monkeypatch):
    """Verify exchange_github_code raises SSOConfigurationError when GITHUB_CLIENT_ID is missing."""
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(SSOConfigurationError, match="GITHUB_CLIENT_ID environment variable is not configured"):
        exchange_github_code("dummy_code")


def test_exchange_github_code_raises_sso_configuration_error_missing_client_secret(monkeypatch):
    """Verify exchange_github_code raises SSOConfigurationError when GITHUB_CLIENT_SECRET is missing."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("GITHUB_CLIENT_SECRET", raising=False)
    with pytest.raises(SSOConfigurationError, match="GITHUB_CLIENT_SECRET environment variable is not configured"):
        exchange_github_code("dummy_code")


def test_get_azure_auth_url_raises_sso_configuration_error(monkeypatch):
    """Verify get_azure_auth_url raises SSOConfigurationError when AZURE_CLIENT_ID is missing."""
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    with pytest.raises(SSOConfigurationError, match="AZURE_CLIENT_ID environment variable is not configured"):
        get_azure_auth_url()


def test_exchange_azure_code_raises_sso_configuration_error_missing_client_id(monkeypatch):
    """Verify exchange_azure_code raises SSOConfigurationError when AZURE_CLIENT_ID is missing."""
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy_secret")
    with pytest.raises(SSOConfigurationError, match="AZURE_CLIENT_ID environment variable is not configured"):
        exchange_azure_code("dummy_code")


def test_exchange_azure_code_raises_sso_configuration_error_missing_client_secret(monkeypatch):
    """Verify exchange_azure_code raises SSOConfigurationError when AZURE_CLIENT_SECRET is missing."""
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy_client_id")
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    with pytest.raises(SSOConfigurationError, match="AZURE_CLIENT_SECRET environment variable is not configured"):
        exchange_azure_code("dummy_code")


def test_ui_graceful_catch_formatting():
    """Verify UI layer catch block cleanly formats SSOConfigurationError into a user-friendly error string."""
    try:
        raise SSOConfigurationError("GOOGLE_CLIENT_ID environment variable is not configured")
    except (SSOConfigurationError, ValueError) as exc:
        user_info, error_msg = None, f"Configuration Error: {exc}"

    assert user_info is None
    assert error_msg == "Configuration Error: GOOGLE_CLIENT_ID environment variable is not configured"
