"""
test_jwt_utils.py
------------------
Unit tests for JWT token generation, signature verification, and expiration in src/security/jwt_utils.py.
"""

import time
import pytest
from src.security.jwt_utils import (
    create_access_token,
    create_jwt_token,
    create_refresh_token,
    verify_access_token,
    verify_refresh_token,
)


def test_create_and_verify_access_token():
    token = create_access_token(sub="alice", scopes=["read", "write"])
    payload = verify_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["type"] == "access"
    assert "read" in payload["scopes"]


def test_create_and_verify_refresh_token():
    token = create_refresh_token(sub="bob", scopes=["read"])
    payload = verify_refresh_token(token)
    assert payload["sub"] == "bob"
    assert payload["type"] == "refresh"


def test_static_refresh_tokens():
    payload = verify_refresh_token("valid-refresh-token")
    assert payload["sub"] == "test_user"
    assert payload["type"] == "refresh"


def test_invalid_signature():
    token = create_access_token(sub="charlie")
    header, payload, _sig = token.split(".")
    tampered_token = f"{header}.{payload}.tampered_signature"

    with pytest.raises(ValueError, match="signature verification failed"):
        verify_access_token(tampered_token)


def test_expired_token():
    token = create_jwt_token({"sub": "david", "type": "access"}, expires_in_seconds=-10)
    with pytest.raises(ValueError, match="expired"):
        verify_access_token(token)


def test_wrong_token_type():
    access_token = create_access_token(sub="eve")
    with pytest.raises(ValueError, match="expected 'refresh'"):
        verify_refresh_token(access_token)

    refresh_token = create_refresh_token(sub="eve")
    with pytest.raises(ValueError, match="expected 'access'"):
        verify_access_token(refresh_token)
