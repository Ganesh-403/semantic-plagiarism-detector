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

"""
test_jwt_utils.py
------------------
Unit tests for JWT token generation, signature verification, and expiration in src/security/jwt_utils.py.
"""

import pytest

from src.security import jwt_utils
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


def test_secret_key_set_after_import(monkeypatch):
    """Regression test for #2050: JWT_SECRET_KEY must be read at call time,
    not only at module import time, so setting the env var after import
    (e.g. via a later dotenv.load_dotenv() call) still works."""
    monkeypatch.setattr(jwt_utils, "JWT_SECRET_KEY", None)
    monkeypatch.setenv("JWT_SECRET_KEY", "late-loaded-secret")

    token = create_jwt_token({"sub": "frank", "type": "access"})
    payload = verify_access_token(token)
    assert payload["sub"] == "frank"

    refresh_token = create_jwt_token({"sub": "frank", "type": "refresh"})
    refresh_payload = verify_refresh_token(refresh_token)
    assert refresh_payload["sub"] == "frank"
