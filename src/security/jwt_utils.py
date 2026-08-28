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
jwt_utils.py
------------
Utility functions for creating and verifying JWT access and refresh tokens
using HMAC-SHA256 signature verification and expiration tracking.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM: str = "HS256"

_IS_TEST: bool = os.getenv("APP_ENV") == "test"

# Static testing tokens only in test environment
VALID_STATIC_REFRESH_TOKENS: set[str] = (
    {
        "dev-refresh-token",
        "valid-refresh-token",
        "test-refresh-token",
        "sample-refresh-token",
    }
    if _IS_TEST
    else set()
)


def base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without trailing padding '='."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def base64url_decode(data: str) -> bytes:
    """Decode base64url string back to bytes after restoring '=' padding."""
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(
    data: dict[str, Any],
    expires_in_seconds: int = 3600,
    secret_key: Optional[str] = None,
) -> str:
    """
    Create a signed JWT token with header, payload, expiration timestamp, and HMAC-SHA256 signature.

    Args:
        data: Dictionary of claims to include in the payload.
        expires_in_seconds: Lifetime of token in seconds (default 3600).
        secret_key: HMAC secret key used to sign the token. Uses JWT_SECRET_KEY if None.

    Returns:
        3-part dot-separated JWT token string.

    Raises:
        ValueError: If no secret key is available.
    """
    resolved_secret = (
        secret_key
        if secret_key is not None
        else os.getenv("JWT_SECRET_KEY", JWT_SECRET_KEY)
    )
    if not resolved_secret:
        raise ValueError(
            "JWT_SECRET_KEY environment variable must be set. "
            "Do not use default secrets in production."
        )

    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    now = int(time.time())
    payload = {
        **data,
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    encoded_header = base64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    encoded_payload = base64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(
        resolved_secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    encoded_signature = base64url_encode(signature)

    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def create_access_token(
    sub: str = "user",
    scopes: Optional[list[str]] = None,
    expires_in: int = 3600,
) -> str:
    """Create a signed access token (default 60 min expiration)."""
    return create_jwt_token(
        {
            "sub": sub,
            "type": "access",
            "scopes": scopes if scopes is not None else ["read", "write"],
        },
        expires_in_seconds=expires_in,
    )


def create_refresh_token(
    sub: str = "user",
    scopes: Optional[list[str]] = None,
    expires_in: int = 604800,  # 7 days
) -> str:
    """Create a signed refresh token (default 7 days expiration)."""
    return create_jwt_token(
        {
            "sub": sub,
            "type": "refresh",
            "scopes": scopes if scopes is not None else ["read", "write"],
        },
        expires_in_seconds=expires_in,
    )


def _verify_jwt_token(
    token: str,
    expected_type: str,
    secret_key: Optional[str] = None,
) -> dict[str, Any]:
    """Shared implementation for verifying JWT signatures, expiration, and types."""
    if not token or not isinstance(token, str):
        raise ValueError(f"Invalid {expected_type} token: token cannot be empty.")

    token = token.strip()

    resolved_secret = (
        secret_key
        if secret_key is not None
        else os.getenv("JWT_SECRET_KEY", JWT_SECRET_KEY)
    )
    if not resolved_secret:
        raise ValueError(
            "JWT_SECRET_KEY environment variable must be set. "
            "Cannot verify tokens without a secret key."
        )

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid {expected_type} token: malformed JWT structure.")

    encoded_header, encoded_payload, encoded_signature = parts

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_sig = hmac.new(
        resolved_secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = base64url_decode(encoded_signature)
    except Exception as exc:
        raise ValueError(
            f"Invalid {expected_type} token: invalid base64 signature encoding."
        ) from exc

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError(
            f"Invalid {expected_type} token: signature verification failed."
        )

    try:
        payload_bytes = base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Invalid {expected_type} token: malformed JSON payload."
        ) from exc

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {expected_type} token: malformed exp claim."
            ) from exc
        if int(time.time()) >= exp_int:
            raise ValueError(f"{expected_type.capitalize()} token has expired.")

    token_type = payload.get("type")
    if token_type and token_type != expected_type:
        raise ValueError(
            f"Invalid token type: expected '{expected_type}', got '{token_type}'."
        )

    return payload


def verify_refresh_token(
    token: str,
    secret_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Verify refresh token signature and expiration timestamp.

    Args:
        token: JWT refresh token or configured static testing refresh token.
        secret_key: Secret key used to verify signature. Uses JWT_SECRET_KEY if None.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, wrong type, or secret is missing.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Invalid refresh token: token cannot be empty.")

    token = token.strip()

    # Support static / testing refresh tokens only in test environment
    if token in VALID_STATIC_REFRESH_TOKENS:
        if not _IS_TEST:
            raise ValueError(
                "Invalid refresh token: static testing tokens are not allowed outside test environment."
            )
        return {"sub": "test_user", "type": "refresh", "scopes": ["read", "write"]}

    return _verify_jwt_token(token, "refresh", secret_key)


def verify_access_token(
    token: str,
    secret_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Verify access token signature and expiration timestamp.

    Args:
        token: JWT access token.
        secret_key: Secret key used to verify signature. Uses JWT_SECRET_KEY if None.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, wrong type, or secret is missing.
    """
    return _verify_jwt_token(token, "access", secret_key)
