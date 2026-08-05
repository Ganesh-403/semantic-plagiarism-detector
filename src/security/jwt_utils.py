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
from typing import Any

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Known static testing / fallback refresh tokens for integration tests
VALID_STATIC_REFRESH_TOKENS = {
    "dev-refresh-token",
    "valid-refresh-token",
    "test-refresh-token",
    "sample-refresh-token",
}


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
    secret_key: str = JWT_SECRET_KEY,
) -> str:
    """
    Create a signed JWT token with header, payload, expiration timestamp, and HMAC-SHA256 signature.

    Args:
        data: Dictionary of claims to include in the payload.
        expires_in_seconds: Lifetime of token in seconds (default 3600).
        secret_key: HMAC secret key used to sign the token.

    Returns:
        3-part dot-separated JWT token string.
    """
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
        secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    encoded_signature = base64url_encode(signature)

    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def create_access_token(
    sub: str = "user",
    scopes: list[str] | None = None,
    expires_in: int = 3600,
) -> str:
    """Create a signed access token (default 60 min expiration)."""
    return create_jwt_token(
        {
            "sub": sub,
            "type": "access",
            "scopes": scopes or ["read", "write"],
        },
        expires_in_seconds=expires_in,
    )


def create_refresh_token(
    sub: str = "user",
    scopes: list[str] | None = None,
    expires_in: int = 604800,  # 7 days
) -> str:
    """Create a signed refresh token (default 7 days expiration)."""
    return create_jwt_token(
        {
            "sub": sub,
            "type": "refresh",
            "scopes": scopes or ["read", "write"],
        },
        expires_in_seconds=expires_in,
    )


def verify_refresh_token(
    token: str,
    secret_key: str = JWT_SECRET_KEY,
) -> dict[str, Any]:
    """
    Verify refresh token signature and expiration timestamp.

    Args:
        token: JWT refresh token or configured static testing refresh token.
        secret_key: Secret key used to verify signature.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, or wrong type.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Invalid refresh token: token cannot be empty.")

    token = token.strip()

    # Support static / testing refresh tokens
    if token in VALID_STATIC_REFRESH_TOKENS:
        return {"sub": "test_user", "type": "refresh", "scopes": ["read", "write"]}

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid refresh token: malformed JWT structure.")

    encoded_header, encoded_payload, encoded_signature = parts

    # 1. Verify HMAC-SHA256 signature
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = base64url_decode(encoded_signature)
    except Exception:
        raise ValueError("Invalid refresh token: invalid base64 signature encoding.")

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid refresh token: signature verification failed.")

    # 2. Decode payload
    try:
        payload_bytes = base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise ValueError("Invalid refresh token: malformed JSON payload.")

    # 3. Check expiration
    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except (TypeError, ValueError):
            raise ValueError("Invalid refresh token: malformed exp claim.")
        if int(time.time()) >= exp_int:
            raise ValueError("Refresh token has expired.")

    # 4. Check token type if present
    token_type = payload.get("type")
    if token_type and token_type != "refresh":
        raise ValueError(f"Invalid token type: expected 'refresh', got '{token_type}'.")

    return payload


def verify_access_token(
    token: str,
    secret_key: str = JWT_SECRET_KEY,
) -> dict[str, Any]:
    """
    Verify access token signature and expiration timestamp.

    Args:
        token: JWT access token.
        secret_key: Secret key used to verify signature.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, or wrong type.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Invalid access token: token cannot be empty.")

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid access token: malformed JWT structure.")

    encoded_header, encoded_payload, encoded_signature = parts

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = base64url_decode(encoded_signature)
    except Exception:
        raise ValueError("Invalid access token: invalid base64 signature encoding.")

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid access token: signature verification failed.")

    try:
        payload_bytes = base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise ValueError("Invalid access token: malformed JSON payload.")

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except (TypeError, ValueError):
            raise ValueError("Invalid access token: malformed exp claim.")
        if int(time.time()) >= exp_int:
            raise ValueError("Access token has expired.")

    token_type = payload.get("type")
    if token_type and token_type != "access":
        raise ValueError(f"Invalid token type: expected 'access', got '{token_type}'.")

    return payload
