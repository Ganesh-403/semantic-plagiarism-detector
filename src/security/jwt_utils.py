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
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256").upper()
JWT_PUBLIC_KEY: Optional[str] = os.getenv("JWT_PUBLIC_KEY")
JWT_PRIVATE_KEY: Optional[str] = os.getenv("JWT_PRIVATE_KEY")


def get_jwt_algorithm() -> str:
    """Return the active JWT signing algorithm from environment (e.g. HS256, RS256)."""
    return os.getenv("JWT_ALGORITHM", JWT_ALGORITHM).upper()


def _sign_rs256(signing_input: bytes, private_key_pem: str) -> bytes:
    """Sign bytes using RS256 (RSA-SHA256) with a PEM-encoded private key."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8") if isinstance(private_key_pem, str) else private_key_pem,
            password=None,
        )
        return key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    except Exception as exc:
        raise ValueError(f"Failed to sign token with RS256 private key: {exc}") from exc


def _verify_rs256(signing_input: bytes, signature: bytes, public_key_pem: str) -> bool:
    """Verify RS256 signature using a PEM-encoded public key."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        key = serialization.load_pem_public_key(
            public_key_pem.encode("utf-8") if isinstance(public_key_pem, str) else public_key_pem
        )
        key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False


class JWTDecodeError(ValueError):
    """Raised when the JWT structure is malformed or cannot be base64-decoded."""
    pass


class JWTExpiredError(ValueError):
    """Raised when the token's 'exp' timestamp is in the past."""
    pass


class JWTNotYetValidError(ValueError):
    """Raised when the token's 'nbf' timestamp is in the future."""
    pass


class JWTSignatureError(ValueError):
    """Raised when the cryptographic signature validation fails."""
    pass


_IS_TEST: bool = os.getenv("APP_ENV") == "test"

# Static testing tokens configuration
VALID_STATIC_REFRESH_TOKENS: set[str] = {
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
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """
    Create a signed JWT token with header, payload, expiration timestamp, and signature.

    Args:
        data: Dictionary of claims to include in the payload.
        expires_in_seconds: Lifetime of token in seconds (default 3600).
        secret_key: HMAC secret key or RSA private key used to sign the token.
        algorithm: Signing algorithm override (e.g. HS256, RS256).

    Returns:
        3-part dot-separated JWT token string.

    Raises:
        ValueError: If no secret key is available or if key validation fails.
    """
    alg = (algorithm or get_jwt_algorithm()).upper()

    # Resolve the signing key before anything else touches it. This block was
    # dropped by a bad merge (see #4081), which left the checks below reading
    # names that were never bound. Key resolution is per-algorithm because an
    # RSA private key and an HMAC shared secret are not interchangeable, and
    # the verify path (verify_jwt_token) is laid out the same way.
    private_key: Optional[str] = None
    resolved_secret: Optional[str] = None

    if alg == "RS256":
        private_key = secret_key or os.getenv("JWT_PRIVATE_KEY", JWT_PRIVATE_KEY)
        if not private_key:
            raise ValueError(
                "JWT_PRIVATE_KEY environment variable or secret_key parameter must be set for RS256 signing."
            )
    else:
        resolved_secret = secret_key if secret_key is not None else os.getenv("JWT_SECRET_KEY", JWT_SECRET_KEY)
        if not resolved_secret:
            raise ValueError(
                "JWT_SECRET_KEY environment variable must be set. "
                "Do not use default secrets in production."
            )

        # HMAC only. A PEM private key is not a shared secret, so counting its
        # characters says nothing about brute-force resistance.
        is_test_env = (os.getenv("APP_ENV") == "test") or _IS_TEST
        if len(resolved_secret) < 32 and not is_test_env:
            logger.critical(
                "SECURITY WARNING: JWT secret key is less than 32 characters! "
                "This makes HMAC signatures vulnerable to offline brute-force attacks."
            )
            raise ValueError("JWT secret key must be at least 32 characters long in production.")

    header = {"alg": alg, "typ": "JWT"}
    now = int(time.time())
    payload = {
        **data,
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": str(uuid.uuid4()),
    }

    encoded_header = base64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    encoded_payload = base64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")

    if alg == "RS256":
        signature = _sign_rs256(signing_input, private_key)
    else:
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
    clock_skew_seconds: int = 10,
    algorithm: Optional[str] = None,
) -> dict[str, Any]:
    """Shared implementation for verifying JWT signatures, expiration, and types."""
    if not token or not isinstance(token, str):
        raise JWTDecodeError(f"Invalid {expected_type} token: token cannot be empty.")

    token = token.strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise JWTDecodeError(f"Invalid {expected_type} token: malformed JWT structure.")

    encoded_header, encoded_payload, encoded_signature = parts

    try:
        header_bytes = base64url_decode(encoded_header)
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        header = {}

    alg = (algorithm or header.get("alg") or get_jwt_algorithm()).upper()
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")

    try:
        actual_sig = base64url_decode(encoded_signature)
    except Exception as exc:
        raise JWTDecodeError(f"Invalid {expected_type} token: invalid base64 signature encoding.") from exc

    if alg == "RS256":
        public_key = secret_key or os.getenv("JWT_PUBLIC_KEY", JWT_PUBLIC_KEY)
        if not public_key:
            raise ValueError(
                "JWT_PUBLIC_KEY environment variable or secret_key parameter must be set for RS256 verification."
            )
        if not _verify_rs256(signing_input, actual_sig, public_key):
            raise JWTSignatureError(f"Invalid {expected_type} token: RS256 signature verification failed.")
    else:
        resolved_secret = secret_key if secret_key is not None else os.getenv("JWT_SECRET_KEY", JWT_SECRET_KEY)
        if not resolved_secret:
            raise ValueError(
                "JWT_SECRET_KEY environment variable must be set. "
                "Cannot verify tokens without a secret key."
            )

        expected_sig = hmac.new(
            resolved_secret.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise JWTSignatureError(f"Invalid {expected_type} token: signature verification failed.")

    try:
        payload_bytes = base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise JWTDecodeError(f"Invalid {expected_type} token: malformed JSON payload.") from exc

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except (TypeError, ValueError) as exc:
            raise JWTDecodeError(f"Invalid {expected_type} token: malformed exp claim.") from exc
        if int(time.time()) >= exp_int + clock_skew_seconds:
            raise JWTExpiredError(f"{expected_type.capitalize()} token has expired.")

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            nbf_int = int(nbf)
        except (TypeError, ValueError) as exc:
            raise JWTDecodeError(f"Invalid {expected_type} token: malformed nbf claim.") from exc
        if int(time.time()) < nbf_int - clock_skew_seconds:
            raise JWTNotYetValidError(f"{expected_type.capitalize()} token is not yet valid (nbf).")

    token_type = payload.get("type")
    if token_type and not hmac.compare_digest(token_type, expected_type):
        raise JWTDecodeError(f"Invalid token type: expected '{expected_type}', got '{token_type}'.")

    return payload


def verify_refresh_token(
    token: str,
    secret_key: Optional[str] = None,
    clock_skew_seconds: int = 10,
) -> dict[str, Any]:
    """
    Verify refresh token signature and expiration timestamp.

    Args:
        token: JWT refresh token or configured static testing refresh token.
        secret_key: Secret key used to verify signature. Uses JWT_SECRET_KEY if None.
        clock_skew_seconds: Clock drift allowance in seconds.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, wrong type, or secret is missing.
    """
    if not token or not isinstance(token, str):
        raise JWTDecodeError("Invalid refresh token: token cannot be empty.")

    token = token.strip()

    # Support static / testing refresh tokens only in test environment
    if token in VALID_STATIC_REFRESH_TOKENS:
        is_test_env = (os.getenv("APP_ENV") == "test") or _IS_TEST
        if not is_test_env:
            logger.warning(
                "SECURITY ALERT: Attempted to use static testing refresh token '%s' outside of test environment (APP_ENV='%s').",
                token,
                os.getenv("APP_ENV", "production"),
            )
            raise JWTDecodeError(
                "Invalid refresh token: static testing tokens are not allowed outside test environment."
            )
        return {"sub": "test_user", "type": "refresh", "scopes": ["read", "write"]}

    return _verify_jwt_token(token, "refresh", secret_key, clock_skew_seconds=clock_skew_seconds)


def get_unverified_jwt_header(token: str) -> dict[str, Any]:
    """Decode and return the header of a JWT without verifying the signature.

    When rotating JWT keys with multiple key IDs (kid), the verification
    layer needs to inspect the unverified header to determine which
    public key to load.  This helper splits the compact JWS token,
    base64-decodes the first segment, and returns the parsed JSON header.

    Args:
        token: A compact-serialised JWT string (three dot-separated segments).

    Returns:
        A dictionary containing the decoded header claims (e.g. ``alg``,
        ``typ``, ``kid``).

    Raises:
        JWTDecodeError: If *token* is empty, not a string, does not
            contain exactly three dot-separated segments, or the header
            cannot be base64-decoded or parsed as JSON.
    """
    if not token or not isinstance(token, str):
        raise JWTDecodeError("Cannot decode header: token must be a non-empty string.")

    token = token.strip()
    parts = token.split(".")

    if len(parts) != 3:
        raise JWTDecodeError(
            f"Cannot decode header: expected 3 dot-separated segments, got {len(parts)}."
        )

    encoded_header = parts[0]

    if not encoded_header:
        raise JWTDecodeError("Cannot decode header: header segment is empty.")

    try:
        header_bytes = base64url_decode(encoded_header)
    except Exception as exc:
        raise JWTDecodeError(
            "Cannot decode header: invalid base64url encoding."
        ) from exc

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JWTDecodeError(
            "Cannot decode header: header is not valid JSON."
        ) from exc

    if not isinstance(header, dict):
        raise JWTDecodeError(
            "Cannot decode header: decoded header is not a JSON object."
        )

    return header


def verify_access_token(
    token: str,
    secret_key: Optional[str] = None,
    clock_skew_seconds: int = 10,
) -> dict[str, Any]:
    """
    Verify access token signature and expiration timestamp.

    Args:
        token: JWT access token.
        secret_key: Secret key used to verify signature. Uses JWT_SECRET_KEY if None.
        clock_skew_seconds: Clock drift allowance in seconds.

    Returns:
        Decoded token payload dict.

    Raises:
        ValueError: If token signature is invalid, expired, wrong type, or secret is missing.
    """
    return _verify_jwt_token(token, "access", secret_key, clock_skew_seconds=clock_skew_seconds)


# ---------------------------------------------------------------------------
# Key-rotation helpers
# ---------------------------------------------------------------------------


class JWTKeyRegistry:
    """In-memory registry mapping ``kid`` values to HMAC secret keys.

    This supports JWT key rotation: new keys are registered with a fresh
    ``kid``, old keys can be retired, and the verification layer looks up
    the right key by inspecting the unverified header.

    Example::

        registry = JWTKeyRegistry()
        registry.register_key("2024-01", secret_a)
        registry.register_key("2024-07", secret_b)

        header = get_unverified_jwt_header(token)
        key = registry.get_key(header["kid"])
        payload = _verify_jwt_token(token, "access", secret_key=key)
    """

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._active_kid: Optional[str] = None

    # -- mutation -----------------------------------------------------------

    def register_key(self, kid: str, secret: str) -> None:
        """Register or update a secret key under the given *kid*.

        Args:
            kid: Key identifier (must be a non-empty string).
            secret: The HMAC secret associated with *kid*.

        Raises:
            ValueError: If *kid* or *secret* is empty.
        """
        if not kid or not isinstance(kid, str):
            raise ValueError("kid must be a non-empty string.")
        if not secret or not isinstance(secret, str):
            raise ValueError("secret must be a non-empty string.")
        self._keys[kid.strip()] = secret

    def retire_key(self, kid: str) -> bool:
        """Remove a key from the registry.

        Returns ``True`` if the key existed and was removed, ``False``
        otherwise.
        """
        removed = self._keys.pop(kid, None) is not None
        if removed and self._active_kid == kid:
            self._active_kid = None
        return removed

    def set_active(self, kid: str) -> None:
        """Mark *kid* as the current active signing key.

        Raises:
            KeyError: If *kid* is not registered.
        """
        if kid not in self._keys:
            raise KeyError(f"kid '{kid}' is not registered.")
        self._active_kid = kid

    # -- queries ------------------------------------------------------------

    def get_key(self, kid: str) -> str:
        """Return the secret for *kid*, or raise ``KeyError``."""
        try:
            return self._keys[kid]
        except KeyError:
            raise KeyError(f"No secret registered for kid '{kid}'.") from None

    def get_active_key(self) -> tuple[str, str]:
        """Return ``(kid, secret)`` for the active key.

        Raises:
            RuntimeError: If no key has been marked active.
        """
        if self._active_kid is None:
            raise RuntimeError("No active key set. Call set_active() first.")
        return self._active_kid, self._keys[self._active_kid]

    @property
    def kids(self) -> list[str]:
        """Return a sorted list of registered kid values."""
        return sorted(self._keys.keys())

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, kid: str) -> bool:
        return kid in self._keys

    def __repr__(self) -> str:
        active = self._active_kid or "(none)"
        return f"JWTKeyRegistry(kids={self.kids!r}, active={active!r})"



def verify_token_with_kid(
    token: str,
    registry: JWTKeyRegistry,
    expected_type: str,
    clock_skew_seconds: int = 10,
) -> dict[str, Any]:
    """Verify a token using the kid-based key registry.

    This is the recommended verification path when key rotation is
    enabled.  It inspects the unverified header, looks up the secret
    in *registry*, and delegates to :func:`_verify_jwt_token`.

    Args:
        token: A compact-serialised JWT.
        registry: A :class:`JWTKeyRegistry` holding all active keys.
        expected_type: Expected ``type`` claim (``"access"`` or ``"refresh"``).
        clock_skew_seconds: Clock-drift allowance.

    Returns:
        Decoded payload dictionary.

    Raises:
        JWTDecodeError: If the header cannot be decoded or the kid is
            unknown.
        JWTSignatureError / JWTExpiredError: Propagated from verification.
    """
    header = get_unverified_jwt_header(token)
    kid = header.get("kid")

    if kid is None:
        raise JWTDecodeError("Token header is missing 'kid' claim.")

    if kid not in registry:
        raise JWTDecodeError(f"Unknown key id '{kid}' in token header.")

    secret = registry.get_key(kid)
    return _verify_jwt_token(token, expected_type, secret_key=secret, clock_skew_seconds=clock_skew_seconds)



def verify_access_token_with_kid(
    token: str,
    registry: JWTKeyRegistry,
    clock_skew_seconds: int = 10,
) -> dict[str, Any]:
    """Convenience wrapper for access-token verification via kid registry."""
    return verify_token_with_kid(token, registry, "access", clock_skew_seconds)



def verify_refresh_token_with_kid(
    token: str,
    registry: JWTKeyRegistry,
    clock_skew_seconds: int = 10,
) -> dict[str, Any]:
    """Convenience wrapper for refresh-token verification via kid registry."""
    return verify_token_with_kid(token, registry, "refresh", clock_skew_seconds)



def create_jwt_token_with_kid(
    data: dict[str, Any],
    kid: str,
    registry: JWTKeyRegistry,
    expires_in_seconds: int = 3600,
) -> str:
    """Create a JWT whose header includes ``kid`` and is signed by the
    corresponding secret from *registry*.

    Args:
        data: Claims to include in the payload.
        kid: Key identifier — must be registered in *registry*.
        registry: The key registry holding secrets.
        expires_in_seconds: Token lifetime.

    Returns:
        Compact JWT string.

    Raises:
        KeyError: If *kid* is not in the registry.
    """
    secret = registry.get_key(kid)
    # Inject kid into the token creation flow
    header = {"alg": JWT_ALGORITHM, "typ": "JWT", "kid": kid}
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
        secret.encode("utf-8"),
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


def _verify_jwt_token(
    token: str,
    expected_type: str,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Shared implementation for verifying JWT signatures, expiration, and types."""
    if not token or not isinstance(token, str):
        raise ValueError(f"Invalid {expected_type} token: token cannot be empty.")

    token = token.strip()

    if secret_key is None:
        secret_key = os.getenv("JWT_SECRET_KEY", JWT_SECRET_KEY)
    if not secret_key:
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
        secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = base64url_decode(encoded_signature)
    except Exception:
        raise ValueError(
            f"Invalid {expected_type} token: invalid base64 signature encoding."
        )

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError(
            f"Invalid {expected_type} token: signature verification failed."
        )

    try:
        payload_bytes = base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise ValueError(f"Invalid {expected_type} token: malformed JSON payload.")

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {expected_type} token: malformed exp claim.")
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
    secret_key: str | None = None,
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
    secret_key: str | None = None,
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
