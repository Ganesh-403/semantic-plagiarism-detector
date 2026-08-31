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
    create_jwt_token_with_kid,
    get_unverified_jwt_header,
    JWTKeyRegistry,
    verify_access_token,
    verify_access_token_with_kid,
    verify_refresh_token,
    verify_refresh_token_with_kid,
    verify_token_with_kid,
    JWTDecodeError,
    JWTExpiredError,
    JWTNotYetValidError,
    JWTSignatureError,
)


@pytest.fixture(autouse=True)
def setup_jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "testing-secret-key-1234567890-test")
    monkeypatch.setattr(jwt_utils, "_IS_TEST", True)
    monkeypatch.setattr(
        jwt_utils,
        "VALID_STATIC_REFRESH_TOKENS",
        {
            "dev-refresh-token",
            "valid-refresh-token",
            "test-refresh-token",
            "sample-refresh-token",
        },
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


def test_static_refresh_token_rejected_in_production(monkeypatch, caplog):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(jwt_utils, "_IS_TEST", False)

    with pytest.raises(JWTDecodeError, match="static testing tokens are not allowed"):
        verify_refresh_token("valid-refresh-token")

    assert "SECURITY ALERT" in caplog.text


def test_invalid_signature():
    token = create_access_token(sub="charlie")
    header, payload, _sig = token.split(".")
    tampered_token = f"{header}.{payload}.tampered_signature"

    with pytest.raises(JWTSignatureError, match="signature verification failed"):
        verify_access_token(tampered_token)


def test_expired_token():
    token = create_jwt_token({"sub": "david", "type": "access"}, expires_in_seconds=-10)
    with pytest.raises(JWTExpiredError, match="expired"):
        verify_access_token(token)


def test_nbf_claim_future_token_rejected():
    import time
    future_nbf = int(time.time()) + 3600
    token = create_jwt_token({"sub": "future_user", "type": "access", "nbf": future_nbf})
    with pytest.raises(JWTNotYetValidError, match="is not yet valid \\(nbf\\)"):
        verify_access_token(token)


def test_wrong_token_type():
    access_token = create_access_token(sub="eve")
    with pytest.raises(JWTDecodeError, match="expected 'refresh'"):
        verify_refresh_token(access_token)

    refresh_token = create_refresh_token(sub="eve")
    with pytest.raises(JWTDecodeError, match="expected 'access'"):
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


def test_expired_token_with_clock_skew():
    """Verify that verify_access_token respects clock_skew_seconds tolerance when check token expiration."""
    # 1. Create a token that expired 5 seconds ago
    token = create_jwt_token({"sub": "skew_user", "type": "access"}, expires_in_seconds=-5)

    # 2. Verify with default clock skew (10 seconds), it should PASS (since -5 is within +10 skew)
    payload = verify_access_token(token)
    assert payload["sub"] == "skew_user"

    # 3. Verify with custom clock skew (3 seconds), it should FAIL (since -5 is past +3 skew)
    with pytest.raises(JWTExpiredError, match="expired"):
        verify_access_token(token, clock_skew_seconds=3)


# ---------------------------------------------------------------------------
# Tests for get_unverified_jwt_header
# ---------------------------------------------------------------------------


class TestGetUnverifiedJwtHeader:
    """Tests for the get_unverified_jwt_header helper (Issue #3688)."""

    def test_returns_header_dict(self):
        token = create_access_token(sub="test")
        header = get_unverified_jwt_header(token)
        assert isinstance(header, dict)
        assert header["alg"] == "HS256"
        assert header["typ"] == "JWT"

    def test_header_from_refresh_token(self):
        token = create_refresh_token(sub="test")
        header = get_unverified_jwt_header(token)
        assert header["alg"] == "HS256"
        assert header["typ"] == "JWT"

    def test_header_does_not_verify_signature(self):
        token = create_access_token(sub="test")
        header_parts = token.split(".")
        tampered = f"{header_parts[0]}.{header_parts[1]}.tampered"
        header = get_unverified_jwt_header(tampered)
        assert header["alg"] == "HS256"

    def test_empty_token_raises(self):
        with pytest.raises(JWTDecodeError, match="non-empty string"):
            get_unverified_jwt_header("")

    def test_none_token_raises(self):
        with pytest.raises(JWTDecodeError, match="non-empty string"):
            get_unverified_jwt_header(None)  # type: ignore[arg-type]

    def test_non_string_token_raises(self):
        with pytest.raises(JWTDecodeError, match="non-empty string"):
            get_unverified_jwt_header(123)  # type: ignore[arg-type]

    def test_wrong_segment_count_two(self):
        with pytest.raises(JWTDecodeError, match="expected 3"):
            get_unverified_jwt_header("a.b")

    def test_wrong_segment_count_four(self):
        with pytest.raises(JWTDecodeError, match="expected 3"):
            get_unverified_jwt_header("a.b.c.d")

    def test_empty_header_segment(self):
        with pytest.raises(JWTDecodeError, match="header segment is empty"):
            get_unverified_jwt_header(".payload.sig")

    def test_invalid_base64_header(self):
        with pytest.raises(JWTDecodeError, match="invalid base64url"):
            get_unverified_jwt_header("!!!.payload.sig")

    def test_non_json_header(self):
        import base64
        raw = base64.urlsafe_b64encode(b"not-json").rstrip(b"=").decode()
        with pytest.raises(JWTDecodeError, match="not valid JSON"):
            get_unverified_jwt_header(f"{raw}.payload.sig")

    def test_header_is_array_not_object(self):
        import base64
        raw = base64.urlsafe_b64encode(b"[1,2,3]").rstrip(b"=").decode()
        with pytest.raises(JWTDecodeError, match="not a JSON object"):
            get_unverified_jwt_header(f"{raw}.payload.sig")

    def test_token_with_whitespace(self):
        token = create_access_token(sub="test")
        padded = f"  {token}  "
        header = get_unverified_jwt_header(padded)
        assert header["alg"] == "HS256"

    def test_header_preserves_kid_when_present(self):
        """When a kid is injected via create_jwt_token_with_kid, it should appear."""
        registry = JWTKeyRegistry()
        registry.register_key("key-1", "testing-secret-key-1234567890-test")
        token = create_jwt_token_with_kid(
            {"sub": "test", "type": "access"},
            kid="key-1",
            registry=registry,
        )
        header = get_unverified_jwt_header(token)
        assert header["kid"] == "key-1"


# ---------------------------------------------------------------------------
# Tests for JWTKeyRegistry
# ---------------------------------------------------------------------------


class TestJWTKeyRegistry:
    """Tests for the JWTKeyRegistry class."""

    def test_register_and_get(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "secret-1")
        assert reg.get_key("k1") == "secret-1"

    def test_register_overwrites(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "old")
        reg.register_key("k1", "new")
        assert reg.get_key("k1") == "new"

    def test_get_missing_key_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(KeyError, match="No secret registered"):
            reg.get_key("nonexistent")

    def test_retire_key(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "s")
        assert reg.retire_key("k1") is True
        assert "k1" not in reg

    def test_retire_missing_returns_false(self):
        reg = JWTKeyRegistry()
        assert reg.retire_key("nope") is False

    def test_retire_active_key_clears_active(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "s")
        reg.set_active("k1")
        reg.retire_key("k1")
        with pytest.raises(RuntimeError, match="No active key"):
            reg.get_active_key()

    def test_set_active(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "s")
        reg.set_active("k1")
        kid, secret = reg.get_active_key()
        assert kid == "k1"
        assert secret == "s"

    def test_set_active_unknown_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.set_active("nope")

    def test_get_active_none_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(RuntimeError, match="No active key"):
            reg.get_active_key()

    def test_kids_sorted(self):
        reg = JWTKeyRegistry()
        reg.register_key("c", "s")
        reg.register_key("a", "s")
        reg.register_key("b", "s")
        assert reg.kids == ["a", "b", "c"]

    def test_len(self):
        reg = JWTKeyRegistry()
        assert len(reg) == 0
        reg.register_key("k1", "s")
        assert len(reg) == 1
        reg.register_key("k2", "s")
        assert len(reg) == 2

    def test_contains(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "s")
        assert "k1" in reg
        assert "k2" not in reg

    def test_register_empty_kid_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(ValueError, match="kid must be"):
            reg.register_key("", "s")

    def test_register_empty_secret_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(ValueError, match="secret must be"):
            reg.register_key("k1", "")

    def test_repr(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "s")
        reg.set_active("k1")
        r = repr(reg)
        assert "k1" in r
        assert "k1" in r


# ---------------------------------------------------------------------------
# Tests for verify_token_with_kid and convenience wrappers
# ---------------------------------------------------------------------------


class TestVerifyTokenWithKid:
    """Tests for kid-based token verification."""

    def _make_registry(self):
        reg = JWTKeyRegistry()
        reg.register_key("key-2024", "testing-secret-key-1234567890-test")
        reg.register_key("key-2025", "another-testing-secret-key-12345")
        reg.set_active("key-2025")
        return reg

    def test_verify_with_correct_kid(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "alice", "type": "access"},
            kid="key-2024",
            registry=reg,
        )
        payload = verify_token_with_kid(token, reg, "access")
        assert payload["sub"] == "alice"
        assert payload["type"] == "access"

    def test_verify_wrong_kid_fails(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "bob", "type": "access"},
            kid="key-2024",
            registry=reg,
        )
        # Tamper the header kid to key-2025 (different secret)
        parts = token.split(".")
        import base64, json
        header_bytes = base64.urlsafe_b64decode(parts[0] + "==")
        header = json.loads(header_bytes)
        header["kid"] = "key-2025"
        new_header = base64.urlsafe_b64encode(
            json.dumps(header, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        tampered = f"{new_header}.{parts[1]}.{parts[2]}"
        with pytest.raises(JWTSignatureError):
            verify_token_with_kid(tampered, reg, "access")

    def test_verify_unknown_kid_raises(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "carol", "type": "access"},
            kid="key-2024",
            registry=reg,
        )
        # Remove the key so kid is unknown
        reg.retire_key("key-2024")
        with pytest.raises(JWTDecodeError, match="Unknown key id"):
            verify_token_with_kid(token, reg, "access")

    def test_verify_missing_kid_in_header(self):
        reg = self._make_registry()
        # Token created without kid
        token = create_access_token(sub="dave")
        with pytest.raises(JWTDecodeError, match="missing 'kid'"):
            verify_token_with_kid(token, reg, "access")

    def test_convenience_access_wrapper(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "eve", "type": "access"},
            kid="key-2024",
            registry=reg,
        )
        payload = verify_access_token_with_kid(token, reg)
        assert payload["sub"] == "eve"

    def test_convenience_refresh_wrapper(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "frank", "type": "refresh"},
            kid="key-2024",
            registry=reg,
        )
        payload = verify_refresh_token_with_kid(token, reg)
        assert payload["sub"] == "frank"
        assert payload["type"] == "refresh"

    def test_wrong_type_rejected(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "grace", "type": "access"},
            kid="key-2024",
            registry=reg,
        )
        with pytest.raises(JWTDecodeError, match="expected 'refresh'"):
            verify_token_with_kid(token, reg, "refresh")

    def test_expired_token_rejected(self):
        reg = self._make_registry()
        token = create_jwt_token_with_kid(
            {"sub": "hank", "type": "access"},
            kid="key-2024",
            registry=reg,
            expires_in_seconds=-60,
        )
        with pytest.raises(JWTExpiredError):
            verify_token_with_kid(token, reg, "access")

    def test_key_rotation_scenario(self):
        """Simulate key rotation: old tokens still verify with old key."""
        reg = JWTKeyRegistry()
        reg.register_key("v1", "testing-secret-key-1234567890-test")
        reg.register_key("v2", "another-testing-secret-key-12345")
        reg.set_active("v2")

        old_token = create_jwt_token_with_kid(
            {"sub": "user", "type": "access"},
            kid="v1",
            registry=reg,
        )
        new_token = create_jwt_token_with_kid(
            {"sub": "user", "type": "access"},
            kid="v2",
            registry=reg,
        )

        # Both verify with their respective keys
        p1 = verify_token_with_kid(old_token, reg, "access")
        p2 = verify_token_with_kid(new_token, reg, "access")
        assert p1["sub"] == "user"
        assert p2["sub"] == "user"

        # Retire v1 — old token no longer verifies
        reg.retire_key("v1")
        with pytest.raises(JWTDecodeError, match="Unknown key id"):
            verify_token_with_kid(old_token, reg, "access")

        # New token still works
        p3 = verify_token_with_kid(new_token, reg, "access")
        assert p3["sub"] == "user"


class TestCreateJwtTokenWithKid:
    """Tests for create_jwt_token_with_kid."""

    def test_kid_in_header(self):
        reg = JWTKeyRegistry()
        reg.register_key("mykid", "testing-secret-key-1234567890-test")
        token = create_jwt_token_with_kid(
            {"sub": "x", "type": "access"},
            kid="mykid",
            registry=reg,
        )
        header = get_unverified_jwt_header(token)
        assert header["kid"] == "mykid"

    def test_unknown_kid_raises(self):
        reg = JWTKeyRegistry()
        with pytest.raises(KeyError, match="No secret registered"):
            create_jwt_token_with_kid(
                {"sub": "x", "type": "access"},
                kid="nope",
                registry=reg,
            )

    def test_roundtrip_with_kid(self):
        reg = JWTKeyRegistry()
        reg.register_key("k1", "testing-secret-key-1234567890-test")
        token = create_jwt_token_with_kid(
            {"sub": "roundtrip", "type": "access", "scopes": ["read"]},
            kid="k1",
            registry=reg,
            expires_in_seconds=7200,
        )
        payload = verify_token_with_kid(token, reg, "access")
        assert payload["sub"] == "roundtrip"
        assert payload["scopes"] == ["read"]
        assert "jti" not in payload or isinstance(payload.get("jti"), str)


def test_rs256_token_creation_and_verification(monkeypatch):
    """Test generating and verifying RS256 asymmetric signed JWT tokens."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate test RSA keypair
    private_key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    public_pem = private_key_obj.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("JWT_PRIVATE_KEY", private_pem)
    monkeypatch.setenv("JWT_PUBLIC_KEY", public_pem)

    token = create_access_token(sub="rsa_user", scopes=["admin"])
    header = get_unverified_jwt_header(token)
    assert header["alg"] == "RS256"

    payload = verify_access_token(token)
    assert payload["sub"] == "rsa_user"
    assert payload["type"] == "access"


def test_rs256_missing_keys_raises_value_error(monkeypatch):
    """Test ValueError is raised if RS256 is configured without public/private keys."""
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.delenv("JWT_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)

    with pytest.raises(ValueError, match="JWT_PRIVATE_KEY"):
        create_access_token(sub="user")


