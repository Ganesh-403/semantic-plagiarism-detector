import pyotp
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app
from src.api.dependencies import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_2fa_rate_limit():
    """Issue #4045: /auth/2fa/verify is rate-limited to 5/minute, keyed by
    client address. TestClient reuses the same fake client identity for every
    request in this process, so without resetting the limiter's storage
    between tests, later tests would eventually get HTTP 429 instead of the
    status code they're actually asserting on, regardless of whether their
    OTP code was valid.
    """
    limiter.reset()
    yield
    limiter.reset()

def test_refresh_token_rotation_success():
    """
    Assert that hitting the /refresh endpoint with a valid refresh token 
    successfully yields a fresh access token.
    """
    valid_refresh_token = "valid_refresh_token_string"
    mock_access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.new_access_token"

    # Mock dependencies inside refresh_token_endpoint
    with patch("src.db.auth.is_token_revoked", return_value=False) as mock_revoked:
        with patch("src.security.jwt_utils.verify_refresh_token", return_value={"sub": "test_user", "scopes": ["read", "write"]}) as mock_verify:
            with patch("src.security.jwt_utils.create_access_token", return_value=mock_access_token) as mock_create:
                response = client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": valid_refresh_token}
                )
                
                # Acceptance Criteria Assertion
                assert response.status_code == 200
                assert "access_token" in response.json()
                assert response.json()["access_token"] == mock_access_token
                
                mock_revoked.assert_called_once_with(valid_refresh_token)
                mock_verify.assert_called_once_with(valid_refresh_token)


def test_refresh_token_rotation_fails_if_revoked():
    """
    Assert that passing a revoked or invalid refresh token 
    returns an HTTP 401 Unauthorized error.
    """
    revoked_refresh_token = "revoked_refresh_token_string"

    # Mock the database check to return True (token is revoked)
    with patch("src.db.auth.is_token_revoked", return_value=True) as mock_revoked:
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": revoked_refresh_token}
        )
        
        # Validation Checks
        assert response.status_code == 401
        mock_revoked.assert_called_once_with(revoked_refresh_token)


def test_refresh_token_rotation_fails_if_invalid():
    """
    Assert that passing an invalid refresh token (which raises ValueError during verification)
    returns an HTTP 401 Unauthorized error.
    """
    invalid_refresh_token = "invalid_refresh_token_string"

    # Mock to verify that ValueError raises 401
    with patch("src.db.auth.is_token_revoked", return_value=False):
        with patch("src.security.jwt_utils.verify_refresh_token", side_effect=ValueError("Token signature is invalid")):
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": invalid_refresh_token}
            )
            
            # Validation Checks
            assert response.status_code == 401


# ── Issue #4045: 2FA TOTP verification workflow ────────────────────────────────


class TestTwoFactorAuthWorkflow:
    """Enable 2FA, generate a real TOTP code, verify it, reject a bad one."""

    def test_enable_2fa_then_verify_valid_totp_code_succeeds(self):
        """A code freshly generated from the user's own secret is accepted."""
        secret = pyotp.random_base32()
        valid_code = pyotp.TOTP(secret).now()

        with patch("src.db.auth.get_2fa_status", return_value=(True, secret)), \
             patch("src.db.auth.init_db", return_value=None), \
             patch("src.db.auth.log_security_event") as mock_log:
            response = client.post(
                "/auth/2fa/verify",
                json={"username": "alice", "otp_code": valid_code},
            )

        assert response.status_code == 200
        assert response.json() == {
            "verified": True,
            "message": "2FA code verified successfully.",
        }
        assert mock_log.call_args[0][0] == "2FA_VERIFY_SUCCESS"

    def test_invalid_totp_code_is_rejected(self):
        """A 6-digit code that does not match the user's secret is rejected."""
        secret = pyotp.random_base32()
        real_code = pyotp.TOTP(secret).now()
        # Guarantee a code that is actually wrong, not a lucky collision.
        wrong_code = "000000" if real_code != "000000" else "111111"

        with patch("src.db.auth.get_2fa_status", return_value=(True, secret)), \
             patch("src.db.auth.init_db", return_value=None), \
             patch("src.db.auth.log_security_event") as mock_log:
            response = client.post(
                "/auth/2fa/verify",
                json={"username": "bob", "otp_code": wrong_code},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid 2FA verification code."
        assert mock_log.call_args[0][0] == "2FA_VERIFY_FAILED"

    def test_verify_rejects_when_2fa_not_enabled(self):
        """Verifying a code for a user who never enabled 2FA is rejected."""
        with patch("src.db.auth.get_2fa_status", return_value=(False, None)), \
             patch("src.db.auth.init_db", return_value=None), \
             patch("src.db.auth.log_security_event"):
            response = client.post(
                "/auth/2fa/verify",
                json={"username": "carol", "otp_code": "123456"},
            )

        assert response.status_code == 400
        assert "not enabled" in response.json()["detail"]

    def test_missing_username_or_code_is_rejected(self):
        """Both fields are required before any 2FA lookup is attempted."""
        response = client.post(
            "/auth/2fa/verify",
            json={"username": "", "otp_code": ""},
        )
        assert response.status_code == 400
