"""End-to-end integration tests for the user authentication & 2FA flow (Issue #966).

Consolidates the full auth lifecycle against the SQLite-backed auth module:
user creation -> password verification -> TOTP 2FA enablement -> TOTP token
verification -> 2FA disablement, plus rejection of invalid TOTP codes.

The TOTP verification mirrors the exact logic used by the login flow in
``app/streamlit_app.py`` (``pyotp.TOTP(secret).verify(code.strip())``), so
these tests validate the same behaviour a user experiences when logging in.
"""

import uuid

import pyotp
import pytest

from src.db.auth import (
    add_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    init_db,
    verify_user,
)

PASSWORD = "SecurePass123!"
TEST_OTP_SECRET = "JBSWY3DPEHPK3PXP"
DIFFERENT_OTP_SECRET = "GEZDGNBVGY3TQOJQ"


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Use the mock_db fixture from conftest.py to isolate DB operations."""
    init_db()
    yield


@pytest.fixture
def user_with_2fa() -> str:
    """Create a user with 2FA enabled and return the username."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, PASSWORD, role="teacher")
    enable_2fa(username, TEST_OTP_SECRET)
    return username


def _generate_otp(secret: str) -> str:
    """Return the current valid TOTP code for the given base32 secret."""
    return pyotp.TOTP(secret).now()


def _verify_otp(secret: str, code: str) -> bool:
    """Mirror the app's TOTP verification logic (app/streamlit_app.py)."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code.strip())


def test_full_auth_and_2fa_lifecycle():
    """Run the complete acceptance sequence end-to-end:
    add_user -> verify_user -> enable_2fa -> verify TOTP token -> disable_2fa.
    """
    username = f"user_{uuid.uuid4().hex[:8]}"

    # 1. add_user creates the account
    add_user(username, PASSWORD, role="teacher")

    # 2. verify_user authenticates with the correct password only
    assert verify_user(username, PASSWORD) is True
    assert verify_user(username, "WrongPass123!") is False

    # 3. 2FA is disabled by default
    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None

    # 4. enable_2fa persists the OTP secret and flips the flag
    enable_2fa(username, TEST_OTP_SECRET)
    enabled, secret = get_2fa_status(username)
    assert enabled is True
    assert secret == TEST_OTP_SECRET

    # 5. A freshly generated TOTP token passes the app's verification logic
    valid_code = _generate_otp(TEST_OTP_SECRET)
    assert _verify_otp(TEST_OTP_SECRET, valid_code) is True

    # 6. disable_2fa clears the secret; password auth still works
    disable_2fa(username)
    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None
    assert verify_user(username, PASSWORD) is True


def test_valid_totp_code_passes(user_with_2fa):
    """A freshly generated TOTP token passes the app's verification logic."""
    valid_code = _generate_otp(TEST_OTP_SECRET)
    assert _verify_otp(TEST_OTP_SECRET, valid_code) is True


def test_invalid_totp_code_fails(user_with_2fa):
    """An invalid TOTP code must fail the 2FA check."""
    valid_code = _generate_otp(TEST_OTP_SECRET)
    invalid_code = str((int(valid_code) + 1) % 1_000_000).zfill(6)

    assert invalid_code != valid_code
    assert _verify_otp(TEST_OTP_SECRET, invalid_code) is False


def test_totp_code_from_different_secret_fails(user_with_2fa):
    """A TOTP code generated from another account's secret must fail."""
    foreign_code = _generate_otp(DIFFERENT_OTP_SECRET)
    assert _verify_otp(TEST_OTP_SECRET, foreign_code) is False


def test_empty_totp_code_fails(user_with_2fa):
    """A blank verification code must fail the 2FA check."""
    assert _verify_otp(TEST_OTP_SECRET, "") is False
    assert _verify_otp(TEST_OTP_SECRET, "   ") is False
