"""
test_2fa_setup_qr_code_issue_4040.py
------------------------------------
Unit tests for Issue #4040: Generating TOTP QR code data URI in 2FA setup endpoint.
"""

from __future__ import annotations

import base64
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routers.auth import generate_totp_qr_code_data_uri
from src.db.auth import get_2fa_status, init_db

init_db()
client = TestClient(app)


def test_generate_totp_qr_code_data_uri():
    """Verify generate_totp_qr_code_data_uri generates a valid base64 PNG data URI."""
    otpauth_url = "otpauth://totp/SemanticPlagiarismDetector:testuser?secret=JBSWY3DPEHPK3PXP&issuer=SemanticPlagiarismDetector"
    data_uri = generate_totp_qr_code_data_uri(otpauth_url)

    assert data_uri.startswith("data:image/png;base64,")
    b64_part = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_part)
    # Check PNG magic bytes header (\x89PNG\r\n\x1a\n)
    assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_2fa_setup_endpoint_returns_qr_code_data_uri():
    """Verify POST /api/v1/auth/2fa/setup returns secret, otpauth URL, and base64 PNG QR code data URI."""
    response = client.post(
        "/api/v1/auth/2fa/setup",
        json={"username": "admin_test_4040_cb4dffb6", "issuer": "TestIssuer"},
    )
    assert response.status_code == 200
    data = response.json()

    assert "secret" in data
    assert "otpauth_url" in data
    assert "qr_code_data_uri" in data
    assert "message" in data

    assert data["otpauth_url"].startswith("otpauth://totp/")
    assert "admin_test_4040_cb4dffb6" in data["otpauth_url"]
    assert "TestIssuer" in data["otpauth_url"]
    assert data["qr_code_data_uri"].startswith("data:image/png;base64,")

    # Verify user 2FA status in DB is enabled
    enabled, secret = get_2fa_status("admin_test_4040_cb4dffb6")
    assert enabled is True
    assert secret == data["secret"]


def test_2fa_setup_endpoint_legacy_path():
    """Verify POST /auth/2fa/setup endpoint path also works."""
    response = client.post(
        "/auth/2fa/setup",
        json={"username": "admin_legacy_4040_cb4dffb6"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["qr_code_data_uri"].startswith("data:image/png;base64,")
