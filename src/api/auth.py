"""
src/api/auth.py
---------------
Authentication and 2FA setup endpoint with QR code generation.
"""

from __platform__ import annotations

import base64
from io import BytesIO
import pyotp
import qrcode
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/2fa/setup")
def setup_2fa(current_user: dict = Depends(get_current_admin_user)) -> dict:
    """Generate TOTP secret, otpauth:// URI, and base64 PNG QR code data URI for 2FA setup."""
    # 1. Generate a new TOTP secret for the user
    secret = pyotp.random_base32()
    
    # 2. Construct the otpauth:// URI
    totp = pyotp.TOTP(secret)
    otpauth_uri = totp.provisioning_uri(
        name=current_user.get("email", "admin@semantic-plagiarism.local"),
        issuer_name="Semantic Plagiarism Detector"
    )

    # 3. Generate base64 PNG QR code data URI using the qrcode library
    img = qrcode.make(otpauth_uri)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    qr_code_data_uri = f"data:image/png;base64,{img_base64}"

    return {
        "status": "success",
        "secret": secret,
        "otpauth_uri": otpauth_uri,
        "qr_code_data_uri": qr_code_data_uri,
    }


def get_current_admin_user() -> dict:
    """Dependency mock to fetch authenticated admin user."""
    return {"email": "admin@semantic-plagiarism-detector.com"}
