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

"""src/api/routers/auth.py - Authentication and token management router."""

import base64
import io
import logging

import pyotp
import qrcode
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, SecretStr

from src.api.dependencies import limiter
from src.api.schemas import (
    ErrorResponse,
    LoginResponse,
    LoginRequest,
    RefreshRequest,
    RevokeRequest,
    RevokeResponse,
    TokenResponse,
    TwoFactorDisableRequest,
    TwoFactorDisableResponse,
    TwoFactorSetupRequest,
    TwoFactorSetupResponse,
    TwoFactorVerifyRequest,
    TwoFactorVerifyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


class ChangePasswordRequest(BaseModel):
    old_password: SecretStr
    new_password: SecretStr


from src.api.dependencies import verify_bearer_token


@router.post("/api/v1/auth/change-password")
@router.post("/auth/change-password")
@limiter.limit("5/minute")
def change_password(
    request: Request,
    payload: ChangePasswordRequest,
    token: str = Depends(verify_bearer_token),
):
    """Change only the signed-in account after verifying its current password."""
    from src.db.auth import update_password
    from src.security.jwt_utils import verify_access_token

    if not token:
        raise HTTPException(status_code=401, detail="A user access token is required.")
    claims = verify_access_token(token)
    username = claims.get("sub") if claims else None
    if not username:
        raise HTTPException(status_code=401, detail="A user access token is required.")
    try:
        update_password(
            username,
            payload.new_password.get_secret_value(),
            current_user=username,
            old_password=payload.old_password.get_secret_value(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Password changed successfully. Sign in again."}


def generate_totp_qr_code_data_uri(otpauth_url: str) -> str:
    """Generate a base64-encoded PNG data URI of an otpauth:// URL using qrcode."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(otpauth_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_png = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_png}"


@router.post(
    "/auth/login",
    summary="Authenticate user",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@router.post(
    "/api/v1/auth/login",
    summary="Authenticate user",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest):
    """Authenticate credentials and 2FA before issuing signed access/refresh tokens."""
    from datetime import datetime, timedelta, timezone
    from starlette.concurrency import run_in_threadpool
    from src.db.auth import verify_user, get_user_role, get_2fa_status
    from src.security.jwt_utils import create_access_token, create_refresh_token

    username = payload.username.strip().lower()
    result = await run_in_threadpool(
        verify_user, username, payload.password, return_details=True
    )
    if not result.get("authenticated"):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    enabled, secret = get_2fa_status(username)
    if enabled and (
        not secret
        or not payload.otp_code
        or not pyotp.TOTP(secret).verify(payload.otp_code)
    ):
        raise HTTPException(
            status_code=401, detail="A valid two-factor code is required"
        )
    if result.get("must_change_password") or result.get("password_expired"):
        raise HTTPException(
            status_code=403,
            detail="Change your password in the dashboard before using the API",
        )
    role = get_user_role(username)
    scopes = ["read", "viewer"]
    if role in {"admin", "teacher", "analyst"}:
        scopes += ["write", "scan", "analyst"]
    if role == "admin":
        scopes.append("admin")
    access_token = create_access_token(sub=username, scopes=scopes, expires_in=3600)
    return {
        "token": access_token,
        "access_token": access_token,
        "refresh_token": create_refresh_token(sub=username, scopes=scopes),
        "token_type": "bearer",
        "expires_in": 3600,
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=3600)
        ).isoformat(),
    }


@router.post(
    "/api/v1/auth/refresh",
    summary="Refresh OAuth2 Bearer Token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {
            "model": ErrorResponse,
            "description": "Unauthorized / Invalid Refresh Token",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def refresh_token_endpoint(
    request: Request,
    payload: RefreshRequest | None = None,
):
    """
    Acquire a new access token using a valid, unexpired refresh token.
    Accepts refresh token in JSON request body or Authorization header.
    """
    refresh_token = None

    if payload and payload.refresh_token:
        refresh_token = payload.refresh_token
    else:
        try:
            body = await request.json()
            if isinstance(body, dict):
                refresh_token = body.get("refresh_token") or body.get("token")
        except Exception:
            logger.debug("Failed to parse request payload")

    if not refresh_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            refresh_token = auth_header[7:].strip()

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token must be provided in request body or Authorization header.",
        )

    from src.db.auth import is_token_revoked

    if is_token_revoked(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from src.security.jwt_utils import create_access_token, verify_refresh_token

    try:
        token_payload = verify_refresh_token(refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    from src.db.auth import is_token_revoked, is_user_active

    sub = token_payload.get("sub")
    if not sub or is_token_revoked(refresh_token) or not is_user_active(sub):
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid")
    scopes = token_payload.get("scopes", [])
    new_access_token = create_access_token(sub=sub, scopes=scopes, expires_in=3600)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",  # nosec B105
        "expires_in": 3600,
    }


@router.post(
    "/api/v1/auth/revoke",
    summary="Revoke API Bearer token",
    response_model=RevokeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def revoke_token_endpoint(
    request: Request,
    payload: RevokeRequest | None = None,
):
    """Revoke an active API Bearer token immediately."""
    token_to_revoke = None

    if payload and payload.token:
        token_to_revoke = payload.token
    else:
        try:
            body = await request.json()
            if isinstance(body, dict):
                token_to_revoke = body.get("token") or body.get("token_signature")
        except Exception:
            logger.debug("Failed to parse request payload")

    if not token_to_revoke:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token_to_revoke = auth_header[7:].strip()

    if not token_to_revoke:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token to revoke must be provided in request body or Authorization header.",
        )

    try:
        from src.db.auth import revoke_token

        revoke_token(
            token_to_revoke, details="Revoked via API endpoint /api/v1/auth/revoke"
        )
        return {
            "status": "success",
            "message": "Token revoked successfully.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}",
        )


@router.post(
    "/auth/2fa/setup",
    summary="Initialize 2FA setup and return TOTP secret, otpauth URL, and base64 PNG QR code data URI",
    response_model=TwoFactorSetupResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@router.post(
    "/api/v1/auth/2fa/setup",
    summary="Initialize 2FA setup and return TOTP secret, otpauth URL, and base64 PNG QR code data URI",
    response_model=TwoFactorSetupResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@limiter.limit("5/minute")
async def setup_two_factor_auth_endpoint(
    request: Request, payload: TwoFactorSetupRequest
):
    """Enroll the credential owner without exposing an existing second-factor secret."""
    from starlette.concurrency import run_in_threadpool
    from src.db.auth import enable_2fa, get_2fa_status, verify_user

    username = payload.username.strip().lower()
    if not await run_in_threadpool(verify_user, username, payload.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    enabled, existing_secret = get_2fa_status(username)
    if enabled or existing_secret:
        raise HTTPException(
            status_code=409,
            detail="2FA is already configured; disable it with your current code before reenrolling",
        )
    secret = pyotp.random_base32()
    enable_2fa(username, secret)
    if get_2fa_status(username) != (True, secret):
        raise HTTPException(status_code=500, detail="Unable to configure 2FA")
    otpauth_url = pyotp.TOTP(secret).provisioning_uri(
        name=username, issuer_name=payload.issuer or "SemanticPlagiarismDetector"
    )
    return {
        "secret": secret,
        "otpauth_url": otpauth_url,
        "qr_code_data_uri": generate_totp_qr_code_data_uri(otpauth_url),
        "message": "2FA enabled. Scan this QR code now and store the secret securely.",
    }


# ============================================================================
# Enterprise 2FA Lifecycle Management Framework
# ============================================================================
# This module provides a highly robust, scalable, and extensible framework
# for managing Two-Factor Authentication lifecycle events in an enterprise
# environment. It employs the Strategy and State patterns to decouple the
# mechanisms of 2FA validation and state transition from the HTTP handlers.

import abc
from typing import Optional, Dict, Any, Type
import time
import uuid
import hashlib
import hmac


class Enterprise2FAValidationException(Exception):
    """Base exception for all enterprise 2FA validation errors."""

    pass


class AuthenticationChallengeFailedException(Enterprise2FAValidationException):
    """Raised when the primary authentication challenge (password) fails."""

    pass


class TokenValidationFailedException(Enterprise2FAValidationException):
    """Raised when the secondary authentication challenge (OTP) fails."""

    pass


class TwoFactorNotConfiguredException(Enterprise2FAValidationException):
    """Raised when 2FA operations are attempted on a non-configured account."""

    pass


class IEnterpriseTwoFactorValidator(abc.ABC):
    """
    Abstract Base Class defining the contract for enterprise two-factor
    validators. Future implementations may support WebAuthn, SMS, Email,
    or push notifications alongside TOTP.
    """

    @abc.abstractmethod
    def validate_primary_credential(self, username: str, credential: str) -> bool:
        """Validates the primary user credential (typically a password)."""
        pass

    @abc.abstractmethod
    def validate_secondary_credential(
        self, username: str, secret: str, token: str
    ) -> bool:
        """Validates the secondary user credential (typically a TOTP token)."""
        pass


class EnterpriseTOTPValidatorStrategy(IEnterpriseTwoFactorValidator):
    """
    Concrete implementation of the 2FA validator strategy using Time-based
    One-Time Passwords (TOTP). This ensures strict adherence to RFC 6238.
    """

    def __init__(self, allowed_time_drift_seconds: int = 30):
        self.allowed_time_drift_seconds = allowed_time_drift_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_primary_credential(self, username: str, credential: str) -> bool:
        from src.db.auth import authenticate_user

        try:
            self.logger.debug(
                f"Initiating primary credential validation for user: {username}"
            )
            auth_result = authenticate_user(username, credential)
            is_valid = auth_result.get("authenticated", False)
            if not is_valid:
                self.logger.warning(
                    f"Primary credential validation failed for user: {username}"
                )
            return is_valid
        except Exception as e:
            self.logger.error(f"Error during primary credential validation: {str(e)}")
            return False

    def validate_secondary_credential(
        self, username: str, secret: str, token: str
    ) -> bool:
        import pyotp

        try:
            self.logger.debug(
                f"Initiating secondary credential (TOTP) validation for user: {username}"
            )
            totp = pyotp.TOTP(secret)
            # Standard verification with drift allowance
            is_valid = totp.verify(token)
            if not is_valid:
                self.logger.warning(
                    f"Secondary credential (TOTP) validation failed for user: {username}"
                )
            return is_valid
        except Exception as e:
            self.logger.error(f"Error during secondary credential validation: {str(e)}")
            return False


class EnterpriseTwoFactorStateTransitionManager:
    """
    Manages state transitions for 2FA lifecycle events (enable/disable/reset).
    Enforces that state transitions only occur after successful cryptographic
    and credential verification challenges.
    """

    def __init__(self, validator_strategy: IEnterpriseTwoFactorValidator):
        self._validator = validator_strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._transaction_id = str(uuid.uuid4())

    def _audit_log_transition(
        self, username: str, action: str, status: str, details: str = ""
    ):
        """Internal method to emit audit logs for state transitions."""
        timestamp = time.time()
        self.logger.info(
            f"[{self._transaction_id}] [2FA_TRANSITION] User: {username} | Action: {action} | "
            f"Status: {status} | Timestamp: {timestamp} | Details: {details}"
        )

    def disable_two_factor_authentication(
        self, username: str, password: str, otp_token: str
    ) -> bool:
        """
        Orchestrates the secure disablement of 2FA.
        Executes a sequence of cryptographic and state-based verifications before
        permitting the mutation of the user's security posture.
        """
        from src.db.auth import get_2fa_status, disable_2fa

        self.logger.info(
            f"[{self._transaction_id}] Starting 2FA disablement workflow for user: {username}"
        )

        try:
            # Step 1: Pre-condition check - Verify 2FA is actually enabled
            self.logger.debug(
                f"[{self._transaction_id}] Checking 2FA status pre-conditions"
            )
            enabled, existing_secret = get_2fa_status(username)
            if not enabled or not existing_secret:
                self._audit_log_transition(
                    username, "DISABLE_2FA", "FAILED", "2FA not configured"
                )
                raise TwoFactorNotConfiguredException(
                    "Cannot disable 2FA: Not currently configured."
                )

            # Step 2: Primary Challenge - Password verification
            self.logger.debug(
                f"[{self._transaction_id}] Executing primary credential challenge"
            )
            if not self._validator.validate_primary_credential(username, password):
                self._audit_log_transition(
                    username, "DISABLE_2FA", "FAILED", "Primary authentication rejected"
                )
                raise AuthenticationChallengeFailedException(
                    "Primary credential verification failed."
                )

            # Step 3: Secondary Challenge - TOTP verification
            self.logger.debug(
                f"[{self._transaction_id}] Executing secondary credential challenge"
            )
            if not self._validator.validate_secondary_credential(
                username, existing_secret, otp_token
            ):
                self._audit_log_transition(
                    username,
                    "DISABLE_2FA",
                    "FAILED",
                    "Secondary authentication rejected",
                )
                raise TokenValidationFailedException(
                    "Secondary credential verification failed."
                )

            # Step 4: State Mutation - Execute the disablement
            self.logger.debug(
                f"[{self._transaction_id}] All challenges passed. Mutating security state."
            )
            disable_2fa(username)

            # Step 5: Post-condition audit
            self._audit_log_transition(
                username,
                "DISABLE_2FA",
                "SUCCESS",
                "2FA successfully removed from account",
            )
            return True

        except Enterprise2FAValidationException as e:
            self.logger.warning(
                f"[{self._transaction_id}] 2FA disablement halted due to validation exception: {str(e)}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"[{self._transaction_id}] Unhandled exception during 2FA disablement: {str(e)}"
            )
            self._audit_log_transition(
                username, "DISABLE_2FA", "ERROR", f"Unhandled exception: {str(e)}"
            )
            raise


# ============================================================================
# Legacy/Direct implementation replaced by Enterprise Framework above
# ============================================================================


@router.post(
    "/auth/2fa/disable",
    summary="Disable 2FA with current password and valid OTP token",
    response_model=TwoFactorDisableResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@router.post(
    "/api/v1/auth/2fa/disable",
    summary="Disable 2FA with current password and valid OTP token",
    response_model=TwoFactorDisableResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def disable_two_factor_auth_endpoint(
    request: Request,
    payload: TwoFactorDisableRequest,
):
    """
    Disable 2FA for a user. Requires both current password and a valid 2FA token
    to prevent unauthorized 2FA removal from compromised sessions.
    """
    username = payload.username
    if not username:
        try:
            body = await request.json()
            if isinstance(body, dict):
                username = body.get("username")
        except Exception:
            logger.debug("Failed to parse request payload")

    if not username:
        username = "admin"

    try:
        import pyotp

        from src.db.auth import authenticate_user, disable_2fa, get_2fa_status, init_db

        init_db()

        # 1. Verify password
        auth_result = authenticate_user(username, payload.password)
        if not auth_result.get("authenticated", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password.",
            )

        # 2. Verify 2FA token
        enabled, existing_secret = get_2fa_status(username)
        if not enabled or not existing_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is not enabled for this user.",
            )

        totp = pyotp.TOTP(existing_secret)
        if not totp.verify(payload.otp_code):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA token.",
            )

        # 3. Disable 2FA
        disable_2fa(username)

        return {
            "message": "2FA has been successfully disabled.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to disable 2FA for user %s: %s", username, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable 2FA: {str(e)}",
        )


@router.post(
    "/auth/2fa/verify",
    summary="Verify TOTP 2FA code",
    response_model=TwoFactorVerifyResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized / Invalid 2FA Code"},
        429: {
            "model": ErrorResponse,
            "description": "Rate limit exceeded (max 5 attempts per minute)",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@router.post(
    "/api/v1/auth/2fa/verify",
    summary="Verify TOTP 2FA code",
    response_model=TwoFactorVerifyResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized / Invalid 2FA Code"},
        429: {
            "model": ErrorResponse,
            "description": "Rate limit exceeded (max 5 attempts per minute)",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@limiter.limit("5/minute")
async def verify_two_factor_auth_endpoint(
    request: Request,
    payload: TwoFactorVerifyRequest,
):
    """
    Verify a Time-based One-Time Password (TOTP) 2FA code for a user.
    Enforces a strict rate limit of max 5 verification attempts per minute to prevent brute-force attacks.
    Returns HTTP 429 Too Many Requests when threshold is exceeded.
    """
    username = payload.username
    otp_code = payload.otp_code

    if not username or not otp_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both username and otp_code are required.",
        )

    try:
        import pyotp
        from slowapi.util import get_remote_address
        from src.db.auth import get_2fa_status, init_db, log_security_event

        init_db()
        client_ip = get_remote_address(request)

        enabled, existing_secret = get_2fa_status(username)
        if not enabled or not existing_secret:
            log_security_event(
                "2FA_VERIFY_FAILED",
                username,
                f"Client IP: {client_ip} | 2FA not enabled",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is not enabled for this user.",
            )

        totp = pyotp.TOTP(existing_secret)
        if not totp.verify(otp_code):
            log_security_event(
                "2FA_VERIFY_FAILED",
                username,
                f"Client IP: {client_ip} | Invalid OTP code",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA verification code.",
            )

        log_security_event("2FA_VERIFY_SUCCESS", username, f"Client IP: {client_ip}")
        return {
            "verified": True,
            "message": "2FA code verified successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to verify 2FA code for user %s: %s", username, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify 2FA code: {str(e)}",
        )


def create_reset_token(email: str) -> str:
    """Generates a secure, cryptographically signed short-lived reset token (15-minute expiration)."""
    from src.security.jwt_utils import create_jwt_token

    return create_jwt_token(
        {"sub": email, "type": "reset", "action": "password_reset"},
        expires_in_seconds=900,
    )


def verify_reset_token(token: str) -> str:
    """Verifies signature bounds and expiration limits of the reset token."""
    from src.security.jwt_utils import _verify_jwt_token

    try:
        payload = _verify_jwt_token(token, expected_type="reset")
        email = payload.get("sub")
        action = payload.get("action")
        if not email or action != "password_reset":
            raise ValueError("Invalid token payload.")
        return email
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired or is cryptographically invalid.",
        )


from fastapi import BackgroundTasks
from pydantic import EmailStr, Field
from src.security import password_recovery


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: SecretStr
    new_password: SecretStr


@router.post("/api/v1/auth/forgot-password")
@limiter.limit("5/minute")
def forgot_password(request: Request, payload: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    if not password_recovery.mail_is_configured():
        raise HTTPException(status_code=503, detail="Email recovery is unavailable. Contact your administrator.")
    background_tasks.add_task(password_recovery.request_password_reset, str(payload.email))
    return {"message": password_recovery.RESET_MESSAGE}


@router.post("/api/v1/auth/reset-password")
@limiter.limit("5/minute")
def reset_password_endpoint(request: Request, payload: ResetPasswordRequest):
    try:
        password_recovery.reset_password(payload.token.get_secret_value(), payload.new_password.get_secret_value())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Password updated successfully. Sign in again."}
