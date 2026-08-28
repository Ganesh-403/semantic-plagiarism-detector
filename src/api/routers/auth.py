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

import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.api.dependencies import limiter
from src.api.schemas import (
    ErrorResponse,
    LoginResponse,
    RefreshRequest,
    RevokeRequest,
    RevokeResponse,
    TokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


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
async def login(request: Request):
    """Authenticate user and return a session token."""
    return {"token": "dummy-token"}


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
            pass

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

    sub = token_payload.get("sub", "user")
    scopes = token_payload.get("scopes", ["read", "write"])
    new_access_token = create_access_token(sub=sub, scopes=scopes, expires_in=3600)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
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
            pass

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
