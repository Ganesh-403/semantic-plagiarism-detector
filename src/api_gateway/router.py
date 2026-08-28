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

"""FastAPI Router for API Gateway management and endpoint execution."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from src.api_gateway.gateway import (
    APIGateway,
    InvalidAPIKeyException,
    RateLimitExceededException,
    RouteNotFoundException,
)
from src.api_gateway.schemas import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadataResponse,
    IntegrationCreateRequest,
    IntegrationResponse,
    WebhookCreateRequest,
    WebhookResponse,
)
from src.api_gateway.webhooks import verify_signature

# Global API Gateway instance
gateway = APIGateway()

router = APIRouter(prefix="/api/v1/gateway", tags=["API Gateway"])


# ── API Key Management Endpoints ──────────────────────────────────────────────


@router.post(
    "/api-keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API Key",
)
def create_api_key(req: APIKeyCreateRequest) -> APIKeyCreateResponse:
    """Generate a new secure API Key. The raw key is returned ONLY once."""
    record, raw_key = gateway.api_keys.create_key(
        name=req.name,
        expires_in_days=req.expires_in_days,
        rate_limit=req.rate_limit,
    )
    return APIKeyCreateResponse(
        id=record.id,
        name=record.name,
        raw_key=raw_key,
        created_at=record.created_at,
        expires_at=record.expires_at,
        rate_limit=record.rate_limit,
    )


@router.get(
    "/api-keys",
    response_model=list[APIKeyMetadataResponse],
    summary="List API Key Metadata",
)
def list_api_keys() -> list[APIKeyMetadataResponse]:
    """List metadata for all API keys (without raw keys)."""
    keys = gateway.api_keys.list_keys()
    return [
        APIKeyMetadataResponse(
            id=k.id,
            name=k.name,
            created_at=k.created_at,
            expires_at=k.expires_at,
            revoked_at=k.revoked_at,
            last_used_at=k.last_used_at,
            is_active=k.is_active,
            rate_limit=k.rate_limit,
        )
        for k in keys
    ]


@router.post(
    "/api-keys/{key_id}/rotate",
    response_model=APIKeyCreateResponse,
    summary="Rotate an existing API Key",
)
def rotate_api_key(
    key_id: str, expires_in_days: int | None = None
) -> APIKeyCreateResponse:
    """Revoke an existing API Key and generate a new key."""
    result = gateway.api_keys.rotate_key(key_id, expires_in_days=expires_in_days)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API Key '{key_id}' not found or already revoked.",
        )
    new_record, new_raw_key = result
    return APIKeyCreateResponse(
        id=new_record.id,
        name=new_record.name,
        raw_key=new_raw_key,
        created_at=new_record.created_at,
        expires_at=new_record.expires_at,
        rate_limit=new_record.rate_limit,
    )


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API Key",
)
def revoke_api_key(key_id: str) -> None:
    """Revoke an active API Key."""
    success = gateway.api_keys.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API Key '{key_id}' not found or already revoked.",
        )


# ── Webhook Management Endpoints ──────────────────────────────────────────────


@router.post(
    "/webhooks/outgoing",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Outgoing Webhook",
)
def register_webhook(req: WebhookCreateRequest) -> WebhookResponse:
    """Register a new outgoing webhook endpoint."""
    record = gateway.webhooks.create_webhook(
        name=req.name,
        url=req.url,
        event=req.event,
        secret=req.secret,
    )
    return WebhookResponse(
        id=record.id,
        name=record.name,
        url=record.url,
        event=record.event,
        active=record.active,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get(
    "/webhooks/outgoing",
    response_model=list[WebhookResponse],
    summary="List Registered Webhooks",
)
def list_webhooks(event: str | None = None) -> list[WebhookResponse]:
    """List registered webhooks, optionally filtered by event."""
    records = gateway.webhooks.list_webhooks(event=event)
    return [
        WebhookResponse(
            id=r.id,
            name=r.name,
            url=r.url,
            event=r.event,
            active=r.active,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]


@router.delete(
    "/webhooks/outgoing/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Webhook",
)
def delete_webhook(webhook_id: str) -> None:
    """Delete a registered webhook."""
    success = gateway.webhooks.delete_webhook(webhook_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook '{webhook_id}' not found.",
        )


@router.post(
    "/webhooks/incoming/{webhook_id}",
    summary="Receive Incoming Webhook Event",
)
async def receive_incoming_webhook(
    webhook_id: str,
    request: Request,
    x_webhook_signature: str | None = Header(None, alias="X-Webhook-Signature"),
) -> dict[str, Any]:
    """Process incoming webhook request after verifying HMAC signature."""
    webhook = gateway.webhooks.get_webhook(webhook_id)
    if webhook is None or not webhook.active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incoming webhook '{webhook_id}' not found or inactive.",
        )

    payload_bytes = await request.body()
    if not x_webhook_signature or not verify_signature(
        payload_bytes, x_webhook_signature, webhook.secret
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing webhook signature.",
        )

    try:
        body_json = await request.json()
    except Exception:
        body_json = {}

    return {
        "status": "accepted",
        "webhook_id": webhook_id,
        "event": webhook.event,
        "data": body_json,
    }


# ── Integration Management Endpoints ──────────────────────────────────────────


@router.post(
    "/integrations",
    response_model=IntegrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Third-Party Integration",
)
def create_integration(req: IntegrationCreateRequest) -> IntegrationResponse:
    """Register a new third-party service integration."""
    record = gateway.integrations.create_integration(
        name=req.name,
        provider=req.provider,
        credentials=req.credentials,
        enabled=req.enabled,
    )
    return IntegrationResponse(
        id=record.id,
        name=record.name,
        provider=record.provider,
        enabled=record.enabled,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get(
    "/integrations",
    response_model=list[IntegrationResponse],
    summary="List Third-Party Integrations",
)
def list_integrations() -> list[IntegrationResponse]:
    """List integrations with credentials masked."""
    records = gateway.integrations.list_integrations()
    return [
        IntegrationResponse(
            id=r.id,
            name=r.name,
            provider=r.provider,
            enabled=r.enabled,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]


@router.post(
    "/integrations/{integration_id}/execute",
    summary="Execute Third-Party Service Request",
)
async def execute_integration(
    integration_id: str,
    method: str = "POST",
    path: str = "/",
    payload: dict[str, Any] | None = None,
) -> Any:
    """Execute API request through third-party connector."""
    try:
        return await gateway.integrations.execute_request(
            integration_id=integration_id,
            method=method,
            path=path,
            json=payload or {},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


# ── Gateway Endpoint Dispatch ──────────────────────────────────────────────────


@router.api_route(
    "/exposed/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    summary="Gateway Entry Point for Exposed APIs",
)
async def gateway_dispatch(
    path: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> Any:
    """Single entry point routing requests to registered internal endpoints."""
    method = request.method
    client_ip = request.client.host if request.client else "127.0.0.1"

    try:
        kwargs: dict[str, Any] = {}
        if method in ("POST", "PUT", "PATCH"):
            try:
                kwargs = await request.json()
            except Exception:
                kwargs = {}

        result = await gateway.dispatch(
            method=method,
            path=path,
            api_key=x_api_key,
            client_ip=client_ip,
            **kwargs,
        )
        return result
    except RouteNotFoundException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except InvalidAPIKeyException as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    except RateLimitExceededException as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from exc
