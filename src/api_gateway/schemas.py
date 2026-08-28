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

"""Pydantic schemas for API Gateway endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class APIKeyCreateRequest(BaseModel):
    """Schema for creating a new API key."""

    name: str = Field(..., description="Descriptive name for the API key")
    expires_in_days: int | None = Field(
        default=None, description="Optional expiry in days"
    )
    rate_limit: int = Field(
        default=100, description="Requests per minute limit for this key"
    )


class APIKeyCreateResponse(BaseModel):
    """Response returning the raw API key (shown only once)."""

    id: str
    name: str
    raw_key: str
    created_at: datetime
    expires_at: datetime | None = None
    rate_limit: int


class APIKeyMetadataResponse(BaseModel):
    """Metadata response for API key listing."""

    id: str
    name: str
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool
    rate_limit: int


class WebhookCreateRequest(BaseModel):
    """Schema for registering a webhook."""

    name: str
    url: str
    event: str
    secret: str | None = None


class WebhookResponse(BaseModel):
    """Response schema for webhook info."""

    id: str
    name: str
    url: str
    event: str
    active: bool
    created_at: datetime
    updated_at: datetime


class WebhookDeliveryResponse(BaseModel):
    """Response schema for webhook delivery history."""

    id: str
    webhook_id: str
    status: str
    response_code: int | None = None
    attempt_count: int
    created_at: datetime
    delivered_at: datetime | None = None
    error_message: str | None = None


class IntegrationCreateRequest(BaseModel):
    """Schema for adding an integration."""

    name: str
    provider: str
    credentials: dict[str, Any]
    enabled: bool = True


class IntegrationResponse(BaseModel):
    """Response schema for integration (credentials masked)."""

    id: str
    name: str
    provider: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class EndpointRegisterRequest(BaseModel):
    """Schema for registering internal functions as API endpoints."""

    method: str
    path: str
    name: str | None = None
    require_api_key: bool = True
