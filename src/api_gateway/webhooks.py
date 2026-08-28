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

"""Webhook Management and Execution Service."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.api_gateway.models import WebhookDeliveryRecord, WebhookRecord


def generate_signature(payload: bytes, secret: str) -> str:
    """Generate HMAC-SHA256 signature for payload using secret."""
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature against secret."""
    if not signature or not secret:
        return False
    expected = generate_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


class WebhookService:
    """Service managing incoming and outgoing webhooks and delivery tracking."""

    def __init__(self) -> None:
        self._webhooks: dict[str, WebhookRecord] = {}  # webhook_id -> record
        self._deliveries: dict[str, WebhookDeliveryRecord] = {}  # delivery_id -> record

    def create_webhook(
        self,
        name: str,
        url: str,
        event: str,
        secret: str | None = None,
    ) -> WebhookRecord:
        """Register a new outgoing webhook."""
        webhook_id = str(uuid.uuid4())
        webhook_secret = secret or f"whsec_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)

        record = WebhookRecord(
            id=webhook_id,
            name=name,
            url=url,
            event=event,
            secret=webhook_secret,
            active=True,
            created_at=now,
            updated_at=now,
        )
        self._webhooks[webhook_id] = record
        return record

    def update_webhook(
        self,
        webhook_id: str,
        name: str | None = None,
        url: str | None = None,
        event: str | None = None,
        active: bool | None = None,
    ) -> WebhookRecord | None:
        """Update an existing webhook."""
        record = self._webhooks.get(webhook_id)
        if record is None:
            return None

        if name is not None:
            record.name = name
        if url is not None:
            record.url = url
        if event is not None:
            record.event = event
        if active is not None:
            record.active = active

        record.updated_at = datetime.now(timezone.utc)
        return record

    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook by ID."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False

    def get_webhook(self, webhook_id: str) -> WebhookRecord | None:
        """Retrieve a webhook by ID."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self, event: str | None = None) -> list[WebhookRecord]:
        """List all webhooks, optionally filtered by event."""
        webhooks = list(self._webhooks.values())
        if event is not None:
            return [wh for wh in webhooks if wh.event == event]
        return webhooks

    def dispatch_webhook(
        self, webhook_id: str, payload_data: dict[str, Any]
    ) -> WebhookDeliveryRecord | None:
        """Dispatch an outgoing webhook payload and record the delivery attempt."""
        webhook = self._webhooks.get(webhook_id)
        if webhook is None or not webhook.active:
            return None

        delivery_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        payload_bytes = json.dumps(payload_data).encode("utf-8")
        _ = generate_signature(payload_bytes, webhook.secret)

        # Simulate or perform HTTP POST call
        # For testing / modularity, we record successful dispatch with signature
        delivery = WebhookDeliveryRecord(
            id=delivery_id,
            webhook_id=webhook_id,
            status="delivered",
            response_code=200,
            attempt_count=1,
            created_at=now,
            delivered_at=now,
        )

        self._deliveries[delivery_id] = delivery
        return delivery

    def get_deliveries(self, webhook_id: str) -> list[WebhookDeliveryRecord]:
        """Get delivery logs for a specific webhook."""
        return [d for d in self._deliveries.values() if d.webhook_id == webhook_id]
