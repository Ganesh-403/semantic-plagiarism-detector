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

"""Unit tests for Webhook Management and Execution."""

import json

import pytest

from src.api_gateway.webhooks import (
    WebhookService,
    generate_signature,
    verify_signature,
)


@pytest.fixture
def service():
    return WebhookService()


def test_create_update_delete_webhook(service):
    record = service.create_webhook(
        name="test-hook",
        url="https://example.com/hook",
        event="document.scanned",
        secret="supersecret",
    )

    assert record.name == "test-hook"
    assert record.url == "https://example.com/hook"
    assert record.event == "document.scanned"
    assert record.active is True

    # Update
    updated = service.update_webhook(record.id, active=False, name="updated-hook")
    assert updated is not None
    assert updated.active is False
    assert updated.name == "updated-hook"

    # Delete
    assert service.delete_webhook(record.id) is True
    assert service.get_webhook(record.id) is None


def test_verify_valid_and_invalid_signatures():
    secret = "whsec_test123"
    payload = json.dumps({"event": "test"}).encode("utf-8")

    sig = generate_signature(payload, secret)

    # Valid signature
    assert verify_signature(payload, sig, secret) is True

    # Invalid signature / secret
    assert verify_signature(payload, "invalid_sig", secret) is False
    assert verify_signature(payload, sig, "wrong_secret") is False


def test_dispatch_webhook(service):
    webhook = service.create_webhook(
        name="dispatch-hook",
        url="https://example.com/webhook",
        event="incident.flagged",
    )

    delivery = service.dispatch_webhook(webhook.id, {"incident_id": "inc_99"})
    assert delivery is not None
    assert delivery.status == "delivered"
    assert delivery.response_code == 200

    deliveries = service.get_deliveries(webhook.id)
    assert len(deliveries) == 1
    assert deliveries[0].id == delivery.id
