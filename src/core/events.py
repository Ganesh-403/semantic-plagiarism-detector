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

"""
src/core/events.py
-------------------
Canonical, schema-versioned event model for outbound webhook notifications.

This module gives integrators a single, stable contract for every event the
application emits over webhooks (plagiarism alerts, incident review updates,
document lifecycle events, scan failures, and system health warnings) instead
of the previously ad-hoc, per-caller payload shapes.

Versioning policy (see docs/WEBHOOK_EVENTS.md for the consumer-facing copy):
    - Additive, backward-compatible changes (new optional payload fields, new
      event types) bump the MINOR version.
    - Breaking changes (removing/renaming a field, changing a field's type or
      meaning) bump the MAJOR version and warrant a new event catalog entry.

Key pieces:
    - ``WebhookEventType``: enum of every event type the app can emit.
    - ``WebhookEvent``: immutable envelope wrapping a type-specific payload.
    - ``create_event``: convenience constructor that fills in ``event_id`` and
      ``occurred_at`` for the caller.
    - ``serialize_event`` / ``deserialize_event``: deterministic JSON codec.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from src.errors import (
    EVENT_MALFORMED_PAYLOAD,
    EVENT_MISSING_FIELD,
    EVENT_UNKNOWN_TYPE,
    EventSchemaError,
)

# Bump per the versioning policy documented in docs/WEBHOOK_EVENTS.md.
CURRENT_SCHEMA_VERSION = "1.0"

# Fixed, canonical order of keys in the serialized envelope. Kept separate
# from the dataclass field order so re-ordering the dataclass in future never
# silently changes the wire format.
_ENVELOPE_KEY_ORDER = (
    "schema_version",
    "event_type",
    "event_id",
    "occurred_at",
    "workspace_id",
    "payload",
)


class WebhookEventType(str, Enum):
    """Every event type the application can emit over webhooks.

    Inherits from ``str`` so members serialize as plain strings (e.g.
    ``json.dumps({"event_type": WebhookEventType.PLAGIARISM_DETECTED})``
    naturally yields ``"plagiarism_detected"``) and compare equal to their
    string value.
    """

    PLAGIARISM_DETECTED = "plagiarism_detected"
    INCIDENT_REVIEWED = "incident_reviewed"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    SCAN_FAILED = "scan_failed"
    SYSTEM_HEALTH_WARNING = "system_health_warning"


_VALID_EVENT_TYPES = {member.value for member in WebhookEventType}


def _utc_now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sort_for_wire(value: Any) -> Any:
    """Recursively normalise a JSON-able value into a deterministic form.

    Dict keys are sorted so that ``payload`` (built and passed in by the
    caller in whatever key order happened to be convenient) always serializes
    identically regardless of insertion order. List order is preserved, since
    lists are semantically ordered.
    """
    if isinstance(value, Mapping):
        return {key: _sort_for_wire(value[key]) for key in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_sort_for_wire(item) for item in value]
    return value


@dataclass(frozen=True)
class WebhookEvent:
    """A single, schema-versioned webhook event envelope.

    Attributes:
        schema_version: Semantic version of this envelope's shape (see
            ``CURRENT_SCHEMA_VERSION``). Consumers should branch on this
            rather than guessing at payload shape.
        event_type: One of ``WebhookEventType``.
        event_id: Unique identifier (UUID4 string) for this specific event
            occurrence, useful for consumer-side idempotency/de-duplication.
        occurred_at: ISO-8601 UTC timestamp of when the event occurred.
        workspace_id: Identifier of the workspace/tenant the event belongs
            to. Present on every event so multi-tenant consumers can route
            without inspecting the payload.
        payload: Event-type-specific data. See docs/WEBHOOK_EVENTS.md for
            the documented shape per event type.
    """

    schema_version: str
    event_type: WebhookEventType
    event_id: str
    occurred_at: str
    workspace_id: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the envelope as a plain dict with a fixed, canonical key order."""
        raw = {
            "schema_version": self.schema_version,
            "event_type": (
                self.event_type.value
                if isinstance(self.event_type, WebhookEventType)
                else str(self.event_type)
            ),
            "event_id": self.event_id,
            "occurred_at": self.occurred_at,
            "workspace_id": self.workspace_id,
            "payload": self.payload,
        }
        return {key: raw[key] for key in _ENVELOPE_KEY_ORDER}


def create_event(
    event_type: WebhookEventType,
    payload: dict | None = None,
    *,
    workspace_id: str = "default",
    event_id: str | None = None,
    occurred_at: str | None = None,
    schema_version: str = CURRENT_SCHEMA_VERSION,
) -> WebhookEvent:
    """Build a new ``WebhookEvent``, filling in identity/timestamp fields.

    Args:
        event_type: The type of event being emitted.
        payload: Event-type-specific data. Defaults to an empty dict.
        workspace_id: Workspace/tenant identifier. Defaults to ``"default"``
            for single-tenant deployments.
        event_id: Explicit event id (mainly for tests). A UUID4 is generated
            when omitted.
        occurred_at: Explicit ISO-8601 timestamp (mainly for tests). The
            current UTC time is used when omitted.
        schema_version: Envelope schema version. Defaults to the current
            version; only override this for compatibility testing.

    Returns:
        A fully populated, immutable ``WebhookEvent``.
    """
    if not isinstance(event_type, WebhookEventType):
        event_type = WebhookEventType(event_type)

    return WebhookEvent(
        schema_version=schema_version,
        event_type=event_type,
        event_id=event_id or str(uuid.uuid4()),
        occurred_at=occurred_at or _utc_now_iso(),
        workspace_id=workspace_id,
        payload=payload or {},
    )


def serialize_event(event: WebhookEvent) -> str:
    """Serialize a ``WebhookEvent`` to a deterministic JSON string.

    The envelope's top-level keys are always emitted in the fixed order
    ``schema_version, event_type, event_id, occurred_at, workspace_id,
    payload``, and the ``payload`` sub-object's keys (and any nested dicts
    within it) are sorted alphabetically. Given the same input event, this
    function always returns byte-identical output.
    """
    envelope = event.to_dict()
    envelope["payload"] = _sort_for_wire(envelope["payload"])
    return json.dumps(envelope, sort_keys=False, separators=(",", ":"))


def deserialize_event(payload: str) -> WebhookEvent:
    """Parse and validate a JSON string into a ``WebhookEvent``.

    Raises:
        EventSchemaError: If the payload is not valid JSON, is missing a
            required field, or has an ``event_type`` that is not one of the
            known ``WebhookEventType`` values.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise EventSchemaError(EVENT_MALFORMED_PAYLOAD.format(error=exc)) from exc

    if not isinstance(data, dict):
        raise EventSchemaError(
            EVENT_MALFORMED_PAYLOAD.format(
                error="top-level JSON value must be an object"
            )
        )

    for required_field in _ENVELOPE_KEY_ORDER:
        if required_field not in data:
            raise EventSchemaError(EVENT_MISSING_FIELD.format(field=required_field))

    raw_event_type = data["event_type"]
    if raw_event_type not in _VALID_EVENT_TYPES:
        raise EventSchemaError(
            EVENT_UNKNOWN_TYPE.format(
                event_type=raw_event_type,
                valid_types=", ".join(sorted(_VALID_EVENT_TYPES)),
            )
        )

    event_payload = data["payload"]
    if not isinstance(event_payload, dict):
        raise EventSchemaError(
            EVENT_MALFORMED_PAYLOAD.format(error="'payload' must be a JSON object")
        )

    return WebhookEvent(
        schema_version=str(data["schema_version"]),
        event_type=WebhookEventType(raw_event_type),
        event_id=str(data["event_id"]),
        occurred_at=str(data["occurred_at"]),
        workspace_id=str(data["workspace_id"]),
        payload=event_payload,
    )
