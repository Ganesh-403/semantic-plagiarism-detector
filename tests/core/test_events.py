"""
tests/core/test_events.py
--------------------------
Unit tests for the canonical, schema-versioned webhook event model:
deterministic serialization, unknown-event-type rejection, and round-trip
fidelity.
"""

import json

import pytest

from src.core.events import (
    CURRENT_SCHEMA_VERSION,
    WebhookEvent,
    WebhookEventType,
    create_event,
    deserialize_event,
    serialize_event,
)
from src.errors import EventSchemaError

FIXED_EVENT_ID = "11111111-1111-4111-8111-111111111111"
FIXED_TIMESTAMP = "2026-01-15T09:30:00+00:00"


def make_event(**overrides) -> WebhookEvent:
    kwargs = dict(
        event_type=WebhookEventType.PLAGIARISM_DETECTED,
        payload={
            "document_a": "essay_a.pdf",
            "document_b": "essay_b.pdf",
            "similarity_score": 0.93,
        },
        workspace_id="workspace-1",
        event_id=FIXED_EVENT_ID,
        occurred_at=FIXED_TIMESTAMP,
    )
    kwargs.update(overrides)
    return create_event(**kwargs)


class TestWebhookEventType:
    def test_all_required_event_types_present(self):
        expected = {
            "plagiarism_detected",
            "incident_reviewed",
            "document_uploaded",
            "document_deleted",
            "scan_failed",
            "system_health_warning",
        }
        actual = {member.value for member in WebhookEventType}
        assert actual == expected

    def test_event_type_is_string_valued(self):
        # WebhookEventType inherits from str so it serializes/compares cleanly.
        assert WebhookEventType.PLAGIARISM_DETECTED == "plagiarism_detected"
        assert isinstance(WebhookEventType.PLAGIARISM_DETECTED.value, str)


class TestCreateEvent:
    def test_defaults_are_filled_in(self):
        event = create_event(WebhookEventType.DOCUMENT_UPLOADED)
        assert event.schema_version == CURRENT_SCHEMA_VERSION
        assert event.event_type == WebhookEventType.DOCUMENT_UPLOADED
        assert event.workspace_id == "default"
        assert event.payload == {}
        # event_id must be a valid UUID string
        import uuid

        uuid.UUID(event.event_id)
        # occurred_at must be ISO-8601 parseable
        from datetime import datetime

        datetime.fromisoformat(event.occurred_at)

    def test_accepts_plain_string_event_type(self):
        event = create_event("scan_failed")
        assert event.event_type == WebhookEventType.SCAN_FAILED

    def test_invalid_string_event_type_raises(self):
        with pytest.raises(ValueError):
            create_event("not_a_real_event_type")

    def test_event_is_immutable(self):
        event = make_event()
        with pytest.raises(Exception):
            event.workspace_id = "other-workspace"  # type: ignore[misc]


class TestSerializeEventDeterminism:
    def test_serialize_is_byte_deterministic(self):
        """Same inputs must produce byte-identical JSON, repeatedly."""
        event = make_event()
        first = serialize_event(event)
        second = serialize_event(make_event())
        assert first == second

    def test_payload_key_order_does_not_affect_output(self):
        """Callers may build the payload dict in any key order; output must match."""
        event_a = make_event(
            payload={
                "similarity_score": 0.93,
                "document_b": "essay_b.pdf",
                "document_a": "essay_a.pdf",
            }
        )
        event_b = make_event(
            payload={
                "document_a": "essay_a.pdf",
                "document_b": "essay_b.pdf",
                "similarity_score": 0.93,
            }
        )
        assert serialize_event(event_a) == serialize_event(event_b)

    def test_exact_string_equality(self):
        """Pin the exact serialized form so accidental format drift is caught."""
        event = create_event(
            WebhookEventType.PLAGIARISM_DETECTED,
            payload={"b": 2, "a": 1},
            workspace_id="ws-1",
            event_id=FIXED_EVENT_ID,
            occurred_at=FIXED_TIMESTAMP,
        )
        expected = (
            '{"schema_version":"1.0","event_type":"plagiarism_detected",'
            f'"event_id":"{FIXED_EVENT_ID}","occurred_at":"{FIXED_TIMESTAMP}",'
            '"workspace_id":"ws-1","payload":{"a":1,"b":2}}'
        )
        assert serialize_event(event) == expected

    def test_top_level_key_order_is_fixed(self):
        event = make_event()
        raw = json.loads(serialize_event(event))
        assert list(raw.keys()) == [
            "schema_version",
            "event_type",
            "event_id",
            "occurred_at",
            "workspace_id",
            "payload",
        ]

    def test_nested_payload_dicts_are_also_sorted(self):
        event_a = make_event(payload={"outer": {"z": 1, "a": 2}})
        event_b = make_event(payload={"outer": {"a": 2, "z": 1}})
        assert serialize_event(event_a) == serialize_event(event_b)

    def test_output_is_valid_json(self):
        event = make_event()
        # Should not raise.
        json.loads(serialize_event(event))


class TestDeserializeEvent:
    def test_round_trip_fidelity(self):
        original = make_event()
        serialized = serialize_event(original)
        restored = deserialize_event(serialized)

        assert restored.schema_version == original.schema_version
        assert restored.event_type == original.event_type
        assert restored.event_id == original.event_id
        assert restored.occurred_at == original.occurred_at
        assert restored.workspace_id == original.workspace_id
        assert restored.payload == original.payload
        # Re-serializing the round-tripped event must be byte-identical.
        assert serialize_event(restored) == serialized

    def test_deserialize_returns_enum_member(self):
        original = make_event(event_type=WebhookEventType.INCIDENT_REVIEWED)
        restored = deserialize_event(serialize_event(original))
        assert restored.event_type is WebhookEventType.INCIDENT_REVIEWED

    def test_unknown_event_type_is_rejected(self):
        forged = serialize_event(make_event()).replace(
            '"plagiarism_detected"', '"totally_made_up_event"'
        )
        with pytest.raises(EventSchemaError, match="Unknown webhook event type"):
            deserialize_event(forged)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "schema_version",
            "event_type",
            "event_id",
            "occurred_at",
            "workspace_id",
            "payload",
        ],
    )
    def test_missing_required_field_is_rejected(self, missing_field):
        data = json.loads(serialize_event(make_event()))
        del data[missing_field]
        with pytest.raises(EventSchemaError, match=missing_field):
            deserialize_event(json.dumps(data))

    def test_malformed_json_is_rejected(self):
        with pytest.raises(EventSchemaError):
            deserialize_event("{not valid json")

    def test_non_object_json_is_rejected(self):
        with pytest.raises(EventSchemaError):
            deserialize_event("[1, 2, 3]")

    def test_non_dict_payload_is_rejected(self):
        data = json.loads(serialize_event(make_event()))
        data["payload"] = "not-a-dict"
        with pytest.raises(EventSchemaError):
            deserialize_event(json.dumps(data))


class TestWebhookEventToDict:
    def test_to_dict_key_order(self):
        event = make_event()
        assert list(event.to_dict().keys()) == [
            "schema_version",
            "event_type",
            "event_id",
            "occurred_at",
            "workspace_id",
            "payload",
        ]

    def test_to_dict_event_type_is_plain_string(self):
        event = make_event()
        assert event.to_dict()["event_type"] == "plagiarism_detected"
        assert isinstance(event.to_dict()["event_type"], str)
