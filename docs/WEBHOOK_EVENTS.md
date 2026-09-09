# Webhook Event Catalog

This document is the canonical, consumer-facing reference for every event the
Semantic Plagiarism Detector emits over its outbound webhook. It replaces
guessing at payload shape by hand-parsing `text`/`content` strings: every
event is a schema-versioned JSON object with a stable set of top-level
fields, defined in [`src/core/events.py`](../src/core/events.py).

> **Relationship to `docs/WEBHOOKS.md`:** that document covers webhook
> _transport_ concerns — configuration, HMAC signing, SSRF protection, retry
> behavior. This document covers the _event schema and catalog_ — what gets
> sent, and how to parse it reliably. Read both if you're building an
> integration.

## Envelope Shape

Every event, regardless of type, is delivered as a JSON object with exactly
these six top-level keys, always in this order:

```json
{
  "schema_version": "1.0",
  "event_type": "plagiarism_detected",
  "event_id": "6b1f7f2e-6c8b-4f7b-9a2e-1a2b3c4d5e6f",
  "occurred_at": "2026-01-15T09:30:00+00:00",
  "workspace_id": "default",
  "payload": {}
}
```

| Field            | Type   | Description                                                                                                     |
| ---------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| `schema_version` | string | Semantic version of this envelope's shape. See [Versioning Policy](#versioning-policy).                         |
| `event_type`     | string | One of the [event types](#event-types) below. Unknown values must be rejected by strict consumers.              |
| `event_id`       | string | UUID4 identifying this specific event occurrence. Use it to de-duplicate on the consumer side.                  |
| `occurred_at`    | string | ISO-8601 UTC timestamp of when the event occurred.                                                              |
| `workspace_id`   | string | Identifier of the workspace/tenant the event belongs to. Defaults to `"default"` for single-tenant deployments. |
| `payload`        | object | Event-type-specific data. Shape documented per event type below.                                                |

For backward compatibility with Slack- and Discord-style incoming webhooks
(which read a bare `text` or `content` field and ignore everything else),
the application also includes `text` and `content` keys alongside the
envelope on every delivered request. These two fields carry a short,
human-readable summary of the event and are **not** part of the versioned
schema — don't build parsing logic against them; use `payload` instead.

### Deterministic Serialization

`src.core.events.serialize_event()` produces byte-identical JSON for the
same input event: top-level keys are always emitted in the order shown
above, and `payload` keys (including nested objects) are sorted
alphabetically. This matters if you're verifying an HMAC signature computed
over the raw body (see `docs/WEBHOOKS.md`) — the signed bytes are exactly
what `serialize_event()` produces (modulo the two Slack/Discord compatibility
fields appended for transport, which are not covered by this guarantee).

### Rejecting Unknown Event Types

`src.core.events.deserialize_event()` raises `EventSchemaError` if
`event_type` is not one of the known values, or if any required envelope
field is missing. Integrators building their own parsers should apply the
same rule: **fail closed** on an unrecognized `event_type` rather than
guessing at its shape. New event types are only ever added as MINOR schema
bumps (see below), so a strict consumer that rejects unknown types will
never silently mis-handle a real event — it will simply need updating once
it wants to support the new type.

---

## Event Types

### `plagiarism_detected`

Emitted when a pair of documents is flagged as a semantic plagiarism match.

```json
{
  "schema_version": "1.0",
  "event_type": "plagiarism_detected",
  "event_id": "6b1f7f2e-6c8b-4f7b-9a2e-1a2b3c4d5e6f",
  "occurred_at": "2026-01-15T09:30:00+00:00",
  "workspace_id": "default",
  "payload": {
    "document_a": "student_essay.pdf",
    "document_b": "wikipedia_source.pdf",
    "similarity_score": 0.925,
    "review_url": "https://your-instance.example.com"
  }
}
```

| Payload field      | Type   | Description                                             |
| ------------------ | ------ | ------------------------------------------------------- |
| `document_a`       | string | Filename of the first document in the matched pair.     |
| `document_b`       | string | Filename of the second document in the matched pair.    |
| `similarity_score` | number | Cosine similarity between `0.0` and `1.0`.              |
| `review_url`       | string | Link to the dashboard instance for reviewing the match. |

Emitted by `src.core.webhook.dispatch_plagiarism_alert()` /
`send_plagiarism_alert()`.

### `incident_reviewed`

Emitted when a teacher/admin updates the review status of a logged
plagiarism incident (e.g. marking it `Resolved`).

```json
{
  "schema_version": "1.0",
  "event_type": "incident_reviewed",
  "event_id": "1c2d3e4f-5678-4abc-9def-0123456789ab",
  "occurred_at": "2026-01-15T09:45:12+00:00",
  "workspace_id": "default",
  "payload": {
    "incident_id": "INC-42",
    "review_status": "Resolved",
    "reviewed_by": "teacher_jane"
  }
}
```

| Payload field   | Type   | Description                                         |
| --------------- | ------ | --------------------------------------------------- |
| `incident_id`   | string | Identifier of the incident that was reviewed.       |
| `review_status` | string | New status — currently `"Pending"` or `"Resolved"`. |
| `reviewed_by`   | string | _(optional)_ Username of the reviewer, when known.  |

Emitted by `src.core.webhook.dispatch_incident_reviewed_event()` /
`send_incident_reviewed_alert()`, wired into the incident review panel in
`app/components/incident_export.py`.

### `document_uploaded`

Reserved for a document successfully entering the corpus (upload, parse, and
indexing complete). Payload will include the document filename, hash, and
upload timestamp. _Not yet wired to a dispatch call site — defined here so
the contract is stable ahead of that integration._

### `document_deleted`

Reserved for a document being removed from the corpus. Payload will include
the document filename and hash. _Not yet wired to a dispatch call site._

### `scan_failed`

Reserved for a plagiarism scan that failed to complete (e.g. parsing error,
pipeline exception). Payload will include the failing document(s) and an
error summary. _Not yet wired to a dispatch call site._

### `system_health_warning`

Reserved for operational warnings (e.g. storage nearing capacity, repeated
webhook delivery failures). Payload will include a warning code and message.
_Not yet wired to a dispatch call site._

---

## Versioning Policy

`schema_version` follows a `MAJOR.MINOR` scheme (currently `"1.0"`).

- **Additive, backward-compatible changes bump MINOR.** Adding a new
  optional field to an existing event's `payload`, or adding an entirely new
  `event_type`, is a MINOR bump. Existing consumers that ignore fields they
  don't recognize continue to work unchanged.
- **Breaking changes bump MAJOR.** Removing or renaming a payload field,
  changing a field's type or meaning, or changing the envelope's top-level
  shape is a MAJOR bump. A MAJOR bump warrants updating this catalog with
  the new shape and a migration note.
- The envelope's six top-level fields (`schema_version` through `payload`)
  are themselves part of the MAJOR-versioned contract and won't be removed
  or reordered within a major version.

Consumers should branch on `schema_version` rather than assuming the current
shape, and should tolerate unknown fields being added to `payload` in future
MINOR versions.

## Building a Consumer

1. Verify the HMAC signature per `docs/WEBHOOKS.md` before trusting the body.
2. Parse the JSON body and check `schema_version` is a version you support.
3. Switch on `event_type`; reject (log and drop, don't guess) anything
   outside the [event types](#event-types) list above.
4. Read the type-specific fields out of `payload`.
5. Use `event_id` to de-duplicate if your delivery path might retry.

See [`src/core/events.py`](../src/core/events.py) for the reference
implementation (`WebhookEventType`, `WebhookEvent`, `serialize_event`,
`deserialize_event`) if you're building a Python consumer and want to reuse
the same validation logic.
