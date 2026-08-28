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
tests/api/test_error_response_schemas.py
------------------------------------------
JSON Schema validation for the API's custom exception-handler payloads.

Previously the 422 (RequestValidationError) and 500 (unhandled exception)
responses were only checked with ad-hoc string/key assertions (see
tests/api/test_validation_errors.py and
tests/api/test_app.py::test_global_exception_handler_returns_standard_payload).
That catches typos in a handful of fields but doesn't guarantee the
payload's *overall shape* -- extra/missing keys, wrong types -- actually
conforms to what the OpenAPI docs promise callers. This module pins that
shape down with real jsonschema.validate() calls against schemas that
describe the *actual* runtime payloads produced by
src.api.app.validation_exception_handler and
src.api.app.global_exception_handler.

Note on src/api/schemas.py::ErrorResponse: the OpenAPI docs metadata for
some routes references an `ErrorResponse` Pydantic model shaped as
`{"detail": str}`, but the exception handlers below never actually
produce that shape -- they return `{"error", "message", "details"}` (422)
or `{"error", "code", "message", "timestamp"}` (500). This is a
pre-existing documented-vs-actual mismatch outside this issue's scope;
the schemas here intentionally describe the real handler output rather
than the (inaccurate) documented model, since that's what clients
actually receive.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import jsonschema
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, global_exception_handler

client = TestClient(app)


# ── Schemas describing the exception handlers' actual JSON payloads ────────

VALIDATION_ERROR_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "422 Validation Error Response",
    "description": (
        "Payload produced by src.api.app.validation_exception_handler for "
        "FastAPI RequestValidationError."
    ),
    "type": "object",
    "properties": {
        "error": {"type": "boolean", "const": True},
        "message": {"type": "string", "minLength": 1},
        "details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "message": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["field", "message", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["error", "message", "details"],
    "additionalProperties": False,
}

INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "500 Internal Server Error Response",
    "description": (
        "Payload produced by src.api.app.global_exception_handler for an "
        "unhandled exception."
    ),
    "type": "object",
    "properties": {
        "error": {"type": "boolean", "const": True},
        "code": {"type": "integer", "const": 500},
        "message": {"type": "string", "minLength": 1},
        # ISO-8601 UTC timestamp, e.g. "2026-08-16T13:04:34.418294+00:00".
        # jsonschema's Draft7Validator doesn't format-check "date-time"
        # unless a FormatChecker is supplied, so pin the exact shape with
        # a pattern instead for a real, unconditional guarantee.
        "timestamp": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+\d{2}:\d{2}|Z)?$",
        },
    },
    "required": ["error", "code", "message", "timestamp"],
    "additionalProperties": False,
}


# ── 422 (RequestValidationError) ────────────────────────────────────────────


def test_422_validation_error_conforms_to_schema():
    """A real request that fails FastAPI's request validation must return
    a payload conforming exactly to VALIDATION_ERROR_RESPONSE_SCHEMA."""
    response = client.post(
        "/api/v1/scan",
        headers={
            "Authorization": "Bearer dummy-token",
            "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        },
    )

    assert response.status_code == 422
    body = response.json()

    jsonschema.validate(instance=body, schema=VALIDATION_ERROR_RESPONSE_SCHEMA)


def test_422_validation_error_schema_rejects_malformed_payload():
    """Sanity check that the schema is actually strict, not vacuously
    permissive -- a payload missing the required 'details' array must be
    rejected."""
    malformed = {"error": True, "message": "Validation failed."}

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=malformed, schema=VALIDATION_ERROR_RESPONSE_SCHEMA)


def test_422_validation_error_schema_rejects_unexpected_extra_fields():
    """additionalProperties: False must reject an undocumented extra key,
    catching accidental payload drift from the documented shape."""
    with_extra_field = {
        "error": True,
        "message": "Validation failed.",
        "details": [],
        "unexpected_field": "should not be here",
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=with_extra_field, schema=VALIDATION_ERROR_RESPONSE_SCHEMA
        )


# ── 500 (unhandled exception / global_exception_handler) ───────────────────


def test_500_internal_server_error_conforms_to_schema(monkeypatch):
    """The global exception handler's real output for an unhandled
    exception must conform exactly to INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA."""
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    mock_request = Mock()

    response = asyncio.run(global_exception_handler(mock_request, ValueError("boom")))
    body = json.loads(response.body)

    assert response.status_code == 500
    jsonschema.validate(instance=body, schema=INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA)


def test_500_internal_server_error_conforms_to_schema_in_production(monkeypatch):
    """The masked production-mode message must still conform to the same
    schema -- masking must not change the payload's shape."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    mock_request = Mock()

    response = asyncio.run(
        global_exception_handler(mock_request, ValueError("sensitive internal detail"))
    )
    body = json.loads(response.body)

    jsonschema.validate(instance=body, schema=INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA)


def test_500_internal_server_error_schema_rejects_missing_timestamp():
    """Sanity check: a payload missing 'timestamp' must be rejected,
    proving the schema actually enforces that required field."""
    malformed = {"error": True, "code": 500, "message": "boom"}

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=malformed, schema=INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA
        )


def test_500_internal_server_error_schema_rejects_wrong_code_type():
    """Sanity check: 'code' must be an integer, not a stringified number --
    guards against a future regression that JSON-serializes it as text."""
    malformed = {
        "error": True,
        "code": "500",
        "message": "boom",
        "timestamp": "2026-08-16T13:04:34.418294+00:00",
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=malformed, schema=INTERNAL_SERVER_ERROR_RESPONSE_SCHEMA
        )
