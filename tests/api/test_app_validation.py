"""
tests/api/test_app_validation.py
--------------------------------
Comprehensive unit tests for the FastAPI validation exception handler.

Verifies that malformed requests trigger the correct logging behavior
and return properly structured 422 JSON responses. Addresses Issue #2564.
"""

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from src.api.app import validation_exception_handler


class SampleModel(BaseModel):
    """Sample Pydantic model for generating validation errors."""

    name: str
    age: int
    score: float


class TestValidationExceptionLogging:
    """Test suite for logger.warning behavior (Issue #2564)."""

    @pytest.mark.asyncio
    async def test_logs_warning_on_validation_error(self, caplog):
        """Verify logger.warning is called with exc.errors() details."""
        # Create a mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/scan"

        # Generate a real Pydantic validation error
        try:
            SampleModel(name=123, age="not_an_int", score="invalid")
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        with caplog.at_level(logging.WARNING):
            await validation_exception_handler(mock_request, exc)

        # Verify the warning was logged
        assert len(caplog.records) >= 1
        warning_record = caplog.records[0]
        assert warning_record.levelno == logging.WARNING
        assert "Request validation failed" in warning_record.message
        assert "POST" in warning_record.message
        assert "/api/v1/scan" in warning_record.message

    @pytest.mark.asyncio
    async def test_logs_contains_error_details(self, caplog):
        """Verify the log message includes the actual Pydantic error details."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "PUT"
        mock_request.url.path = "/api/v1/users"

        try:
            SampleModel(name="Alice", age="twenty", score=95.5)
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        with caplog.at_level(logging.WARNING):
            await validation_exception_handler(mock_request, exc)

        # The log args should contain the errors list
        log_message = caplog.records[0].getMessage()
        assert "age" in log_message or "Input should be a valid integer" in str(
            caplog.records[0].args
        )

    @pytest.mark.asyncio
    async def test_logs_multiple_errors(self, caplog):
        """Verify all validation errors are passed to the logger."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/submit"

        # Missing all required fields
        try:
            SampleModel()
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        with caplog.at_level(logging.WARNING):
            await validation_exception_handler(mock_request, exc)

        # Should log the errors (which contains 3 missing field errors)
        assert len(caplog.records) == 1


class TestValidationExceptionResponse:
    """Test suite for the JSON response structure."""

    @pytest.mark.asyncio
    async def test_returns_422_status_code(self):
        """Verify the response has HTTP 422 Unprocessable Entity status."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        try:
            SampleModel(name="Test", age="invalid", score=1.0)
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        response = await validation_exception_handler(mock_request, exc)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_response_contains_error_flag(self):
        """Verify the JSON response includes 'error': True."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        try:
            SampleModel(name="Test", age="invalid", score=1.0)
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        response = await validation_exception_handler(mock_request, exc)

        # Parse the JSON body
        import json

        body = json.loads(response.body)
        assert body["error"] is True

    @pytest.mark.asyncio
    async def test_response_contains_hardcoded_message(self):
        """Verify the top-level message is 'Validation failed.' (Issue #2564 context)."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        try:
            SampleModel(name="Test", age="invalid", score=1.0)
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        response = await validation_exception_handler(mock_request, exc)

        import json

        body = json.loads(response.body)
        assert body["message"] == "Validation failed."

    @pytest.mark.asyncio
    async def test_response_details_structure(self):
        """Verify the 'details' array contains field, message, and type."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        try:
            SampleModel(name="Test", age="invalid", score=1.0)
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        response = await validation_exception_handler(mock_request, exc)

        import json

        body = json.loads(response.body)

        assert "details" in body
        assert isinstance(body["details"], list)
        assert len(body["details"]) > 0

        detail = body["details"][0]
        assert "field" in detail
        assert "message" in detail
        assert "type" in detail

    @pytest.mark.asyncio
    async def test_response_formats_nested_field_paths(self):
        """Verify nested Pydantic fields are joined with dots (e.g., 'user.address.city')."""

        class NestedModel(BaseModel):
            city: str

        class ParentModel(BaseModel):
            address: NestedModel

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        try:
            ParentModel(address={"city": 123})  # city should be str
        except ValidationError as e:
            exc = RequestValidationError(e.errors())

        response = await validation_exception_handler(mock_request, exc)

        import json

        body = json.loads(response.body)

        # The field should be "address.city"
        assert body["details"][0]["field"] == "address.city"


class TestValidationExceptionEdgeCases:
    """Test suite for edge cases and malformed error objects."""

    @pytest.mark.asyncio
    async def test_handles_empty_errors_list(self):
        """Verify handler doesn't crash if exc.errors() returns empty list."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"

        # Mock an exception with no errors
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = []

        response = await validation_exception_handler(mock_request, exc)

        import json

        body = json.loads(response.body)
        assert body["details"] == []
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_handles_missing_loc_in_error(self):
        """Verify handler gracefully handles errors missing the 'loc' tuple."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/test"

        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = [
            {"msg": "Custom error", "type": "value_error"}  # Missing 'loc'
        ]

        # This should not raise a KeyError
        response = await validation_exception_handler(mock_request, exc)

        import json

        body = json.loads(response.body)
        # The field should be empty string since loc is missing
        assert body["details"][0]["field"] == ""
