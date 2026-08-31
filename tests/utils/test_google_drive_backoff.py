from unittest.mock import MagicMock

import pytest

from src.utils.google_drive_client import GoogleDriveResilientClient


class MockGoogleAPIError(Exception):
    """Simulates a Google API Client payload exception package."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        # Map internal attributes to mirror googleapiclient.errors.HttpError structures
        self.resp = MagicMock()
        self.resp.status = status_code


def test_successful_api_call_without_retries():
    """Scenario: Verify operations complete instantly under optimal network states."""
    mock_callable = MagicMock(return_value={"file_id": "drive_uuid_101"})

    result = GoogleDriveResilientClient.execute_with_backoff(mock_callable)

    assert result["file_id"] == "drive_uuid_101"
    mock_callable.assert_called_once()


def test_backoff_recovery_on_rate_limits():
    """Scenario: API triggers initial HTTP 429 drops but resolves successfully within max retries."""
    mock_callable = MagicMock()
    # Mock behavior: Drop twice with 429, then return a valid file metadata dictionary
    mock_callable.side_effect = [
        MockGoogleAPIError(429),
        MockGoogleAPIError(429),
        {"file_id": "recovered_drive_uuid"},
    ]

    result = GoogleDriveResilientClient.execute_with_backoff(
        mock_callable,
        base_delay_seconds=0.01,  # Squashing delay window size purely for fast local test suites
    )

    assert result["file_id"] == "recovered_drive_uuid"
    assert mock_callable.call_count == 3


def test_immediate_failure_on_non_retryable_codes():
    """Scenario: Ensure critical structural errors (like HTTP 401 Unauthorized) crash directly without stalling."""
    mock_callable = MagicMock(side_effect=MockGoogleAPIError(401))

    with pytest.raises(MockGoogleAPIError) as exc_info:
        GoogleDriveResilientClient.execute_with_backoff(mock_callable)

    assert exc_info.value.status_code == 401
    mock_callable.assert_called_once()
