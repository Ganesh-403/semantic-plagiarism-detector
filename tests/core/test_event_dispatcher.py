"""
test_event_dispatcher.py
-------------------------
Unit tests for EventDispatcher service (LMS webhook integration).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.webhook import EventDispatcher, compute_webhook_signature
from src.security.ssrf_protector import SSRFSecurityException


@pytest.fixture
def dispatcher():
    return EventDispatcher(
        default_webhook_url="https://canvas.example.edu/api/v1/webhook",
        secret_key="test_secret_key_123",
    )


def test_event_dispatcher_init(dispatcher):
    assert dispatcher.default_webhook_url == "https://canvas.example.edu/api/v1/webhook"
    assert dispatcher.secret_key == "test_secret_key_123"


def test_register_and_unregister_lms_webhook(dispatcher):
    with patch("src.security.ssrf_protector.SSRFProtector.validate_webhook_url"):
        dispatcher.register_lms_webhook(
            lms_id="moodle_tenant_1",
            webhook_url="https://moodle.example.edu/webhook",
            secret_key="moodle_secret",
        )

        reg = dispatcher.get_registered_lms("moodle_tenant_1")
        assert reg is not None
        assert reg["webhook_url"] == "https://moodle.example.edu/webhook"
        assert reg["secret_key"] == "moodle_secret"

        unregistered = dispatcher.unregister_lms_webhook("moodle_tenant_1")
        assert unregistered is True
        assert dispatcher.get_registered_lms("moodle_tenant_1") is None


def test_dispatch_analysis_complete_success(dispatcher):
    """Verify dispatch_analysis_complete fires POST request with valid JSON payload to LMS."""
    with (
        patch("src.security.ssrf_protector.SSRFProtector.validate_webhook_url"),
        patch("src.core.webhook.requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        success, attempts = dispatcher.dispatch_analysis_complete(
            document_id="doc_88412",
            filename="essay_submission_1.pdf",
            similarity_score=0.85,
            matches_count=3,
            status="completed",
            extra_metadata={"course_id": "CS101", "assignment_id": "A4"},
        )

        assert success is True
        assert attempts == 1

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        url = mock_post.call_args.args[0]
        json_payload = call_kwargs["json"]
        headers = call_kwargs["headers"]

        assert url == "https://canvas.example.edu/api/v1/webhook"
        assert json_payload["event"] == "document.analysis.complete"
        assert json_payload["data"]["document_id"] == "doc_88412"
        assert json_payload["data"]["filename"] == "essay_submission_1.pdf"
        assert json_payload["data"]["similarity_score"] == 0.85
        assert json_payload["data"]["similarity_percentage"] == 85.0
        assert json_payload["data"]["matches_count"] == 3
        assert json_payload["data"]["metadata"]["course_id"] == "CS101"

        # Verify HMAC signature header presence
        assert "X-Plagiarism-Signature" in headers
        assert headers["X-Plagiarism-Signature"].startswith("t=")


def test_dispatch_analysis_complete_registered_lms_tenant(dispatcher):
    """Verify dispatching event for a specific registered LMS tenant."""
    with (
        patch("src.security.ssrf_protector.SSRFProtector.validate_webhook_url"),
        patch("src.core.webhook.requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        dispatcher.register_lms_webhook(
            lms_id="blackboard_99",
            webhook_url="https://blackboard.univ.edu/api/webhook",
            secret_key="bb_secret_99",
        )

        success, attempts = dispatcher.dispatch_analysis_complete(
            document_id="doc_771",
            filename="history_paper.docx",
            similarity_score=0.42,
            lms_id="blackboard_99",
        )

        assert success is True
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "https://blackboard.univ.edu/api/webhook"


def test_dispatch_ssrf_blocked_url(dispatcher):
    """Verify dispatch fails cleanly when SSRF validation rejects target URL."""
    with patch(
        "src.security.ssrf_protector.SSRFProtector.validate_webhook_url",
        side_effect=SSRFSecurityException("Blocked loopback IP address"),
    ):
        success, attempts = dispatcher.dispatch(
            event_type="document.analysis.complete",
            payload={"doc": "test"},
            webhook_url="https://127.0.0.1/internal",
        )

        assert success is False
        assert attempts == 0


def test_dispatch_no_webhook_url():
    """Verify dispatch returns False when no webhook URL is configured."""
    empty_dispatcher = EventDispatcher()
    success, attempts = empty_dispatcher.dispatch(
        event_type="test_event",
        payload={},
    )
    assert success is False
    assert attempts == 0
