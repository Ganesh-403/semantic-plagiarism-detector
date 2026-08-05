import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.webhook import send_plagiarism_alert

import json
import time

from src.core.webhook import (
    compute_webhook_signature,
    verify_webhook_signature,
)


WEBHOOK_URL = "https://mock-webhook.url"


def make_response(status_code: int) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code

    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} response",
            response=response,
        )

    return response


@pytest.fixture(autouse=True)
def disable_retry_wait(monkeypatch):
    """Keep retry tests immediate while retaining production backoff."""
    from src.core import webhook

    monkeypatch.setattr(
        webhook._post_webhook.retry,
        "sleep",
        lambda _seconds: None,
    )


@patch.dict(os.environ, {}, clear=True)
def test_send_plagiarism_alert_no_url():
    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)
    assert success is False
    assert attempts == 0


@patch.dict(
    os.environ,
    {
        "PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL,
        "APP_BASE_URL": "http://test-dashboard",
    },
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_send_plagiarism_alert_success(
    mock_post,
    mock_validate_url,
):
    mock_post.return_value = make_response(200)

    success, attempts = send_plagiarism_alert(
        "student_essay.pdf",
        "wikipedia_source.pdf",
        0.925,
    )

    assert success is True
    assert attempts == 1
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)
    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args
    assert args[0] == WEBHOOK_URL
    assert kwargs["timeout"] == 10

    payload = kwargs["json"]
    assert "text" in payload
    assert "content" in payload
    assert "student_essay.pdf" in payload["text"]
    assert "wikipedia_source.pdf" in payload["text"]
    assert "92.5%" in payload["text"]
    assert "http://test-dashboard" in payload["text"]


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_connection_error_retries_three_times(
    mock_post,
    mock_validate_url,
):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection timed out")

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.99)

    assert success is False
    assert attempts == 3
    assert mock_post.call_count == 3
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_502_retries_then_succeeds(
    mock_post,
    mock_validate_url,
):
    mock_post.side_effect = [
        make_response(502),
        make_response(502),
        make_response(200),
    ]

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.91)

    assert success is True
    assert attempts == 3
    assert mock_post.call_count == 3
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_500_503_504_server_errors_retried(
    mock_post,
    mock_validate_url,
):
    mock_post.side_effect = [
        make_response(500),
        make_response(503),
        make_response(504),
    ]

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.88)

    assert success is False
    assert attempts == 3
    assert mock_post.call_count == 3
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_timeout_retries_then_succeeds(
    mock_post,
    mock_validate_url,
):
    mock_post.side_effect = [
        requests.exceptions.Timeout("temporary timeout"),
        make_response(200),
    ]

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.93)

    assert success is True
    assert attempts == 2
    assert mock_post.call_count == 2
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_permanent_400_error_is_not_retried(
    mock_post,
    mock_validate_url,
):
    mock_post.return_value = make_response(400)

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.94)

    assert success is False
    assert attempts == 1
    mock_post.assert_called_once()
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.requests.post")
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
def test_ssrf_failure_does_not_send_or_retry(
    mock_validate_url,
    mock_post,
):
    from src.security.ssrf_protector import SSRFSecurityException

    mock_validate_url.side_effect = SSRFSecurityException("blocked destination")

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.96)

    assert success is False
    assert attempts == 0
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)
    mock_post.assert_not_called()


@patch("src.security.ssrf_protector.socket.getaddrinfo")
@patch("src.core.webhook.requests.post")
def test_send_plagiarism_alert_with_domain_whitelist(
    mock_post,
    mock_getaddrinfo,
    monkeypatch,
):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    mock_post.return_value = make_response(200)

    monkeypatch.setenv("PLAGIARISM_WEBHOOK_URL", "https://discord.com/api/webhooks/123")
    monkeypatch.setenv("ALLOWED_WEBHOOK_DOMAINS", "slack.com, discord.com")

    # Allowed domain sends successfully
    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)
    assert success is True
    assert attempts == 1
    mock_post.assert_called_once()

    # Change webhook URL to unallowed domain
    mock_post.reset_mock()
    monkeypatch.setenv("PLAGIARISM_WEBHOOK_URL", "https://unallowed.org/webhook")

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)
    assert success is False
    assert attempts == 0
    mock_post.assert_not_called()


# ─── HMAC Signature Tests (Issue #1373) ───────────────────────────────────────


class TestWebhookHMACSignature:
    """Tests for webhook HMAC-SHA256 signature generation and verification."""

    def test_compute_signature_returns_hex_string(self):
        """Signature must be a valid hexadecimal string."""
        payload = json.dumps({"test": "data"}).encode("utf-8")
        signature = compute_webhook_signature(
            payload, "secret_key", timestamp=1234567890
        )

        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 produces 64 hex chars
        # Verify it's valid hex
        int(signature, 16)

    def test_compute_signature_deterministic(self):
        """Same inputs must produce identical signatures."""
        payload = json.dumps({"alert": "plagiarism"}).encode("utf-8")
        timestamp = 1234567890

        sig1 = compute_webhook_signature(payload, "my_secret", timestamp=timestamp)
        sig2 = compute_webhook_signature(payload, "my_secret", timestamp=timestamp)

        assert sig1 == sig2

    def test_compute_signature_different_payloads(self):
        """Different payloads must produce different signatures."""
        payload1 = json.dumps({"doc": "A"}).encode("utf-8")
        payload2 = json.dumps({"doc": "B"}).encode("utf-8")
        timestamp = 1234567890

        sig1 = compute_webhook_signature(payload1, "secret", timestamp=timestamp)
        sig2 = compute_webhook_signature(payload2, "secret", timestamp=timestamp)

        assert sig1 != sig2

    def test_compute_signature_different_secrets(self):
        """Different secrets must produce different signatures."""
        payload = json.dumps({"test": "data"}).encode("utf-8")
        timestamp = 1234567890

        sig1 = compute_webhook_signature(payload, "secret1", timestamp=timestamp)
        sig2 = compute_webhook_signature(payload, "secret2", timestamp=timestamp)

        assert sig1 != sig2

    def test_compute_signature_different_timestamps(self):
        """Different timestamps must produce different signatures."""
        payload = json.dumps({"test": "data"}).encode("utf-8")

        sig1 = compute_webhook_signature(payload, "secret", timestamp=1000)
        sig2 = compute_webhook_signature(payload, "secret", timestamp=2000)

        assert sig1 != sig2

    def test_compute_signature_empty_secret(self):
        """Empty secret key must return empty signature."""
        payload = json.dumps({"test": "data"}).encode("utf-8")
        signature = compute_webhook_signature(payload, "", timestamp=1234567890)

        assert signature == ""

    def test_verify_signature_valid(self):
        """Valid signature must pass verification."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        secret = "my_webhook_secret"
        timestamp = int(time.time())

        signature = compute_webhook_signature(payload, secret, timestamp=timestamp)

        is_valid = verify_webhook_signature(
            payload, signature, secret, timestamp=timestamp
        )

        assert is_valid is True

    def test_verify_signature_invalid(self):
        """Invalid signature must fail verification."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        secret = "my_webhook_secret"
        timestamp = int(time.time())

        # Use wrong signature
        is_valid = verify_webhook_signature(
            payload, "wrong_signature", secret, timestamp=timestamp
        )

        assert is_valid is False

    def test_verify_signature_wrong_secret(self):
        """Signature computed with different secret must fail."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        timestamp = int(time.time())

        signature = compute_webhook_signature(payload, "secret1", timestamp=timestamp)

        is_valid = verify_webhook_signature(
            payload, signature, "secret2", timestamp=timestamp
        )

        assert is_valid is False

    def test_verify_signature_tampered_payload(self):
        """Modified payload must fail signature verification."""
        original_payload = json.dumps({"alert": "original"}).encode("utf-8")
        tampered_payload = json.dumps({"alert": "tampered"}).encode("utf-8")
        secret = "my_secret"
        timestamp = int(time.time())

        signature = compute_webhook_signature(
            original_payload, secret, timestamp=timestamp
        )

        is_valid = verify_webhook_signature(
            tampered_payload, signature, secret, timestamp=timestamp
        )

        assert is_valid is False

    def test_verify_signature_expired_timestamp(self):
        """Signature with old timestamp must fail (replay attack prevention)."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        secret = "my_secret"

        # Use timestamp from 10 minutes ago
        old_timestamp = int(time.time()) - 600

        signature = compute_webhook_signature(payload, secret, timestamp=old_timestamp)

        is_valid = verify_webhook_signature(
            payload, signature, secret, timestamp=old_timestamp, max_age_seconds=300
        )

        assert is_valid is False

    def test_verify_signature_missing_secret(self):
        """Missing secret key must fail verification."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        signature = "some_signature"

        is_valid = verify_webhook_signature(
            payload, signature, "", timestamp=1234567890
        )

        assert is_valid is False

    def test_verify_signature_missing_signature(self):
        """Missing signature must fail verification."""
        payload = json.dumps({"alert": "test"}).encode("utf-8")
        secret = "my_secret"

        is_valid = verify_webhook_signature(payload, "", secret, timestamp=1234567890)

        assert is_valid is False

    @patch.dict(os.environ, {"WEBHOOK_SECRET_KEY": "test_secret_123"})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_post_webhook_includes_signature_header(self, mock_post, mock_validate_url):
        """Verify _post_webhook adds X-Plagiarism-Signature header."""
        mock_post.return_value = make_response(200)

        from src.core.webhook import _post_webhook

        payload = {"text": "test alert"}
        _post_webhook("https://webhook.url", payload)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]

        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        assert "X-Plagiarism-Signature" in headers

        signature_header = headers["X-Plagiarism-Signature"]
        assert signature_header.startswith("t=")
        assert ",v1=" in signature_header

    @patch.dict(os.environ, {"WEBHOOK_SECRET_KEY": ""})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_post_webhook_no_signature_without_secret(
        self, mock_post, mock_validate_url
    ):
        """Without WEBHOOK_SECRET_KEY, signature header should not be added."""
        mock_post.return_value = make_response(200)

        from src.core.webhook import _post_webhook

        payload = {"text": "test alert"}
        _post_webhook("https://webhook.url", payload)

        call_kwargs = mock_post.call_args[1]
        headers = call_kwargs.get("headers", {})

        # Should not have signature header if no secret
        assert "X-Plagiarism-Signature" not in headers
