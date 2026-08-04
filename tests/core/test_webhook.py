import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.webhook import send_plagiarism_alert


WEBHOOK_URL = "https://mock-webhook.url"


def make_response(status_code: int) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code

    if status_code >= 400:
        response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(
                f"{status_code} response",
                response=response,
            )
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
    mock_post.side_effect = requests.exceptions.ConnectionError(
        "Connection timed out"
    )

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

    mock_validate_url.side_effect = SSRFSecurityException(
        "blocked destination"
    )

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
