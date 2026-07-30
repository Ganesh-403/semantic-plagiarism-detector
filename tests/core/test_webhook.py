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
    assert send_plagiarism_alert("DocA", "DocB", 0.95) is False


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

    result = send_plagiarism_alert(
        "student_essay.pdf",
        "wikipedia_source.pdf",
        0.925,
    )

    assert result is True
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

    result = send_plagiarism_alert("DocA", "DocB", 0.99)

    assert result is False
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

    result = send_plagiarism_alert("DocA", "DocB", 0.91)

    assert result is True
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

    result = send_plagiarism_alert("DocA", "DocB", 0.93)

    assert result is True
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

    result = send_plagiarism_alert("DocA", "DocB", 0.94)

    assert result is False
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

    result = send_plagiarism_alert("DocA", "DocB", 0.96)

    assert result is False
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)
    mock_post.assert_not_called()
