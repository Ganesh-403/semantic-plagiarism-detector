"""Webhook notification delivery with transient-failure retries."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.security.ssrf_protector import (
    SSRFProtector,
    SSRFSecurityException,
)


logger = logging.getLogger(__name__)

load_dotenv()

WEBHOOK_TIMEOUT_SECONDS = 10
WEBHOOK_MAX_ATTEMPTS = 3
WEBHOOK_RETRY_MIN_SECONDS = 1
WEBHOOK_RETRY_MAX_SECONDS = 4

_RETRYABLE_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


def _is_retryable_request_error(exception: BaseException) -> bool:
    """Return whether a webhook request error is temporary.

    Connection errors and timeouts are considered transient. HTTP responses
    are retried only for throttling, request timeout, and common server-side
    failures. Permanent client errors such as 400 or 401 are not retried.
    """
    if isinstance(
        exception,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    ):
        return True

    if isinstance(exception, requests.exceptions.HTTPError):
        response = exception.response
        return (
            response is not None
            and response.status_code in _RETRYABLE_STATUS_CODES
        )

    return False


def _log_retry(retry_state: RetryCallState) -> None:
    """Log a webhook retry without exposing the payload or secret URL."""
    exception = retry_state.outcome.exception()
    wait_seconds = retry_state.next_action.sleep

    logger.warning(
        "Webhook attempt %s/%s failed with %s. Retrying in %.1f seconds.",
        retry_state.attempt_number,
        WEBHOOK_MAX_ATTEMPTS,
        exception,
        wait_seconds,
    )


_attempt_counter = 0


@retry(
    retry=retry_if_exception(_is_retryable_request_error),
    stop=stop_after_attempt(WEBHOOK_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=WEBHOOK_RETRY_MIN_SECONDS,
        max=WEBHOOK_RETRY_MAX_SECONDS,
    ),
    before_sleep=_log_retry,
    reraise=True,
)
def _post_webhook(
    webhook_url: str,
    payload: dict[str, Any],
) -> requests.Response:
    """POST one webhook attempt.

    Tenacity retries this function for transient request failures only.
    """
    global _attempt_counter
    _attempt_counter += 1
    response = requests.post(
        webhook_url,
        json=payload,
        timeout=WEBHOOK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def send_plagiarism_alert(
    doc_a: str,
    doc_b: str,
    similarity: float,
) -> tuple[bool, int]:
    """Send a plagiarism alert to the configured webhook with retry logic.

    The webhook URL is validated once before any outbound request. Temporary
    connection failures, timeouts, rate limiting, and selected 5xx responses
    (500, 502, 503, 504) are retried up to 3 times with exponential backoff.

    Args:
        doc_a: Name of the first student document.
        doc_b: Name of the second student document.
        similarity: Cosine similarity score between 0.0 and 1.0.

    Returns:
        A tuple of ``(success, total_attempts)`` where ``success`` is a boolean
        indicating delivery success, and ``total_attempts`` is the total number
        of HTTP delivery attempts made.
    """
    global _attempt_counter
    _attempt_counter = 0

    webhook_url = os.getenv("PLAGIARISM_WEBHOOK_URL")

    if not webhook_url:
        logger.warning(
            "PLAGIARISM_WEBHOOK_URL is not configured in the environment."
        )
        return False, 0

    base_url = os.getenv(
        "APP_BASE_URL",
        "http://localhost:8501",
    ).rstrip("/")
    similarity_percent = similarity * 100

    message = (
        "🚨 *Plagiarism Alert!* "
        f"Student document *{doc_a}* matches *{doc_b}* by "
        f"*{similarity_percent:.1f}%*.\n"
        f"Review details here: {base_url}"
    )
    payload = {
        "text": message,
        "content": message,
    }

    try:
        # Validate once. Retrying cannot make an unsafe URL safe.
        SSRFProtector.validate_webhook_url(webhook_url)
    except SSRFSecurityException as exception:
        logger.error(
            "SECURITY BLOCKED: Webhook failed SSRF validation: %s",
            exception,
        )
        return False, 0

    try:
        _post_webhook(webhook_url, payload)
        attempts = _attempt_counter
    except requests.exceptions.RequestException as exception:
        attempts = _attempt_counter
        logger.error(
            "Failed to send webhook notification for pair %s <-> %s "
            "after %s attempt(s): %s",
            doc_a,
            doc_b,
            attempts,
            exception,
        )
        return False, attempts

    logger.info(
        "Webhook alert successfully sent for pair %s <-> %s (%.1f%%) after %s attempt(s).",
        doc_a,
        doc_b,
        similarity_percent,
        attempts,
    )
    return True, attempts
# Alias or main function for dispatching plagiarism alerts
def dispatch_plagiarism_alert(doc_a: str, doc_b: str, similarity: float, webhook_url: str | None = None) -> bool:
    """Dispatch a plagiarism alert payload to the configured webhook endpoint."""
    return send_plagiarism_alert(doc_a=doc_a, doc_b=doc_b, similarity=similarity, webhook_url=webhook_url)