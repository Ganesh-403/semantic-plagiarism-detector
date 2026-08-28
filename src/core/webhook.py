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
src/core/webhook.py
-------------------
Webhook notification delivery with transient-failure retries and HMAC signatures.

Provides secure webhook delivery with:
- Cryptographic HMAC-SHA256 signatures for payload authenticity
- Transient-failure retries with exponential backoff
- SSRF protection for outbound URLs
- Thread-safe attempt counting for concurrent background tasks (Issue #1994)

Recent Additions (Issue #1994):
- Replaced mutable global _attempt_counter with threading.local() to prevent
  race conditions when multiple webhook deliveries run concurrently via
  background_tasks thread pools.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.security.ssrf_protector import SSRFProtector, SSRFSecurityException

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

# Thread-local storage for tracking retry attempts per thread.
# This replaces the previous mutable global `_attempt_counter` which caused
# race conditions when multiple webhooks were dispatched concurrently
# via FastAPI's BackgroundTasks or asyncio thread pools (Issue #1994).
_thread_local = threading.local()


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
        return response is not None and response.status_code in _RETRYABLE_STATUS_CODES

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


def compute_webhook_signature(
    payload_bytes: bytes,
    secret_key: str,
    timestamp: Optional[int] = None,
) -> str:
    """
    Compute HMAC-SHA256 signature for webhook payload authentication.

    Generates a cryptographic signature that receivers can use to verify
    the authenticity and integrity of webhook payloads. The signature is
    computed over the raw payload bytes using the shared secret key.

    Args:
        payload_bytes: The raw bytes of the webhook payload (JSON encoded).
        secret_key: The shared secret key used for HMAC computation.
        timestamp: Optional Unix timestamp to include in signature.
                   If None, current time is used.

    Returns:
        Hexadecimal string of the HMAC-SHA256 signature.

    Examples:
        >>> payload = json.dumps({"text": "alert"}).encode()
        >>> sig = compute_webhook_signature(payload, "my_secret_key")
        >>> print(sig)
        'a1b2c3d4e5f6...'
    """
    if not secret_key:
        logger.warning("compute_webhook_signature: No secret key provided")
        return ""

    if timestamp is None:
        timestamp = int(time.time())

    # Create the signed payload: timestamp + payload
    # This prevents replay attacks by binding signature to specific time
    signed_content = f"{timestamp}.".encode("utf-8") + payload_bytes

    signature = hmac.new(
        secret_key.encode("utf-8"),
        signed_content,
        hashlib.sha256,
    ).hexdigest()

    return signature


def verify_webhook_signature(
    payload_bytes: bytes,
    signature: str,
    secret_key: str,
    timestamp: Optional[int] = None,
    max_age_seconds: int = 300,
) -> bool:
    """
    Verify the authenticity of a webhook payload using HMAC-SHA256.

    Recomputes the signature using the provided secret key and compares
    it with the received signature using constant-time comparison to
    prevent timing attacks.

    Args:
        payload_bytes: The raw bytes of the received webhook payload.
        signature: The signature string to verify (hex format).
        secret_key: The shared secret key used for verification.
        timestamp: The timestamp from the webhook header. If None,
                   timestamp validation is skipped.
        max_age_seconds: Maximum allowed age of the webhook in seconds.
                         Defaults to 300 (5 minutes).

    Returns:
        True if signature is valid and timestamp is within acceptable range,
        False otherwise.

    Examples:
        >>> payload = json.dumps({"text": "alert"}).encode()
        >>> sig = compute_webhook_signature(payload, "secret", timestamp=1234567890)
        >>> verify_webhook_signature(payload, sig, "secret", timestamp=1234567890)
        True
    """
    if not secret_key or not signature:
        logger.warning("verify_webhook_signature: Missing secret key or signature")
        return False

    # Validate timestamp to prevent replay attacks
    if timestamp is not None:
        current_time = int(time.time())
        age = abs(current_time - timestamp)

        if age > max_age_seconds:
            logger.warning(
                "verify_webhook_signature: Timestamp too old (%d seconds). "
                "Max allowed: %d seconds.",
                age,
                max_age_seconds,
            )
            return False

    # Recompute the expected signature
    expected_signature = compute_webhook_signature(
        payload_bytes,
        secret_key,
        timestamp=timestamp,
    )

    if not expected_signature:
        return False

    # Use constant-time comparison to prevent timing attacks
    # hmac.compare_digest is designed for this purpose
    is_valid = hmac.compare_digest(
        expected_signature.encode("utf-8"),
        signature.encode("utf-8"),
    )

    if not is_valid:
        logger.warning("verify_webhook_signature: Signature mismatch detected")

    return is_valid


def _get_attempt_counter() -> int:
    """Safely retrieve the current thread's attempt counter."""
    return getattr(_thread_local, "attempt_counter", 0)


def _increment_attempt_counter() -> None:
    """Safely increment the current thread's attempt counter."""
    if not hasattr(_thread_local, "attempt_counter"):
        _thread_local.attempt_counter = 0
    _thread_local.attempt_counter += 1


def _reset_attempt_counter() -> None:
    """Safely reset the current thread's attempt counter to zero."""
    _thread_local.attempt_counter = 0


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
    """POST one webhook attempt with HMAC signature header.

    Tenacity retries this function for transient request failures only.
    Adds X-Plagiarism-Signature header for payload authentication.

    Thread Safety (Issue #1994):
        Uses thread-local storage to track attempt counts, ensuring that
        concurrent webhook deliveries do not clobber each other's counters.
    """
    _increment_attempt_counter()

    # Serialize payload to JSON bytes for signature computation
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")

    # Get secret key from environment
    secret_key = os.getenv("WEBHOOK_SECRET_KEY", "")

    # Generate timestamp and signature
    timestamp = int(time.time())
    signature = compute_webhook_signature(payload_bytes, secret_key, timestamp)

    # Construct signature header: t=timestamp,v1=signature
    signature_header = f"t={timestamp},v1={signature}" if signature else ""

    headers = {
        "Content-Type": "application/json",
    }

    # Only add signature header if we have a valid signature
    if signature_header:
        headers["X-Plagiarism-Signature"] = signature_header

    response = requests.post(
        webhook_url,
        json=payload,
        headers=headers,
        timeout=WEBHOOK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def send_plagiarism_alert(
    doc_a: str,
    doc_b: str,
    similarity: float,
    webhook_url: str | None = None,  # Added parameter for override (Issue #1995)
) -> tuple[bool, int]:
    """Send a plagiarism alert to the configured webhook with retry logic.

    Thread Safety (Issue #1994):
        The attempt counter is now tracked via thread-local storage, ensuring
        that concurrent calls to this function from different threads (e.g.,
        FastAPI BackgroundTasks) maintain isolated counters.

    Args:
        doc_a: Name of the first student document.
        doc_b: Name of the second student document.
        similarity: Cosine similarity score between 0.0 and 1.0.
        webhook_url: Optional explicit webhook URL. If provided, this overrides
                     the PLAGIARISM_WEBHOOK_URL environment variable. This is
                     useful for routing alerts to different endpoints based on
                     severity or tenant configuration.

    Returns:
        A tuple of `(success, total_attempts)` where `success` is a boolean
        indicating delivery success, and `total_attempts` is the total number
        of HTTP delivery attempts made by the current thread.
    """
    # Reset the thread-local counter before starting a new delivery sequence
    _reset_attempt_counter()

    # Issue #1995: Use explicit webhook_url if provided, otherwise fallback to env var
    if webhook_url is None:
        webhook_url = os.getenv("PLAGIARISM_WEBHOOK_URL")

    if not webhook_url:
        logger.warning(
            "PLAGIARISM_WEBHOOK_URL is not configured in the environment and "
            "no explicit webhook_url was provided."
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
        SSRFProtector.validate_webhook_url(webhook_url)
    except SSRFSecurityException as exception:
        logger.error(
            "SECURITY BLOCKED: Webhook failed SSRF validation: %s",
            exception,
        )
        return False, 0

    try:
        _post_webhook(webhook_url, payload)
        attempts = _get_attempt_counter()
    except requests.exceptions.RequestException as exception:
        attempts = _get_attempt_counter()
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


def dispatch_plagiarism_alert(
    doc_a: str,
    doc_b: str,
    similarity: float,
    webhook_url: str | None = None,
) -> bool:
    """Dispatch a plagiarism alert payload to the configured webhook endpoint.

    Args:
        doc_a: Name of the first student document.
        doc_b: Name of the second student document.
        similarity: Cosine similarity score between 0.0 and 1.0.
        webhook_url: Optional explicit webhook URL to override the environment variable.

    Returns:
        True if the alert was successfully delivered, False otherwise.
    """
    # Issue #1995: Pass webhook_url through to send_plagiarism_alert
    success, _ = send_plagiarism_alert(
        doc_a=doc_a,
        doc_b=doc_b,
        similarity=similarity,
        webhook_url=webhook_url,  # Explicitly pass the override parameter
    )
    return success


class EventDispatcher:
    """Service for dispatching webhook events to registered external LMS systems.

    Allows external LMS systems (Canvas, Blackboard, Moodle, custom REST clients)
    that trigger background document analysis scans to receive POST notifications
    with JSON payloads upon completion, eliminating the need for continuous polling.

    Features:
    - SSRF URL validation prior to HTTP dispatch (`SSRFProtector.validate_webhook_url`)
    - Cryptographic HMAC-SHA256 payload signatures (`X-Plagiarism-Signature`)
    - Transient-failure retries with exponential backoff
    - LMS endpoint registry management
    """

    def __init__(
        self,
        default_webhook_url: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.default_webhook_url = (
            default_webhook_url
            or os.getenv("LMS_WEBHOOK_URL")
            or os.getenv("PLAGIARISM_WEBHOOK_URL")
        )
        self.secret_key = secret_key or os.getenv("WEBHOOK_SECRET_KEY", "")
        self._registry: dict[str, dict[str, str]] = {}

    def register_lms_webhook(
        self,
        lms_id: str,
        webhook_url: str,
        secret_key: Optional[str] = None,
    ) -> None:
        """Register a webhook endpoint for a specific LMS tenant or system."""
        SSRFProtector.validate_webhook_url(webhook_url)
        self._registry[lms_id] = {
            "webhook_url": webhook_url,
            "secret_key": secret_key or self.secret_key or "",
        }

    def unregister_lms_webhook(self, lms_id: str) -> bool:
        """Remove a registered LMS webhook endpoint."""
        return self._registry.pop(lms_id, None) is not None

    def get_registered_lms(self, lms_id: str) -> Optional[dict[str, str]]:
        """Retrieve details of a registered LMS webhook endpoint."""
        return self._registry.get(lms_id)

    def dispatch(
        self,
        event_type: str,
        payload: dict[str, Any],
        webhook_url: Optional[str] = None,
        secret_key: Optional[str] = None,
        lms_id: Optional[str] = None,
    ) -> tuple[bool, int]:
        """Dispatch a webhook event payload to a target LMS webhook URL.

        Args:
            event_type: Name of the event (e.g. 'document.analysis.complete').
            payload: JSON-serializable event data dictionary.
            webhook_url: Explicit webhook URL target. If None, resolves from lms_id or default_webhook_url.
            secret_key: Optional secret key for HMAC signature computation.
            lms_id: Optional registered LMS tenant ID.

        Returns:
            Tuple of (success: bool, attempts: int).
        """
        target_url = webhook_url
        target_secret = secret_key

        if not target_url and lms_id and lms_id in self._registry:
            reg_info = self._registry[lms_id]
            target_url = reg_info.get("webhook_url")
            if not target_secret:
                target_secret = reg_info.get("secret_key")

        if not target_url:
            target_url = self.default_webhook_url

        if not target_secret:
            target_secret = self.secret_key or ""

        if not target_url:
            logger.warning(
                "EventDispatcher.dispatch: No webhook URL configured or registered for event '%s'.",
                event_type,
            )
            return False, 0

        # Enforce SSRF validation
        try:
            SSRFProtector.validate_webhook_url(target_url)
        except SSRFSecurityException as exc:
            logger.error(
                "EventDispatcher.dispatch SECURITY BLOCKED: Webhook URL '%s' failed SSRF validation: %s",
                target_url,
                exc,
            )
            return False, 0

        # Standardized event envelope
        full_payload = {
            "event": event_type,
            "timestamp": int(time.time()),
            "data": payload,
        }

        _reset_attempt_counter()
        old_secret = os.getenv("WEBHOOK_SECRET_KEY")
        if target_secret:
            os.environ["WEBHOOK_SECRET_KEY"] = target_secret
        try:
            _post_webhook(target_url, full_payload)
            attempts = _get_attempt_counter()
            logger.info(
                "EventDispatcher: Successfully dispatched event '%s' to '%s' after %d attempt(s).",
                event_type,
                target_url,
                attempts,
            )
            return True, attempts
        except requests.exceptions.RequestException as exc:
            attempts = _get_attempt_counter()
            logger.error(
                "EventDispatcher: Failed to dispatch event '%s' to '%s' after %d attempt(s): %s",
                event_type,
                target_url,
                attempts,
                exc,
            )
            return False, attempts
        finally:
            if old_secret is None:
                os.environ.pop("WEBHOOK_SECRET_KEY", None)
            else:
                os.environ["WEBHOOK_SECRET_KEY"] = old_secret

    def dispatch_analysis_complete(
        self,
        document_id: str,
        filename: str,
        similarity_score: float,
        matches_count: int = 0,
        status: str = "completed",
        webhook_url: Optional[str] = None,
        lms_id: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, int]:
        """Dispatch a 'document.analysis.complete' webhook notification once document analysis finishes.

        Args:
            document_id: Unique identifier for the analyzed document.
            filename: Original filename of the processed document.
            similarity_score: Highest similarity score detected (0.0 to 1.0).
            matches_count: Total number of matching document/chunk pairs flagged.
            status: Processing status ('completed', 'failed', 'flagged').
            webhook_url: Optional target URL override.
            lms_id: Optional LMS tenant ID.
            extra_metadata: Optional dict of extra details to attach to the payload.

        Returns:
            Tuple of (success: bool, attempts: int).
        """
        payload = {
            "document_id": str(document_id),
            "filename": str(filename),
            "status": status,
            "similarity_score": float(similarity_score),
            "similarity_percentage": round(float(similarity_score) * 100, 2),
            "matches_count": int(matches_count),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_metadata:
            payload["metadata"] = extra_metadata

        return self.dispatch(
            event_type="document.analysis.complete",
            payload=payload,
            webhook_url=webhook_url,
            lms_id=lms_id,
        )
