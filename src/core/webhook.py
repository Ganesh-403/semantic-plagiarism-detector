"""
webhook.py
----------
Utility to dispatch notifications to a Slack or Discord webhook channel
when high-similarity plagiarism incidents (>= 90%) are detected.
"""

import logging
import os
import threading
import time
from collections import deque

import requests
from dotenv import load_dotenv

from src.security.ssrf_protector import SSRFProtector, SSRFSecurityException

# Set up logging
logger = logging.getLogger(__name__)

_WEBHOOK_RATE_LIMIT = 5
_WEBHOOK_RATE_WINDOW_SECONDS = 60.0
_rate_limit_lock = threading.Lock()
_webhook_dispatches: dict[str, deque[float]] = {}

# Load environment variables from .env
load_dotenv()


def _allow_webhook_dispatch(webhook_url: str) -> bool:
    """Return whether this webhook is below the per-minute dispatch limit."""
    now = time.monotonic()
    with _rate_limit_lock:
        dispatches = _webhook_dispatches.setdefault(webhook_url, deque())
        cutoff = now - _WEBHOOK_RATE_WINDOW_SECONDS
        while dispatches and dispatches[0] <= cutoff:
            dispatches.popleft()

        if len(dispatches) >= _WEBHOOK_RATE_LIMIT:
            return False

        dispatches.append(now)
        return True


def send_plagiarism_alert(doc_a: str, doc_b: str, similarity: float) -> bool:
    """
    Send an alert to the configured PLAGIARISM_WEBHOOK_URL when high-similarity matches occur.

    Args:
        doc_a: Name of the first student document.
        doc_b: Name of the second student document.
        similarity: Cosine similarity score (between 0.0 and 1.0).

    Returns:
        bool: True if the alert was successfully sent, False otherwise.
    """
    webhook_url = os.getenv("PLAGIARISM_WEBHOOK_URL")

    if not webhook_url:
        logger.warning("PLAGIARISM_WEBHOOK_URL is not configured in the environment.")
        return False

    if not _allow_webhook_dispatch(webhook_url):
        logger.warning(
            "Webhook rate limit reached; suppressing plagiarism alert for %s <-> %s.",
            doc_a,
            doc_b,
        )
        return False

    # Get base URL of the Streamlit dashboard for the review link
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip("/")

    # Format similarity percentage
    sim_percent = similarity * 100

    # Construct the message payload
    message = (
        f"🚨 *Plagiarism Alert!* Student document *{doc_a}* matches *{doc_b}* by *{sim_percent:.1f}%*.\n"
        f"Review details here: {base_url}"
    )

    # Webhook payload compatible with both Slack (expects 'text') and Discord (expects 'content')
    payload = {"text": message, "content": message}

    try:
        # Prevent Server-Side Request Forgery (Issue #301)
        SSRFProtector.validate_webhook_url(webhook_url)
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        # Check if request returned an unsuccessful status code (4xx, 5xx)
        response.raise_for_status()
        logger.info(
            f"Webhook alert successfully sent for pair: {doc_a} <-> {doc_b} ({sim_percent:.1f}%)"
        )
        return True
    except SSRFSecurityException as e:
        logger.error(f"SECURITY BLOCKED: Webhook {webhook_url} failed SSRF validation: {e}")
        return False
    except requests.exceptions.RequestException as e:
        # Gracefully handle all network / request failures so indexing is not blocked
        logger.error(
            f"Failed to send webhook notification for pair: {doc_a} <-> {doc_b}. Error: {e}"
        )
        return False

def dispatch_plagiarism_alert(doc_a: str, doc_b: str, similarity: float) -> None:
    """
    Asynchronously dispatches a plagiarism alert via the background thread pool.
    This prevents the UI from blocking during network requests.
    """
    from src.core.synchronization import run_background
    
    def _dispatch():
        try:
            send_plagiarism_alert(doc_a, doc_b, similarity)
        except Exception:
            logger.exception("Webhook dispatch failed")
            
    run_background(_dispatch)
