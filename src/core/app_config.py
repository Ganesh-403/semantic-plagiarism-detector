"""Application-level environment configuration."""

from __future__ import annotations

import os
from typing import Final

DEFAULT_APP_TITLE: Final[str] = "Semantic Plagiarism Detection System"
DEFAULT_PDF_FOOTER_TEXT: Final[str] = ""


SUPPORTED_OCR_LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
}



def get_app_title() -> str:
    """Return the configured application title.

    Empty or whitespace-only values fall back to the default so a
    malformed environment variable cannot leave the browser or page
    title blank.
    """
    configured_title = os.getenv("APP_TITLE", "").strip()
    return configured_title or DEFAULT_APP_TITLE


def get_pdf_footer_text() -> str:
    """Return the configured PDF footer text.

    Empty or whitespace-only values fall back to the default.
    """
    configured_footer = os.getenv("PDF_FOOTER_TEXT", "").strip()
    return configured_footer or DEFAULT_PDF_FOOTER_TEXT


def get_welcome_message() -> str:
    """Return the configured dashboard welcome banner message.

    Empty or whitespace-only values return an empty string, meaning no
    banner is shown by default.
    """
    return os.getenv("APP_WELCOME_MESSAGE", "").strip()


def get_lock_timeout() -> int:
    """Return the configured lock timeout in seconds (default 30)."""
    try:
        timeout = int(os.getenv("LOCK_TIMEOUT_SECONDS", "30"))
        return max(1, timeout)
    except ValueError:
        return 30


def get_verification_base_url() -> str:
    """Return the configured base URL for report verification."""
    return os.getenv("VERIFICATION_BASE_URL", "https://example.com/verify").strip()
