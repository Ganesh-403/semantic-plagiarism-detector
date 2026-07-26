"""Application-level environment configuration."""

from __future__ import annotations

import os
from typing import Final

DEFAULT_APP_TITLE: Final[str] = (
    "Semantic Plagiarism Detection System"
)


def get_app_title() -> str:
    """Return the configured application title.

    Empty or whitespace-only values fall back to the default so a
    malformed environment variable cannot leave the browser or page
    title blank.
    """
    configured_title = os.getenv("APP_TITLE", "").strip()
    return configured_title or DEFAULT_APP_TITLE

def get_lock_timeout() -> int:
    """Return the configured lock timeout in seconds (default 30)."""
    try:
        timeout = int(os.getenv("LOCK_TIMEOUT_SECONDS", "30"))
        return max(1, timeout)
    except ValueError:
        return 30
