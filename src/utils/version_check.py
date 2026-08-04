"""
version_check.py
----------------
Async utility for checking whether a newer release exists on GitHub.

The function ``fetch_latest_github_version`` makes a single GET request to
the GitHub Releases API endpoint::

    GET https://api.github.com/repos/{owner}/{repo}/releases/latest

It returns the tag name of the latest published release (e.g. ``"v1.2.0"``).
The ``is_update_available`` helper compares that tag against the locally
running ``APP_VERSION`` string (e.g. ``"1.0.0"`` or ``"v1.0.0"``) using
standard ``packaging.version`` semantics, so pre-release suffixes such as
``-rc1`` are handled correctly.

Usage example::

    import asyncio
    from src.utils.version_check import fetch_latest_github_version, is_update_available, APP_VERSION

    latest = asyncio.run(fetch_latest_github_version())
    if latest and is_update_available(APP_VERSION, latest):
        print(f"Update available: {latest}")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Local version ──────────────────────────────────────────────────────────────
# Bump this constant in lock-step with CHANGELOG.md when cutting a new release.
APP_VERSION: str = "1.0.0"

# ── GitHub repository coordinates ─────────────────────────────────────────────
GITHUB_OWNER: str = "Ganesh-403"
GITHUB_REPO: str = "semantic-plagiarism-detector"
GITHUB_RELEASES_URL: str = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# Timeout (seconds) for the outbound HTTP request. Kept short so a slow/absent
# network doesn't block the UI render.
_REQUEST_TIMEOUT: float = 5.0


def _normalise_tag(tag: str) -> str:
    """Strip a leading ``v`` from a version tag so comparisons are stable.

    Parameters
    ----------
    tag:
        Raw tag string, e.g. ``"v1.2.0"`` or ``"1.2.0"``.

    Returns
    -------
    str
        Version string without a leading ``v``, e.g. ``"1.2.0"``.
    """
    return tag.lstrip("v")


async def fetch_latest_github_version(
    url: str = GITHUB_RELEASES_URL,
    timeout: float = _REQUEST_TIMEOUT,
) -> Optional[str]:
    """Return the tag name of the latest GitHub release, or ``None`` on failure.

    The request is fire-and-forget from the UI's perspective: any network
    error, timeout, or unexpected API response is logged at DEBUG level and
    ``None`` is returned so the caller can degrade gracefully.

    Parameters
    ----------
    url:
        The GitHub releases API endpoint to query.  Override in tests.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    str | None
        The raw tag string (e.g. ``"v1.2.0"``), or ``None`` if the request
        failed or the response did not contain a ``tag_name`` field.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                headers={"Accept": "application/vnd.github+json"},
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
            tag: Optional[str] = data.get("tag_name")
            if not tag:
                logger.debug(
                    "GitHub releases API response missing 'tag_name': %s", data
                )
            return tag
    except Exception as exc:  # noqa: BLE001 – network errors are non-fatal
        logger.debug("Version check request failed: %s", exc)
        return None


def is_update_available(local_version: str, remote_tag: str) -> bool:
    """Return ``True`` when *remote_tag* is strictly newer than *local_version*.

    Both strings are normalised (leading ``v`` stripped) before comparison.
    Falls back to a plain string inequality check when ``packaging`` is not
    installed.

    Parameters
    ----------
    local_version:
        The version string of the currently running application, e.g.
        ``"1.0.0"``.
    remote_tag:
        The tag name returned by the GitHub API, e.g. ``"v1.2.0"``.

    Returns
    -------
    bool
        ``True`` if a newer version is available, ``False`` otherwise.
    """
    local = _normalise_tag(local_version)
    remote = _normalise_tag(remote_tag)
    try:
        from packaging.version import Version  # type: ignore[import-untyped]

        return Version(remote) > Version(local)
    except Exception:  # noqa: BLE001 – packaging not installed or bad tag
        return remote != local


def check_for_update_sync(
    local_version: str = APP_VERSION,
    url: str = GITHUB_RELEASES_URL,
    timeout: float = _REQUEST_TIMEOUT,
) -> Optional[str]:
    """Synchronous wrapper around :func:`fetch_latest_github_version`.

    Returns the remote tag when an update is available, or ``None`` when the
    current version is already the latest (or the check could not be performed).

    This is the primary entry-point used by the Streamlit UI because Streamlit
    re-runs are synchronous; the async helper is preserved for callers that
    already live inside an event loop.

    Parameters
    ----------
    local_version:
        The version string of the currently running application.
    url:
        GitHub releases API endpoint override (useful for testing).
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    str | None
        The newer tag string, or ``None``.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            remote_tag = loop.run_until_complete(
                fetch_latest_github_version(url=url, timeout=timeout)
            )
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("check_for_update_sync failed: %s", exc)
        return None

    if remote_tag and is_update_available(local_version, remote_tag):
        return remote_tag
    return None
