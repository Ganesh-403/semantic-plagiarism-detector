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
import os
import time
from typing import Optional

import httpx

from src.version import APP_VERSION

logger = logging.getLogger(__name__)

# ── GitHub repository coordinates ─────────────────────────────────────────────
GITHUB_OWNER: str = "Ganesh-403"
GITHUB_REPO: str = "semantic-plagiarism-detector"
GITHUB_RELEASES_URL: str = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# Timeout (seconds) for the outbound HTTP request. Kept short so a slow/absent
# network doesn't block the UI render.
_REQUEST_TIMEOUT: float = 5.0

# ── In-memory response cache ──────────────────────────────────────────────────
_CACHE_TTL_SECONDS: float = 3600.0  # 1 hour
_version_cache: dict[str, tuple[float, Optional[str]]] = {}


def clear_version_cache() -> None:
    """Clear the in-memory GitHub version check response cache."""
    _version_cache.clear()


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
    use_cache: bool = True,
) -> Optional[str]:
    """Return the tag name of the latest GitHub release, or ``None`` on failure.

    Supports token authentication via ``GITHUB_TOKEN`` or ``GH_TOKEN`` env vars
    to increase GitHub API rate limits up to 5,000 requests/hour. Responses are
    cached in memory for 1 hour to prevent redundant requests during Streamlit re-runs.

    The request is fire-and-forget from the UI's perspective: any network
    error, timeout, or unexpected API response is logged at DEBUG level and
    ``None`` is returned so the caller can degrade gracefully.

    Parameters
    ----------
    url:
        The GitHub releases API endpoint to query. Override in tests.
    timeout:
        HTTP request timeout in seconds.
    use_cache:
        Whether to check and update the 1-hour in-memory cache.

    Returns
    -------
    str | None
        The raw tag string (e.g. ``"v1.2.0"``), or ``None`` if the request
        failed or the response did not contain a ``tag_name`` field.
    """
    now = time.time()
    if use_cache and url in _version_cache:
        cached_time, cached_tag = _version_cache[url]
        if now - cached_time < _CACHE_TTL_SECONDS:
            return cached_tag

    headers = {"Accept": "application/vnd.github+json"}
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if token:
        if token.startswith("Bearer ") or token.startswith("token "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"

    try:
        custom_headers = {
            "User-Agent": f"SemanticPlagiarismDetector/{APP_VERSION}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient(headers=custom_headers, timeout=timeout) as client:
            response = await client.get(
                url,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
            tag: Optional[str] = data.get("tag_name")
            if not tag:
                logger.debug(
                    "GitHub releases API response missing 'tag_name': %s", data
                )
            if use_cache:
                _version_cache[url] = (now, tag)
            return tag
    except Exception as exc:  # noqa: BLE001 – network errors are non-fatal
        logger.debug("Version check request failed: %s", exc)
        if use_cache:
            _version_cache[url] = (now, None)
        return None


def _parse_semver_tuple(version: str) -> tuple[int, int, int]:
    """Parse the ``major.minor.patch`` core of *version* as a 3-int tuple.

    Used only as the fallback comparison for :func:`is_update_available` when
    ``packaging`` is unavailable, so this is deliberately basic: it looks at
    the first three dot-separated numeric components and ignores everything
    else (pre-release/build suffixes such as ``-rc1`` or ``+build.5``).
    A missing or non-numeric component defaults to ``0`` rather than raising,
    so a malformed tag degrades gracefully instead of crashing the comparison.

    Parameters
    ----------
    version:
        A normalised version string, e.g. ``"1.2.0"`` or ``"1.2.0-rc1"``.

    Returns
    -------
    tuple[int, int, int]
        The ``(major, minor, patch)`` components.
    """
    core = version.split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    major, minor, patch = (
        int(parts[i]) if i < len(parts) and parts[i].isdigit() else 0
        for i in range(3)
    )
    return (major, minor, patch)


def is_update_available(local_version: str, remote_tag: str) -> bool:
    """Return ``True`` when *remote_tag* is strictly newer than *local_version*.

    Both strings are normalised (leading ``v`` stripped) before comparison.
    Falls back to a basic ``(major, minor, patch)`` integer-tuple comparison
    when ``packaging`` is not installed (or a tag fails to parse as a
    ``packaging.version.Version``), rather than a plain string inequality
    check — a string-inequality fallback would report an update as available
    for *any* difference, including when the remote version is actually
    older than the local one.

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
        return _parse_semver_tuple(remote) > _parse_semver_tuple(local)


def check_for_update_sync(
    local_version: str = APP_VERSION,
    url: str = GITHUB_RELEASES_URL,
    timeout: float = _REQUEST_TIMEOUT,
    use_cache: bool = True,
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
    use_cache:
        Whether to check and update the 1-hour in-memory cache.

    Returns
    -------
    str | None
        The newer tag string, or ``None``.
    """
    try:
        remote_tag = asyncio.run(
            fetch_latest_github_version(url=url, timeout=timeout, use_cache=use_cache)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("check_for_update_sync failed: %s", exc)
        return None

    if remote_tag and is_update_available(local_version, remote_tag):
        return remote_tag
    return None
