"""
test_version_check_rate_limiting_issue_3961.py
-----------------------------------------------
Unit tests for Issue #3961: GitHub API rate limiting in version_check.py.
Tests GITHUB_TOKEN / GH_TOKEN authorization header support and 1-hour in-memory caching.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.version_check import (
    _CACHE_TTL_SECONDS,
    clear_version_cache,
    fetch_latest_github_version,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure in-memory version cache is cleared before and after each test."""
    clear_version_cache()
    yield
    clear_version_cache()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_github_token_header_support(monkeypatch):
    """Verify GITHUB_TOKEN environment variable is passed in Authorization header."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken123")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(return_value=None)
    mock_response.json.return_value = {"tag_name": "v2.5.0"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        tag = _run(fetch_latest_github_version(use_cache=False))

    assert tag == "v2.5.0"
    mock_client.get.assert_called_once()
    headers = mock_client.get.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer ghp_testtoken123"


def test_gh_token_header_support(monkeypatch):
    """Verify GH_TOKEN is used when GITHUB_TOKEN is missing."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_ghtoken456")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(return_value=None)
    mock_response.json.return_value = {"tag_name": "v2.5.0"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        tag = _run(fetch_latest_github_version(use_cache=False))

    assert tag == "v2.5.0"
    mock_client.get.assert_called_once()
    headers = mock_client.get.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer ghp_ghtoken456"


def test_in_memory_caching_within_ttl():
    """Verify subsequent requests within TTL return cached tag without calling HTTP client again."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(return_value=None)
    mock_response.json.return_value = {"tag_name": "v3.0.0"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    custom_url = "https://api.github.com/repos/test-owner/test-repo/releases/latest"

    with patch("httpx.AsyncClient", return_value=mock_client):
        # First call fetches from API and caches
        tag1 = _run(fetch_latest_github_version(url=custom_url, use_cache=True))
        # Second call returns from cache
        tag2 = _run(fetch_latest_github_version(url=custom_url, use_cache=True))

    assert tag1 == "v3.0.0"
    assert tag2 == "v3.0.0"
    # HTTP client get should only have been called once
    assert mock_client.get.call_count == 1


def test_in_memory_caching_expiration():
    """Verify cache expires after TTL (1 hour)."""
    mock_response1 = MagicMock()
    mock_response1.raise_for_status = MagicMock(return_value=None)
    mock_response1.json.return_value = {"tag_name": "v3.0.0"}

    mock_response2 = MagicMock()
    mock_response2.raise_for_status = MagicMock(return_value=None)
    mock_response2.json.return_value = {"tag_name": "v3.1.0"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=[mock_response1, mock_response2])

    custom_url = "https://api.github.com/repos/test-owner/test-repo/releases/latest"
    start_time = 100000.0

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("time.time", return_value=start_time):
            tag1 = _run(fetch_latest_github_version(url=custom_url, use_cache=True))
        assert tag1 == "v3.0.0"

        # Advance time by 3601 seconds (past 1 hour TTL)
        with patch("time.time", return_value=start_time + _CACHE_TTL_SECONDS + 1):
            tag2 = _run(fetch_latest_github_version(url=custom_url, use_cache=True))
        assert tag2 == "v3.1.0"

    assert mock_client.get.call_count == 2
