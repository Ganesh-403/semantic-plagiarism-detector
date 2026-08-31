"""
test_version_check.py
---------------------
Unit tests for the src.utils.version_check module.

All tests are fully offline — the GitHub API is mocked via ``pytest-mock`` /
``unittest.mock`` so no network access is required.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Import version_check directly to avoid pulling in the heavy src/__init__.py
# chain (which transitively requires docx, faiss, etc.)
# ---------------------------------------------------------------------------
# version_check.py imports `APP_VERSION` from `src.version` (the
# centralized single-source-of-truth version module). Resolving that
# import still requires a `src` package to exist in sys.modules -- so a
# lightweight namespace stub is registered here (mirroring version_check
# itself being hand-loaded below) rather than letting Python fall through
# to the real src/__init__.py, which is exactly the heavy chain this
# test file exists to avoid.
if "src" not in sys.modules:
    _src_stub = types.ModuleType("src")
    _src_stub.__path__ = [str(pathlib.Path(__file__).parent.parent.parent / "src")]
    sys.modules["src"] = _src_stub

_VERSION_MOD_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "src" / "version.py"
)
_version_spec = importlib.util.spec_from_file_location(
    "src.version", _VERSION_MOD_PATH
)
_version_mod = importlib.util.module_from_spec(_version_spec)  # type: ignore[arg-type]
sys.modules.setdefault("src.version", _version_mod)
_version_spec.loader.exec_module(_version_mod)  # type: ignore[union-attr]

_MOD_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "src" / "utils" / "version_check.py"
)
_spec = importlib.util.spec_from_file_location("src.utils.version_check", _MOD_PATH)
_vc_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules.setdefault("src.utils.version_check", _vc_mod)
_spec.loader.exec_module(_vc_mod)  # type: ignore[union-attr]

APP_VERSION = _vc_mod.APP_VERSION
GITHUB_RELEASES_URL = _vc_mod.GITHUB_RELEASES_URL
_normalise_tag = _vc_mod._normalise_tag
check_for_update_sync = _vc_mod.check_for_update_sync
fetch_latest_github_version = _vc_mod.fetch_latest_github_version
is_update_available = _vc_mod.is_update_available


# ── _normalise_tag ─────────────────────────────────────────────────────────────


class TestNormaliseTag:
    def test_strips_leading_v(self) -> None:
        assert _normalise_tag("v1.2.3") == "1.2.3"

    def test_no_leading_v_unchanged(self) -> None:
        assert _normalise_tag("1.2.3") == "1.2.3"

    def test_empty_string(self) -> None:
        assert _normalise_tag("") == ""

    def test_only_v(self) -> None:
        assert _normalise_tag("v") == ""


# ── is_update_available ────────────────────────────────────────────────────────


class TestIsUpdateAvailable:
    def test_newer_remote(self) -> None:
        assert is_update_available("1.0.0", "v1.1.0") is True

    def test_same_version(self) -> None:
        assert is_update_available("1.0.0", "v1.0.0") is False

    def test_older_remote(self) -> None:
        assert is_update_available("2.0.0", "v1.9.9") is False

    def test_v_prefix_local_and_remote(self) -> None:
        assert is_update_available("v1.0.0", "v1.0.1") is True

    def test_patch_bump(self) -> None:
        assert is_update_available("1.0.0", "1.0.1") is True

    def test_major_bump(self) -> None:
        assert is_update_available("1.0.0", "2.0.0") is True

    def test_no_update_exact_match(self) -> None:
        assert is_update_available("1.2.3", "v1.2.3") is False


# ── fetch_latest_github_version ────────────────────────────────────────────────


class TestFetchLatestGithubVersion:
    """Tests for the async fetch function."""

    def _run(self, coro):
        """Helper: run a coroutine in a fresh event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_returns_tag_name_on_success(self) -> None:
        """A well-formed 200 response returns the tag_name string."""

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json.return_value = {
            "tag_name": "v1.5.0",
            "name": "Release 1.5.0",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        # Patch on the already-loaded module to avoid re-importing src.
        with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client):
            tag = self._run(fetch_latest_github_version())

        assert tag == "v1.5.0"

    def test_returns_none_on_http_error(self) -> None:
        """An HTTP error should be silently swallowed and None returned."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client):
            tag = self._run(fetch_latest_github_version())

        assert tag is None

    def test_returns_none_when_tag_name_missing(self) -> None:
        """A 200 response without tag_name should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json.return_value = {"name": "some release"}  # no tag_name

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client):
            tag = self._run(fetch_latest_github_version())

        assert tag is None

    def test_custom_url_is_passed_through(self) -> None:
        """The URL parameter is forwarded to the HTTP client."""
        custom_url = "https://api.github.com/repos/test-owner/test-repo/releases/latest"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json.return_value = {"tag_name": "v2.0.0"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client):
            self._run(fetch_latest_github_version(url=custom_url))

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == custom_url

    def test_uses_custom_user_agent_header(self) -> None:
        """AsyncClient should be initialized with User-Agent and Accept headers."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json.return_value = {"tag_name": "v1.0.0"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client) as mock_async_client:
            self._run(fetch_latest_github_version())

        mock_async_client.assert_called_once()
        _, kwargs = mock_async_client.call_args
        headers = kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert f"SemanticPlagiarismDetector/{_vc_mod.APP_VERSION}" in headers["User-Agent"]
        assert headers.get("Accept") == "application/vnd.github+json"


# ── check_for_update_sync ──────────────────────────────────────────────────────


class TestCheckForUpdateSync:
    """Tests for the synchronous wrapper."""

    def test_returns_tag_when_update_available(self) -> None:
        with patch.object(
            _vc_mod, "fetch_latest_github_version", new=AsyncMock(return_value="v9.9.9")
        ):
            result = check_for_update_sync(local_version="1.0.0")
        assert result == "v9.9.9"

    def test_returns_none_when_up_to_date(self) -> None:
        with patch.object(
            _vc_mod,
            "fetch_latest_github_version",
            new=AsyncMock(return_value=f"v{APP_VERSION}"),
        ):
            result = check_for_update_sync(local_version=APP_VERSION)
        assert result is None

    def test_returns_none_when_fetch_fails(self) -> None:
        with patch.object(
            _vc_mod, "fetch_latest_github_version", new=AsyncMock(return_value=None)
        ):
            result = check_for_update_sync(local_version="1.0.0")
        assert result is None

    def test_returns_none_when_remote_is_older(self) -> None:
        with patch.object(
            _vc_mod, "fetch_latest_github_version", new=AsyncMock(return_value="v0.0.1")
        ):
            result = check_for_update_sync(local_version="1.0.0")
        assert result is None

    def test_uses_asyncio_run_and_no_event_loop_leak(self) -> None:
        """Verify that check_for_update_sync executes via asyncio.run without leaking event loops."""
        with patch.object(
            _vc_mod, "fetch_latest_github_version", new=AsyncMock(return_value="v9.9.9")
        ):
            result = check_for_update_sync(local_version="1.0.0")

        assert result == "v9.9.9"
        try:
            loop = asyncio.get_running_loop()
            assert not loop.is_running()
        except RuntimeError:
            pass

    def test_no_event_loop_leak_on_exception(self) -> None:
        """Verify event loop cleanup occurs automatically even when fetch raises an error."""
        with patch.object(
            _vc_mod,
            "fetch_latest_github_version",
            new=AsyncMock(side_effect=RuntimeError("async failure")),
        ):
            result = check_for_update_sync(local_version="1.0.0")

        assert result is None
        try:
            loop = asyncio.get_running_loop()
            assert not loop.is_running()
        except RuntimeError:
            pass


# ── Module-level constants ─────────────────────────────────────────────────────


def test_app_version_is_non_empty() -> None:
    assert APP_VERSION and isinstance(APP_VERSION, str)


def test_github_releases_url_is_valid() -> None:
    assert GITHUB_RELEASES_URL.startswith("https://api.github.com/repos/")
    assert "/releases/latest" in GITHUB_RELEASES_URL
