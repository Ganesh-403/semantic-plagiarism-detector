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

import pytest

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

_VERSION_MOD_PATH = pathlib.Path(__file__).parent.parent.parent / "src" / "version.py"
_version_spec = importlib.util.spec_from_file_location("src.version", _VERSION_MOD_PATH)
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
_parse_semver_tuple = _vc_mod._parse_semver_tuple


def _reload_version_check_with_env(monkeypatch, owner=None, repo=None):
    """Import a fresh copy of version_check.py with GITHUB_OWNER/GITHUB_REPO
    applied from the environment (Issue #3964).

    GITHUB_OWNER, GITHUB_REPO, and GITHUB_RELEASES_URL are module-level
    constants resolved once at import time, so exercising the env-var
    override means importing the module again after setting the env var --
    reusing the already-loaded ``_vc_mod`` above would just show the values
    it was first imported with. A distinct module name keeps this out of
    ``sys.modules`` under the name every other test in this file relies on.
    """
    if owner is not None:
        monkeypatch.setenv("GITHUB_OWNER", owner)
    else:
        monkeypatch.delenv("GITHUB_OWNER", raising=False)
    if repo is not None:
        monkeypatch.setenv("GITHUB_REPO", repo)
    else:
        monkeypatch.delenv("GITHUB_REPO", raising=False)

    spec = importlib.util.spec_from_file_location(
        "src.utils.version_check_env_override_test", _MOD_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestGithubRepoCoordinatesEnvOverride:
    """Issue #3964: GITHUB_OWNER / GITHUB_REPO are configurable via env vars."""

    def test_defaults_when_env_unset(self, monkeypatch) -> None:
        mod = _reload_version_check_with_env(monkeypatch)
        assert mod.GITHUB_OWNER == "Ganesh-403"
        assert mod.GITHUB_REPO == "semantic-plagiarism-detector"

    def test_owner_overridden_by_env(self, monkeypatch) -> None:
        mod = _reload_version_check_with_env(monkeypatch, owner="my-fork-org")
        assert mod.GITHUB_OWNER == "my-fork-org"
        assert mod.GITHUB_REPO == "semantic-plagiarism-detector"

    def test_repo_overridden_by_env(self, monkeypatch) -> None:
        mod = _reload_version_check_with_env(monkeypatch, repo="my-mirror")
        assert mod.GITHUB_OWNER == "Ganesh-403"
        assert mod.GITHUB_REPO == "my-mirror"

    def test_both_overridden_reflected_in_releases_url(self, monkeypatch) -> None:
        mod = _reload_version_check_with_env(
            monkeypatch, owner="my-fork-org", repo="my-mirror"
        )
        assert mod.GITHUB_OWNER == "my-fork-org"
        assert mod.GITHUB_REPO == "my-mirror"
        assert mod.GITHUB_RELEASES_URL == (
            "https://api.github.com/repos/my-fork-org/my-mirror/releases/latest"
        )


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

    def test_v1_0_0_prefix(self) -> None:
        assert _normalise_tag("v1.0.0") == "1.0.0"

    def test_no_prefix_1_0_0(self) -> None:
        assert _normalise_tag("1.0.0") == "1.0.0"

    def test_uppercase_v_prefix_unchanged(self) -> None:
        # NOTE: current implementation is case-sensitive (str.lstrip("v")),
        # so uppercase "V" is NOT stripped — documenting actual behavior.
        assert _normalise_tag("V1.0.0") == "V1.0.0"

    def test_release_prefix_unchanged(self) -> None:
        # NOTE: current implementation does not handle a "release-" prefix —
        # documenting actual behavior, not the full acceptance criteria in
        # issue #3969 (see PR description for details).
        assert _normalise_tag("release-1.0.0") == "release-1.0.0"


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

    def test_numeric_sort_not_lexicographic(self) -> None:
        assert is_update_available("1.2.0", "1.10.0") is True

    def test_numeric_sort_reverse_direction(self) -> None:
        assert is_update_available("1.10.0", "1.2.0") is False

    def test_major_version_bump_two_vs_one_nine_nine(self) -> None:
        assert is_update_available("1.9.9", "2.0.0") is True

    def test_no_update_when_remote_older_major(self) -> None:
        assert is_update_available("2.0.0", "1.9.9") is False

    def test_equal_versions_no_v_prefix(self) -> None:
        assert is_update_available("1.0.0", "1.0.0") is False

    def test_equal_versions_mixed_v_prefix(self) -> None:
        assert is_update_available("1.0.0", "v1.0.0") is False


# ── is_update_available fallback (Issue #3965) ─────────────────────────────────
# These force the ``from packaging.version import Version`` import to fail
# (``packaging`` uninstalled, or any other ImportError) by registering
# ``None`` for the submodule in ``sys.modules`` -- Python raises
# ``ImportError`` for any import of a module that maps to ``None`` there --
# so ``is_update_available`` falls through to ``_parse_semver_tuple``.


def _without_packaging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    monkeypatch.setitem(sys.modules, "packaging", None)


class TestIsUpdateAvailableFallback:
    def test_local_newer_is_not_reported_as_update(self, monkeypatch) -> None:
        """Issue #3965: remote older than local must not report an update.

        The old fallback (``remote != local``) reported True here, which is
        backwards -- the local version is the newer one.
        """
        _without_packaging(monkeypatch)
        assert is_update_available("1.1.0", "1.0.0") is False

    def test_remote_newer_patch_reports_update(self, monkeypatch) -> None:
        _without_packaging(monkeypatch)
        assert is_update_available("1.0.0", "1.0.1") is True

    def test_remote_newer_minor_reports_update(self, monkeypatch) -> None:
        _without_packaging(monkeypatch)
        assert is_update_available("1.0.5", "1.1.0") is True

    def test_remote_newer_major_reports_update(self, monkeypatch) -> None:
        _without_packaging(monkeypatch)
        assert is_update_available("1.9.9", "2.0.0") is True

    def test_equal_versions_no_update(self, monkeypatch) -> None:
        _without_packaging(monkeypatch)
        assert is_update_available("1.2.3", "v1.2.3") is False

    def test_numeric_sort_not_lexicographic(self, monkeypatch) -> None:
        """"1.10.0" must sort after "1.2.0" numerically, not lexicographically."""
        _without_packaging(monkeypatch)
        assert is_update_available("1.2.0", "1.10.0") is True
        assert is_update_available("1.10.0", "1.2.0") is False

    def test_malformed_tag_does_not_raise(self, monkeypatch) -> None:
        _without_packaging(monkeypatch)
        assert is_update_available("1.0.0", "not-a-version") is False
        assert is_update_available("not-a-version", "1.0.0") is True


class TestParseSemverTuple:
    def test_basic_triple(self) -> None:
        assert _parse_semver_tuple("1.2.3") == (1, 2, 3)

    def test_missing_patch_defaults_to_zero(self) -> None:
        assert _parse_semver_tuple("1.2") == (1, 2, 0)

    def test_missing_minor_and_patch_default_to_zero(self) -> None:
        assert _parse_semver_tuple("1") == (1, 0, 0)

    def test_pre_release_suffix_is_ignored(self) -> None:
        assert _parse_semver_tuple("1.2.3-rc1") == (1, 2, 3)

    def test_build_metadata_suffix_is_ignored(self) -> None:
        assert _parse_semver_tuple("1.2.3+build.5") == (1, 2, 3)

    def test_non_numeric_component_defaults_to_zero(self) -> None:
        assert _parse_semver_tuple("abc") == (0, 0, 0)

    def test_extra_components_beyond_patch_are_ignored(self) -> None:
        assert _parse_semver_tuple("1.2.3.4") == (1, 2, 3)


# ── fetch_latest_github_version ────────────────────────────────────────────────


class TestFetchLatestGithubVersion:
    """Tests for the async fetch function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset version cache before and after each test."""
        if hasattr(_vc_mod, "clear_version_cache"):
            _vc_mod.clear_version_cache()
        yield
        if hasattr(_vc_mod, "clear_version_cache"):
            _vc_mod.clear_version_cache()

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

    def test_fetch_latest_github_version_graceful_failures(self) -> None:
        """Verify that 403, 404, and 500 error codes safely return None without throwing exceptions."""
        import httpx

        for status_code in [403, 404, 500]:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"Error {status_code}",
                request=MagicMock(spec=httpx.Request),
                response=mock_response
            )

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)

            with patch.object(_vc_mod.httpx, "AsyncClient", return_value=mock_client):
                tag = self._run(fetch_latest_github_version())

            assert tag is None


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
    