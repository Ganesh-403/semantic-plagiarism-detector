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
tests/utils/test_os_compat.py
-----------------------------
Comprehensive unit tests for the OS compatibility and asyncio patching module.

Verifies that platform detection works correctly, Windows-specific patches
are applied safely, and Unix systems are left untouched.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.utils.os_compat import (
    apply_asyncio_patches,
    get_os_platform,
    reset_patches_state,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure patch state is reset before and after every test."""
    reset_patches_state()
    yield
    reset_patches_state()


class TestGetOsPlatform:
    """Test suite for OS platform detection logic."""

    @patch("platform.system", return_value="Windows")
    def test_detects_windows(self, mock_system):
        """Verify Windows is correctly identified."""
        assert get_os_platform() == "windows"

    @patch("platform.system", return_value="Darwin")
    def test_detects_macos(self, mock_system):
        """Verify macOS (Darwin) is correctly identified."""
        assert get_os_platform() == "macos"

    @patch("platform.system", return_value="Linux")
    def test_detects_linux(self, mock_system):
        """Verify Linux is correctly identified."""
        assert get_os_platform() == "linux"

    @patch("platform.system", return_value="FreeBSD")
    def test_detects_unknown(self, mock_system):
        """Verify unrecognized systems return 'unknown'."""
        assert get_os_platform() == "unknown"

    @patch("platform.system", return_value="")
    def test_handles_empty_string(self, mock_system):
        """Verify empty platform strings are handled gracefully."""
        assert get_os_platform() == "unknown"


class TestApplyAsyncioPatches:
    """Test suite for the asyncio event loop policy patching."""

    @patch("src.utils.os_compat.get_os_platform", return_value="windows")
    @patch("asyncio.get_event_loop_policy")
    @patch("asyncio.set_event_loop_policy")
    def test_applies_selector_policy_on_windows(
        self, mock_set_policy, mock_get_policy, mock_os
    ):
        """Verify WindowsSelectorEventLoopPolicy is set on Windows."""
        # Mock the current policy as the default Proactor policy
        mock_get_policy.return_value = asyncio.WindowsProactorEventLoopPolicy()

        result = apply_asyncio_patches()

        assert result is True
        mock_set_policy.assert_called_once()

        # Verify the exact policy class passed to set_event_loop_policy
        call_args = mock_set_policy.call_args[0]
        assert isinstance(call_args[0], asyncio.WindowsSelectorEventLoopPolicy)

    @patch("src.utils.os_compat.get_os_platform", return_value="windows")
    @patch("asyncio.get_event_loop_policy")
    @patch("asyncio.set_event_loop_policy")
    def test_skips_if_selector_already_active_on_windows(
        self, mock_set_policy, mock_get_policy, mock_os
    ):
        """Verify no reassignment occurs if Selector policy is already active."""
        mock_get_policy.return_value = asyncio.WindowsSelectorEventLoopPolicy()

        result = apply_asyncio_patches()

        # Should return False because no change was made
        assert result is False
        mock_set_policy.assert_not_called()

    @patch("src.utils.os_compat.get_os_platform", return_value="linux")
    @patch("asyncio.set_event_loop_policy")
    def test_does_not_patch_on_linux(self, mock_set_policy, mock_os):
        """Verify no patches are applied on Linux systems."""
        result = apply_asyncio_patches()

        assert result is False
        mock_set_policy.assert_not_called()

    @patch("src.utils.os_compat.get_os_platform", return_value="macos")
    @patch("asyncio.set_event_loop_policy")
    def test_does_not_patch_on_macos(self, mock_set_policy, mock_os):
        """Verify no patches are applied on macOS systems."""
        result = apply_asyncio_patches()

        assert result is False
        mock_set_policy.assert_not_called()

    @patch("src.utils.os_compat.get_os_platform", return_value="windows")
    @patch("asyncio.get_event_loop_policy")
    @patch("asyncio.set_event_loop_policy")
    def test_prevents_duplicate_application(
        self, mock_set_policy, mock_get_policy, mock_os
    ):
        """Verify the function is idempotent and doesn't run twice."""
        mock_get_policy.return_value = asyncio.WindowsProactorEventLoopPolicy()

        # First call should apply the patch
        apply_asyncio_patches()
        assert mock_set_policy.call_count == 1

        # Second call should be skipped due to internal state flag
        result = apply_asyncio_patches()
        assert result is False
        assert mock_set_policy.call_count == 1  # Still 1

    @patch("src.utils.os_compat.get_os_platform", return_value="windows")
    @patch(
        "asyncio.get_event_loop_policy", side_effect=RuntimeError("Event loop error")
    )
    def test_handles_asyncio_exceptions_gracefully(
        self, mock_get_policy, mock_os, caplog
    ):
        """Verify exceptions during policy retrieval are caught and logged."""
        import logging

        with caplog.at_level(logging.ERROR):
            result = apply_asyncio_patches()

        assert result is False
        assert any(
            "Failed to apply Windows asyncio" in record.message
            for record in caplog.records
        )
