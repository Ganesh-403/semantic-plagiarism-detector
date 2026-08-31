"""
test_temp_manager_retention_issue_3182.py
-------------------------------------------
Unit test suite for Issue #3182:
Validates that cleanup_temp_files uses TEMP_FILE_RETENTION_HOURS environment variable
as the default retention period, falling back to 1.0 hours if unconfigured or invalid.
"""

import os
import time
from unittest.mock import patch

from src.utils.temp_manager import (
    cleanup_temp_files,
    get_default_temp_file_retention_hours,
    register_temp_path,
    unregister_temp_path,
)


def test_default_retention_hours_fallback():
    """Verify fallback to 1.0 hours when TEMP_FILE_RETENTION_HOURS is absent or invalid."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_default_temp_file_retention_hours() == 1.0

    with patch.dict(os.environ, {"TEMP_FILE_RETENTION_HOURS": "invalid"}):
        assert get_default_temp_file_retention_hours() == 1.0


def test_configurable_retention_hours_from_env(tmp_path):
    """Verify TEMP_FILE_RETENTION_HOURS env var is respected by cleanup_temp_files."""
    # Set retention to 0.001 hours (3.6 seconds)
    retention_hours_str = "0.001"
    with patch.dict(os.environ, {"TEMP_FILE_RETENTION_HOURS": retention_hours_str}):
        assert get_default_temp_file_retention_hours() == float(retention_hours_str)

        # Create a temp file and manipulate its mtime to be older than 5 seconds
        test_file = tmp_path / "old_temp_file.txt"
        test_file.write_text("sample content")
        file_path_str = str(test_file)
        register_temp_path(file_path_str)

        # Set mtime to 10 seconds ago
        old_time = time.time() - 10
        os.utime(file_path_str, (old_time, old_time))

        # Cleanup with default retention (from env)
        cleanup_temp_files()

        # Verify old temp file was removed
        assert not test_file.exists()
        unregister_temp_path(file_path_str)


def test_explicit_retention_hours_override(tmp_path):
    """Verify explicit retention_hours argument overrides environment variable setting."""
    with patch.dict(os.environ, {"TEMP_FILE_RETENTION_HOURS": "10.0"}):
        test_file = tmp_path / "recent_temp_file.txt"
        test_file.write_text("sample content")
        file_path_str = str(test_file)
        register_temp_path(file_path_str)

        # Set mtime to 5 seconds ago
        old_time = time.time() - 5
        os.utime(file_path_str, (old_time, old_time))

        # Cleanup with explicit retention of 0.0001 hours (~0.36 seconds)
        cleanup_temp_files(retention_hours=0.0001)

        # Verify file was cleaned up despite env var being 10.0
        assert not test_file.exists()
        unregister_temp_path(file_path_str)
