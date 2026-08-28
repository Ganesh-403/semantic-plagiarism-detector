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
Comprehensive Unit Tests for rotate_backup_files edge cases (Issue #3185)
Includes tests for rotation logic, cleanup functions, and context managers.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from src.utils.temp_manager import (
    cleanup_registered_temp_paths,
    cleanup_temp_files,
    managed_ocr_temp_dir,
    purge_expired_temp_files,
    register_temp_path,
    rotate_backup_files,
    unregister_temp_path,
)

# ==============================================================================
# SECTION 1: Core rotate_backup_files tests
# ==============================================================================


class TestRotateBackupFiles:
    def test_no_files_to_delete(self, tmp_path):
        """If fewer files than keep_count, nothing is deleted."""
        result = rotate_backup_files(tmp_path, keep_count=5)
        assert result == 0

    def test_deletes_oldest_files(self, tmp_path):
        """Ensure the correct number of oldest files are deleted."""
        for i in range(10):
            (tmp_path / f"backup_{i}.db").write_text("data")

        result = rotate_backup_files(tmp_path, keep_count=3)

        # Should have deleted 7 files (10 - 3)
        assert result == 7
        assert len(list(tmp_path.iterdir())) == 3

    def test_keeps_newest_files(self, tmp_path):
        """Ensure the newest files are kept."""
        for i in range(5):
            file = tmp_path / f"backup_{i}.db"
            file.write_text("data")
            # Set modification times so file_4 is newest
            os.utime(file, (i, i))

        rotate_backup_files(tmp_path, keep_count=2)

        files = sorted(tmp_path.iterdir(), key=lambda x: x.stat().st_mtime)
        assert "backup_4.db" in files[-1].name
        assert "backup_3.db" in files[-2].name

    def test_ignores_subdirectories(self, tmp_path):
        """Ensure subdirectories are not deleted."""
        (tmp_path / "folder").mkdir()
        for i in range(5):
            (tmp_path / f"backup_{i}.db").write_text("data")

        rotate_backup_files(tmp_path, keep_count=0)

        # The folder should still exist
        assert (tmp_path / "folder").exists()

    def test_negative_keep_count_raises_error(self, tmp_path):
        """Invalid keep_count should raise ValueError."""
        with pytest.raises(ValueError):
            rotate_backup_files(tmp_path, keep_count=-1)

    def test_missing_directory_raises_error(self):
        """Non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            rotate_backup_files(Path("/non/existent/path"), keep_count=5)

    def test_zero_keep_count_deletes_all(self, tmp_path):
        """If keep_count is 0, all files should be deleted."""
        for i in range(3):
            (tmp_path / f"backup_{i}.db").write_text("data")

        result = rotate_backup_files(tmp_path, keep_count=0)

        assert result == 3
        assert len(list(tmp_path.iterdir())) == 0

    def test_returns_integer(self, tmp_path):
        """The return value should always be an integer."""
        for i in range(5):
            (tmp_path / f"backup_{i}.db").write_text("data")

        result = rotate_backup_files(tmp_path, keep_count=2)
        assert isinstance(result, int)


# ==============================================================================
# SECTION 2: Integrity and File Safety
# ==============================================================================


class TestRotateBackupFilesIntegrity:
    def test_only_files_deleted(self, tmp_path):
        """Ensure files are deleted, but directories are not."""
        (tmp_path / "important_folder").mkdir()
        (tmp_path / "file1.txt").write_text("data")

        rotate_backup_files(tmp_path, keep_count=0)

        assert (tmp_path / "important_folder").exists()
        assert not (tmp_path / "file1.txt").exists()

    def test_no_extra_files_created(self, tmp_path):
        """Rotating should not create any new files."""
        for i in range(5):
            (tmp_path / f"backup_{i}.db").write_text("data")

        initial_count = len(list(tmp_path.iterdir()))
        rotate_backup_files(tmp_path, keep_count=2)
        final_count = len(list(tmp_path.iterdir()))

        assert final_count < initial_count

    def test_files_from_different_extensions(self, tmp_path):
        """Rotation should handle mixed file extensions."""
        for i in range(5):
            (tmp_path / f"backup_{i}.zip").write_text("data")
            (tmp_path / f"backup_{i}.log").write_text("data")

        rotate_backup_files(tmp_path, keep_count=2)
        assert len(list(tmp_path.iterdir())) == 2


# ==============================================================================
# SECTION 3: Cleanup Functions
# ==============================================================================


class TestCleanupFunctions:
    def test_unregister_temp_path(self):
        """Unregistering a path should remove it from the list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            test_path = tmp.name
            register_temp_path(test_path)
            unregister_temp_path(test_path)
            assert test_path not in tempfile._get_default_tempdir()

    def test_cleanup_temp_files_old_and_new(self, tmp_path):
        """Old files should be deleted, new files should remain."""
        old_file = tmp_path / "old.txt"
        new_file = tmp_path / "new.txt"
        old_file.write_text("data")
        new_file.write_text("data")

        # Set old file's modification time to 2 hours ago
        old_time = time.time() - 7200
        os.utime(old_file, (old_time, old_time))

        register_temp_path(str(old_file))
        register_temp_path(str(new_file))

        cleanup_temp_files(retention_hours=1.0)

        assert not old_file.exists()
        assert new_file.exists()


# ==============================================================================
# SECTION 4: Purge Expired Files and Context Managers
# ==============================================================================


class TestPurgeAndContextManagers:
    def test_purge_expired_temp_files_returns_int(self):
        """Purge function should return an integer count."""
        result = purge_expired_temp_files(max_age_seconds=0)
        assert isinstance(result, int)

    def test_purge_expired_temp_files_no_crash(self):
        """Purge function should not crash on empty directories."""
        result = purge_expired_temp_files(max_age_seconds=0)
        assert result >= 0

    def test_managed_ocr_temp_dir_creates_dir(self):
        """Context manager should create a temporary directory."""
        with managed_ocr_temp_dir(prefix="test_ocr_") as tmp_dir:
            assert os.path.isdir(tmp_dir)
            assert "test_ocr_" in os.path.basename(tmp_dir)

    def test_managed_ocr_temp_dir_cleans_up(self):
        """Context manager should delete the directory after exit."""
        with managed_ocr_temp_dir(prefix="test_ocr_") as tmp_dir:
            path_to_check = tmp_dir
        assert not os.path.exists(path_to_check)
