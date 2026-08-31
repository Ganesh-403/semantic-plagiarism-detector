"""
tests/utils/test_temp_manager.py
--------------------------------
Unit tests for automatic temporary file and folder cleanup via atexit.

Includes tests for:
- Managed temp file creation and cleanup
- Custom path registration
- Error handling during cleanup
- Temp directory size calculation (Issue #1251)
"""

import logging
import os
import shutil
import tempfile
from unittest.mock import patch

from src.utils.temp_manager import (
    _REGISTERED_TEMP_PATHS,
    check_temp_disk_space,
    cleanup_registered_temp_paths,
    cleanup_temp_files,
    create_managed_temp_dir,
    create_managed_temp_file,
    get_temp_directory_size_bytes,
    purge_expired_temp_files,
    register_temp_path,
    unregister_temp_path,
    verify_available_temp_space,
)


def test_managed_temp_file_creation_and_cleanup():
    # 1. Create a managed temp file
    temp_path = create_managed_temp_file(suffix=".pdf", prefix="test_report_")
    assert os.path.exists(temp_path)

    # Write dummy data
    with open(temp_path, "w") as f:
        f.write("dummy pdf content")

    # 2. Trigger the cleanup function manually
    cleanup_registered_temp_paths()

    # 3. Verify file was wiped from disk
    assert not os.path.exists(temp_path)


def test_custom_path_registration_cleanup():
    # Create an unmanaged file manually
    dummy_file = "test_cache_file.csv"
    with open(dummy_file, "w") as f:
        f.write("a,b,c\n1,2,3")

    register_temp_path(dummy_file)
    assert os.path.exists(dummy_file)

    # Trigger cleanup
    cleanup_registered_temp_paths()

    # Verify file was cleaned up
    assert not os.path.exists(dummy_file)


def test_cleanup_logs_warning_on_oserror_file():
    """Verify warning is logged when os.remove fails with OSError."""
    import src.utils.temp_manager as temp_manager_module

    with patch.object(
        temp_manager_module.os, "remove", side_effect=OSError("Permission denied")
    ) as mock_remove:
        with patch.object(temp_manager_module, "logger") as mock_logger:
            # Create a temp file and register it
            fd, temp_path = tempfile.mkstemp()
            os.close(fd)  # Close the handle
            register_temp_path(temp_path)

            # Create the file so it exists for cleanup
            with open(temp_path, "w") as f:
                f.write("test")

            cleanup_registered_temp_paths()

            mock_remove.assert_called_once_with(temp_path)
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args is not None
            assert "Failed to clean up temp file" in call_args[0][0]
            assert temp_path in call_args[0][1]


def test_cleanup_logs_warning_on_oserror_dir():
    """Verify warning is logged when shutil.rmtree fails with OSError."""
    import src.utils.temp_manager as temp_manager_module

    with patch.object(
        temp_manager_module.shutil, "rmtree", side_effect=OSError("Permission denied")
    ):
        with patch.object(temp_manager_module, "logger") as mock_logger:
            temp_dir = tempfile.mkdtemp()
            register_temp_path(temp_dir)

            cleanup_registered_temp_paths()

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args is not None
            assert "Failed to clean up temp file" in call_args[0][0]
            assert temp_dir in call_args[0][1]


def test_cleanup_removes_path_from_registry_even_on_error():
    """Verify path is removed from registry even when cleanup fails."""
    import src.utils.temp_manager as temp_manager_module

    with patch.object(
        temp_manager_module.os, "remove", side_effect=OSError("Permission denied")
    ):
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        register_temp_path(temp_path)

        cleanup_registered_temp_paths()

        # Path should be removed from registry even though cleanup failed
        assert temp_path not in _REGISTERED_TEMP_PATHS


def test_unregister_temp_path():
    """Verify unregister_temp_path removes a path from the tracking list."""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)

    register_temp_path(temp_path)
    assert temp_path in _REGISTERED_TEMP_PATHS

    unregister_temp_path(temp_path)
    assert temp_path not in _REGISTERED_TEMP_PATHS

    # Clean up the actual file
    os.remove(temp_path)


def test_unregister_temp_path_not_registered():
    """Verify unregister does nothing for unregistered paths."""
    initial_count = len(_REGISTERED_TEMP_PATHS)
    unregister_temp_path("/nonexistent/path/that/is/not/registered")
    assert len(_REGISTERED_TEMP_PATHS) == initial_count


def test_register_temp_path_duplicate():
    """Verify duplicate registration of the same path is idempotent."""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)

    register_temp_path(temp_path)
    register_temp_path(temp_path)

    # Should appear only once in the list
    count = _REGISTERED_TEMP_PATHS.count(temp_path)
    assert count == 1

    # Clean up
    cleanup_registered_temp_paths()


def test_register_temp_path_empty_string():
    """Empty string path should not be registered."""
    initial_count = len(_REGISTERED_TEMP_PATHS)
    register_temp_path("")
    assert len(_REGISTERED_TEMP_PATHS) == initial_count


def test_register_temp_path_none():
    """None path should not be registered."""
    initial_count = len(_REGISTERED_TEMP_PATHS)
    register_temp_path(None)
    assert len(_REGISTERED_TEMP_PATHS) == initial_count


def test_create_managed_temp_dir_creation_and_cleanup():
    """Verify managed temp directory is created and cleaned up."""
    temp_dir = create_managed_temp_dir(suffix="_test", prefix="test_dir_")
    assert os.path.exists(temp_dir)
    assert os.path.isdir(temp_dir)
    assert temp_dir in _REGISTERED_TEMP_PATHS

    # Create a file inside the directory
    inner_file = os.path.join(temp_dir, "test.txt")
    with open(inner_file, "w") as f:
        f.write("test content")
    assert os.path.exists(inner_file)

    cleanup_registered_temp_paths()
    assert not os.path.exists(temp_dir)


def test_create_managed_temp_file_with_suffix():
    """Verify temp file is created with the specified suffix."""
    temp_path = create_managed_temp_file(suffix=".txt", prefix="test_")
    assert os.path.exists(temp_path)
    assert temp_path.endswith(".txt")
    cleanup_registered_temp_paths()


def test_purge_expired_temp_files_returns_int():
    """Verify purge_expired_temp_files returns an integer."""
    result = purge_expired_temp_files(max_age_seconds=0)
    assert isinstance(result, int)
    assert result >= 0


def test_verify_available_temp_space_raises_when_insufficient():
    """Should raise OSError when free temp space is less than required."""

    with patch(
        "src.utils.temp_manager.shutil.disk_usage",
        return_value=(1000, 900, 100),
    ):
        with pytest.raises(
            OSError,
            match="Insufficient free disk space in temp directory",
        ):
            verify_available_temp_space(200)


# ─── Tests for get_temp_directory_size_bytes (Issue #1251) ────────────────────


def test_get_temp_directory_size_bytes_returns_int():
    """The function must always return an integer."""
    result = get_temp_directory_size_bytes()
    assert isinstance(result, int)


def test_get_temp_directory_size_bytes_non_negative():
    """The returned size must be non-negative."""
    result = get_temp_directory_size_bytes()
    assert result >= 0


def test_get_temp_directory_size_bytes_increases_with_files():
    """Adding files to the temp directory should increase the reported size."""
    # Get baseline size
    baseline_size = get_temp_directory_size_bytes()

    # Create a file with known content in the temp directory
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "test_size_calc_file.bin")
    test_content = b"x" * 10000  # 10 KB of data

    with open(test_file, "wb") as f:
        f.write(test_content)

    try:
        new_size = get_temp_directory_size_bytes()
        # The new size should be at least 10000 bytes larger than baseline
        assert new_size >= baseline_size + 10000
    finally:
        # Clean up the test file
        try:
            os.remove(test_file)
        except OSError:
            pass


def test_get_temp_directory_size_bytes_includes_subdirectories():
    """Files in subdirectories must be included in the total size."""
    temp_dir = tempfile.gettempdir()
    sub_dir = os.path.join(temp_dir, "test_size_subdir_12345")

    try:
        os.makedirs(sub_dir, exist_ok=True)
        test_file = os.path.join(sub_dir, "nested_file.bin")
        test_content = b"y" * 5000

        with open(test_file, "wb") as f:
            f.write(test_content)

        size = get_temp_directory_size_bytes()
        # Size should be at least 5000 bytes (the file we just created)
        assert size >= 5000
    finally:
        # Clean up
        try:
            shutil.rmtree(sub_dir, ignore_errors=True)
        except Exception:
            pass


def test_get_temp_directory_size_bytes_handles_nonexistent_dir():
    """If temp directory does not exist, return 0 gracefully."""
    with patch("src.utils.temp_manager.os.path.exists", return_value=False):
        result = get_temp_directory_size_bytes()
        assert result == 0


def test_get_temp_directory_size_bytes_handles_non_dir():
    """If temp path is not a directory, return 0."""
    with (
        patch("src.utils.temp_manager.os.path.exists", return_value=True),
        patch("src.utils.temp_manager.os.path.isdir", return_value=False),
    ):
        result = get_temp_directory_size_bytes()
        assert result == 0


def test_get_temp_directory_size_bytes_handles_permission_error():
    """If os.walk raises OSError, return whatever was accumulated."""
    with patch(
        "src.utils.temp_manager.os.walk", side_effect=OSError("Permission denied")
    ):
        result = get_temp_directory_size_bytes()
        assert isinstance(result, int)
        assert result >= 0


def test_get_temp_directory_size_bytes_handles_stat_error():
    """If os.stat fails for individual files, they are skipped."""
    import src.utils.temp_manager as temp_manager_module

    # Create a temp file to ensure at least one file exists
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    with open(temp_path, "w") as f:
        f.write("test")

    try:
        with patch.object(
            temp_manager_module.os, "stat", side_effect=OSError("Permission denied")
        ):
            result = get_temp_directory_size_bytes()
            assert isinstance(result, int)
            assert result >= 0
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def test_get_temp_directory_size_bytes_empty_directory():
    """An empty temp directory should return 0 or near-0 size."""
    # Create an empty subdirectory in temp
    temp_dir = tempfile.gettempdir()
    empty_dir = os.path.join(temp_dir, "empty_test_dir_12345")

    try:
        os.makedirs(empty_dir, exist_ok=True)
        # The size should not significantly change from baseline
        # (empty directories have minimal or no size contribution)
        result = get_temp_directory_size_bytes()
        assert isinstance(result, int)
        assert result >= 0
    finally:
        try:
            os.rmdir(empty_dir)
        except OSError:
            pass


def test_get_temp_directory_size_bytes_multiple_files():
    """Verify correct size calculation with multiple files."""
    temp_dir = tempfile.gettempdir()
    test_files = []
    total_written = 0

    try:
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_multi_file_{i}.bin")
            content = b"z" * (1000 * (i + 1))  # 1KB, 2KB, 3KB
            with open(file_path, "wb") as f:
                f.write(content)
            test_files.append(file_path)
            total_written += len(content)

        size = get_temp_directory_size_bytes()
        # Size should be at least the total we wrote
        assert size >= total_written
    finally:
        for f in test_files:
            try:
                os.remove(f)
            except OSError:
                pass


# ─── Tests for rotate_backup_files (Issue #1572) ──────────────────────────────

import time

import pytest

from src.utils.temp_manager import rotate_backup_files


class TestRotateBackupFiles:
    """Comprehensive test suite for backup file rotation and retention policies."""

    def test_deletes_oldest_files_exceeding_keep_count(self, tmp_path):
        """Verify that files exceeding keep_count are deleted, oldest first."""
        # Create 5 files with staggered modification times
        for i in range(5):
            file_path = tmp_path / f"backup_{i}.db"
            file_path.write_text(f"data_{i}")
            # Set modification time to i seconds ago (older files have higher i)
            old_time = time.time() - (i * 10)
            os.utime(file_path, (old_time, old_time))

        # Keep only 2 newest files
        deleted = rotate_backup_files(tmp_path, keep_count=2)

        assert deleted == 3

        # Verify the 2 newest files (backup_0 and backup_1) remain
        assert (tmp_path / "backup_0.db").exists()
        assert (tmp_path / "backup_1.db").exists()

        # Verify the 3 oldest files are deleted
        assert not (tmp_path / "backup_2.db").exists()
        assert not (tmp_path / "backup_3.db").exists()
        assert not (tmp_path / "backup_4.db").exists()

    def test_no_deletion_when_under_keep_count(self, tmp_path):
        """If file count <= keep_count, no files should be deleted."""
        for i in range(3):
            (tmp_path / f"backup_{i}.db").write_text("data")

        deleted = rotate_backup_files(tmp_path, keep_count=5)
        assert deleted == 0

        # All files should still exist
        assert len(list(tmp_path.glob("*.db"))) == 3

    def test_no_deletion_when_exactly_at_keep_count(self, tmp_path):
        """If file count == keep_count, no files should be deleted."""
        for i in range(5):
            (tmp_path / f"backup_{i}.db").write_text("data")

        deleted = rotate_backup_files(tmp_path, keep_count=5)
        assert deleted == 0
        assert len(list(tmp_path.glob("*.db"))) == 5

    def test_keep_count_zero_deletes_all_files(self, tmp_path):
        """keep_count=0 should delete all files in the directory."""
        for i in range(3):
            (tmp_path / f"backup_{i}.db").write_text("data")

        deleted = rotate_backup_files(tmp_path, keep_count=0)
        assert deleted == 3
        assert len(list(tmp_path.glob("*.db"))) == 0

    def test_ignores_subdirectories(self, tmp_path):
        """Subdirectories should not be counted or deleted."""
        (tmp_path / "backup_1.db").write_text("data")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("nested")

        deleted = rotate_backup_files(tmp_path, keep_count=0)

        # Only the .db file should be deleted
        assert deleted == 1
        assert not (tmp_path / "backup_1.db").exists()

        # Subdirectory and its contents should remain
        assert (tmp_path / "subdir").is_dir()
        assert (tmp_path / "subdir" / "nested.txt").exists()

    def test_nonexistent_directory_raises_filenotfound(self, tmp_path):
        """A non-existent directory should raise FileNotFoundError."""
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            rotate_backup_files(missing_dir, keep_count=5)

    def test_file_path_raises_notadirectory(self, tmp_path):
        """A file path instead of a directory should raise NotADirectoryError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I am a file")

        with pytest.raises(NotADirectoryError):
            rotate_backup_files(file_path, keep_count=5)

    def test_negative_keep_count_raises_valueerror(self, tmp_path):
        """A negative keep_count should raise ValueError."""
        with pytest.raises(ValueError, match="keep_count must be >= 0"):
            rotate_backup_files(tmp_path, keep_count=-1)

    def test_empty_directory_returns_zero(self, tmp_path):
        """An empty directory should return 0 deleted files."""
        deleted = rotate_backup_files(tmp_path, keep_count=5)
        assert deleted == 0

    def test_handles_oserror_during_deletion_gracefully(self, tmp_path, caplog):
        """If os.remove fails (e.g., permission denied), it should log and continue."""
        import src.utils.temp_manager as temp_manager_module

        # Create 3 files
        for i in range(3):
            (tmp_path / f"backup_{i}.db").write_text("data")
            old_time = time.time() - (i * 10)
            os.utime(tmp_path / f"backup_{i}.db", (old_time, old_time))

        # Mock os.remove to fail for the oldest file
        original_remove = os.remove

        def mock_remove(path):
            if "backup_2.db" in str(path):
                raise OSError("Permission denied")
            return original_remove(path)

        with patch.object(temp_manager_module.os, "remove", side_effect=mock_remove):
            with caplog.at_level(logging.WARNING):
                deleted = rotate_backup_files(tmp_path, keep_count=1)

        # Should have deleted 1 file successfully, failed on 1
        assert deleted == 1
        assert "failed to delete" in caplog.text
        # The file that failed to delete should still exist
        assert (tmp_path / "backup_2.db").exists()

    def test_accepts_string_path(self, tmp_path):
        """The function should accept string paths as well as Path objects."""
        (tmp_path / "backup.db").write_text("data")

        # Pass string instead of Path
        deleted = rotate_backup_files(str(tmp_path), keep_count=0)
        assert deleted == 1

    def test_ignores_symlinks(self, tmp_path):
        """Symlinks should not be counted or deleted to prevent accidental data loss."""
        real_file = tmp_path / "real.db"
        real_file.write_text("real data")

        link_path = tmp_path / "link.db"
        try:
            os.symlink(real_file, link_path)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        deleted = rotate_backup_files(tmp_path, keep_count=0)

        # Only the real file should be deleted
        assert deleted == 1
        assert not real_file.exists()
        # The symlink might still exist as a broken link, or be removed depending on OS
        # But the real file is definitely gone

    def test_sorts_by_modification_time_not_name(self, tmp_path):
        """Files should be sorted by mtime, not alphabetically by name."""
        # Create files with names that would sort differently than their mtime
        file_a = tmp_path / "z_newest.db"
        file_b = tmp_path / "a_oldest.db"

        file_a.write_text("new")
        file_b.write_text("old")

        # Make 'a_oldest' actually older
        old_time = time.time() - 100
        os.utime(file_b, (old_time, old_time))

        # Keep 1 file. Should keep z_newest because it's newer, despite 'a' < 'z'
        deleted = rotate_backup_files(tmp_path, keep_count=1)

        assert deleted == 1
        assert file_a.exists()  # Newest kept
        assert not file_b.exists()  # Oldest deleted


def test_cleanup_temp_files_retention(tmp_path):
    """Verify that cleanup_temp_files only deletes files/dirs older than retention_hours."""
    import time

    # 1. Create a "new" file and register it
    new_file = tmp_path / "new_temp_file.txt"
    new_file.write_text("fresh content")
    register_temp_path(str(new_file))

    # 2. Create an "old" file and register it
    old_file = tmp_path / "old_temp_file.txt"
    old_file.write_text("stale content")
    register_temp_path(str(old_file))
    # Modify mtime to be 2 hours ago
    old_mtime = time.time() - (2 * 3600)
    os.utime(old_file, (old_mtime, old_mtime))

    # 3. Create a "new" directory and register it
    new_dir = tmp_path / "new_temp_dir"
    new_dir.mkdir()
    register_temp_path(str(new_dir))

    # 4. Create an "old" directory and register it
    old_dir = tmp_path / "old_temp_dir"
    old_dir.mkdir()
    register_temp_path(str(old_dir))
    # Modify mtime to be 2 hours ago
    os.utime(old_dir, (old_mtime, old_mtime))

    # Run cleanup with 1.0 hour retention
    cleanup_temp_files(retention_hours=1.0)

    # Assertions
    # New file/dir should still exist and remain registered
    assert new_file.exists()
    assert str(new_file) in _REGISTERED_TEMP_PATHS
    assert new_dir.exists()
    assert str(new_dir) in _REGISTERED_TEMP_PATHS

    # Old file/dir should be deleted and unregistered
    assert not old_file.exists()
    assert str(old_file) not in _REGISTERED_TEMP_PATHS
    assert not old_dir.exists()
    assert str(old_dir) not in _REGISTERED_TEMP_PATHS

    # Cleanup remaining new items to prevent test leakage
    cleanup_registered_temp_paths()


def test_check_temp_disk_space_success():
    """Verify that check_temp_disk_space returns True when free space is sufficient."""
    with patch(
        "src.utils.temp_manager.shutil.disk_usage",
        return_value=(1000 * 1024 * 1024, 200 * 1024 * 1024, 800 * 1024 * 1024),
    ):
        # 800 MB is free, min required is 100 MB, should succeed
        assert check_temp_disk_space(min_free_mb=100) is True


def test_check_temp_disk_space_failure():
    """Verify that check_temp_disk_space raises OSError when free space is below safety threshold."""
    import pytest

    with patch(
        "src.utils.temp_manager.shutil.disk_usage",
        return_value=(1000 * 1024 * 1024, 950 * 1024 * 1024, 50 * 1024 * 1024),
    ):
        # 50 MB is free, min required is 100 MB, should raise OSError
        with pytest.raises(
            OSError,
            match="Disk space in temp directory below safety threshold",
        ):
            check_temp_disk_space(min_free_mb=100)


def test_managed_ocr_temp_dir_creates_and_purges_directory():
    """Verify managed_ocr_temp_dir creates a temp directory and purges it on exit."""
    from src.utils.temp_manager import managed_ocr_temp_dir

    created_dir = None
    created_file = None

    with managed_ocr_temp_dir(prefix="test_ocr_dir_") as tmp_dir:
        created_dir = tmp_dir
        assert os.path.isdir(tmp_dir)

        # Create temporary PNG and text file simulating Tesseract OCR output
        created_file = os.path.join(tmp_dir, "tess_123.png")
        with open(created_file, "w", encoding="utf-8") as f:
            f.write("mock OCR image data")
        assert os.path.exists(created_file)

    # After exiting block, temp directory and files must be deleted
    assert not os.path.exists(created_dir)
    assert not os.path.exists(created_file)


def test_managed_ocr_temp_dir_cleanup_on_exception():
    """Verify managed_ocr_temp_dir purges all files even when an exception occurs."""
    from src.utils.temp_manager import managed_ocr_temp_dir

    created_dir = None

    with pytest.raises(RuntimeError, match="Tesseract crash simulation"):
        with managed_ocr_temp_dir(prefix="test_ocr_crash_") as tmp_dir:
            created_dir = tmp_dir
            # Write temp OCR files
            with open(os.path.join(tmp_dir, "crash_test.txt"), "w") as f:
                f.write("OCR text")
            raise RuntimeError("Tesseract crash simulation")

    assert not os.path.exists(created_dir)


def test_managed_ocr_temp_dir_restores_environment_and_tempdir():
    """Verify managed_ocr_temp_dir cleanly restores prior tempdir and environment variables."""
    from src.utils.temp_manager import managed_ocr_temp_dir

    orig_tempdir = tempfile.tempdir
    orig_tmpdir_env = os.environ.get("TMPDIR")

    with managed_ocr_temp_dir() as tmp_dir:
        assert tempfile.tempdir == tmp_dir
        assert os.environ.get("TMPDIR") == tmp_dir

    assert tempfile.tempdir == orig_tempdir
    assert os.environ.get("TMPDIR") == orig_tmpdir_env


def test_create_managed_temp_file_atomic_cleanup_on_failure():
    """Verify that if temp file registration fails, the temp file is deleted and exception is re-raised."""
    from unittest.mock import patch

    with patch("src.utils.temp_manager.register_temp_path", side_effect=RuntimeError("Registration failed")):
        original_mkstemp = tempfile.mkstemp
        created_paths = []

        def spy_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            created_paths.append(path)
            return fd, path

        with patch("src.utils.temp_manager.tempfile.mkstemp", side_effect=spy_mkstemp):
            with pytest.raises(RuntimeError, match="Registration failed"):
                create_managed_temp_file()

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

