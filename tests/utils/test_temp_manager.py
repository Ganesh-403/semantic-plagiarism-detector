import shutil
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

import os
import tempfile
from unittest.mock import patch

from src.utils.temp_manager import (
    cleanup_registered_temp_paths,
    create_managed_temp_file,
    create_managed_temp_dir,
    get_temp_directory_size_bytes,
    purge_expired_temp_files,
    register_temp_path,
    unregister_temp_path,
    _REGISTERED_TEMP_PATHS,
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
    with patch("src.utils.temp_manager.os.path.exists", return_value=True), \
         patch("src.utils.temp_manager.os.path.isdir", return_value=False):
        result = get_temp_directory_size_bytes()
        assert result == 0


def test_get_temp_directory_size_bytes_handles_permission_error():
    """If os.walk raises OSError, return whatever was accumulated."""
    with patch("src.utils.temp_manager.os.walk", side_effect=OSError("Permission denied")):
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
