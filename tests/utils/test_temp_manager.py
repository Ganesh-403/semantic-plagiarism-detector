"""
tests/utils/test_temp_manager.py
--------------------------------
Unit tests for automatic temporary file and folder cleanup via atexit.
"""

import os
import tempfile
from unittest.mock import patch

from src.utils.temp_manager import (
    cleanup_registered_temp_paths,
    create_managed_temp_file,
    register_temp_path,
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
    from src.utils.temp_manager import _REGISTERED_TEMP_PATHS

    assert temp_path not in _REGISTERED_TEMP_PATHS
