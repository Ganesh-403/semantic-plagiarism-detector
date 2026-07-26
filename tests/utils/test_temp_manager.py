"""
tests/utils/test_temp_manager.py
--------------------------------
Unit tests for automatic temporary file and folder cleanup via atexit.
"""

import os

from src.utils.temp_manager import (cleanup_registered_temp_paths,
                                    create_managed_temp_file,
                                    register_temp_path)


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