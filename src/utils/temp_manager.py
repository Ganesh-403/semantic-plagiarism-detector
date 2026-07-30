"""
src/utils/temp_manager.py
-------------------------
Utility for tracking and automatically cleaning up temporary files and directories
on application exit using Python's atexit module and tempfile utilities.
"""

import atexit
import os
import shutil
import tempfile
from typing import List, Optional

# Global list of registered temporary paths to clean up
_REGISTERED_TEMP_PATHS: List[str] = []


def register_temp_path(path: str) -> str:
    """Registers a file or directory path for automatic cleanup on process exit."""
    if path and path not in _REGISTERED_TEMP_PATHS:
        _REGISTERED_TEMP_PATHS.append(path)
    return path


def unregister_temp_path(path: str) -> None:
    """Removes a path from the cleanup tracking list if manually deleted earlier."""
    if path in _REGISTERED_TEMP_PATHS:
        _REGISTERED_TEMP_PATHS.remove(path)


def cleanup_registered_temp_paths() -> None:
    """
    Cleans up all registered temporary files and directories.
    Registered as an atexit hook.
    """
    for path in list(_REGISTERED_TEMP_PATHS):
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass  # Suppress errors during process teardown
        finally:
            if path in _REGISTERED_TEMP_PATHS:
                _REGISTERED_TEMP_PATHS.remove(path)


# Register the exit handler automatically on module import
atexit.register(cleanup_registered_temp_paths)


def create_managed_temp_file(
    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> str:
    """
    Creates a temporary file on disk and registers it for automatic deletion on exit.

    Returns:
        str: Absolute path to the created temporary file.
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close file descriptor so other components can open/write to it freely
    register_temp_path(temp_path)
    return temp_path


def create_managed_temp_dir(
    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> str:
    """
    Creates a temporary directory on disk and registers it for automatic deletion on exit.

    Returns:
        str: Absolute path to the created temporary directory.
    """
    temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    register_temp_path(temp_dir)
    return temp_dir

