"""
src/utils/temp_manager.py
-------------------------
Utility for tracking and automatically cleaning up temporary files and directories
on application exit using Python's atexit module and tempfile utilities.

Provides functions to register temporary paths for automatic cleanup,
create managed temp files/directories, purge expired files, calculate
total disk space consumed by temporary work files, and rotate backup files.

Recent Additions (Issue #1251):
- Added `get_temp_directory_size_bytes` function that calculates total disk
  space occupied by all files inside the active temp directory recursively.

Recent Additions (Issue #1572):
- Added `rotate_backup_files` function that enforces retention policies on
  backup directories by keeping only the N most recent backup files.
"""

import atexit
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# Global list of registered temporary paths to clean up
_REGISTERED_TEMP_PATHS: List[str] = []

logger = logging.getLogger(__name__)


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
        except OSError as exc:
            logger.warning("Failed to clean up temp file %s: %s", path, exc)
        finally:
            if path in _REGISTERED_TEMP_PATHS:
                _REGISTERED_TEMP_PATHS.remove(path)


# Register the exit handler automatically on module import
atexit.register(cleanup_registered_temp_paths)


def cleanup_temp_files(retention_hours: float = 1.0) -> None:
    """
    Cleans up registered temporary files and directories that are older than the specified retention hours.

    Args:
        retention_hours: The number of hours to retain files. Files/directories older than this
                         (based on modification time st_mtime) will be deleted. Defaults to 1.0.
    """
    now = time.time()
    retention_seconds = retention_hours * 3600.0

    for path in list(_REGISTERED_TEMP_PATHS):
        try:
            if os.path.exists(path) or os.path.islink(path):
                mtime = os.path.getmtime(path)
                if now - mtime > retention_seconds:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    if path in _REGISTERED_TEMP_PATHS:
                        _REGISTERED_TEMP_PATHS.remove(path)
            else:
                if path in _REGISTERED_TEMP_PATHS:
                    _REGISTERED_TEMP_PATHS.remove(path)
        except OSError as exc:
            logger.warning("Failed to clean up temp path %s: %s", path, exc)


def create_managed_temp_file(
    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> str:
    """
    Creates a temporary file on disk and registers it for automatic deletion on exit.

    Returns:
        str: Absolute path to the created temporary file.
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(
        fd
    )  # Close file descriptor so other components can open/write to it freely
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


def purge_expired_temp_files(max_age_seconds: int = 7200) -> int:
    """
    Scans the system temp directory and removes files whose last modification
    time is older than max_age_seconds (default: 2 hours). Intended to run on
    application startup or on a periodic schedule to prevent temp file buildup.

    Returns:
        int: Number of files purged.
    """
    temp_dir = tempfile.gettempdir()
    now = time.time()
    purged_count = 0
    freed_bytes = 0

    try:
        with os.scandir(temp_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    file_stat = entry.stat()
                    age_seconds = now - file_stat.st_mtime
                    if age_seconds > max_age_seconds:
                        file_size = file_stat.st_size
                        os.remove(entry.path)
                        purged_count += 1
                        freed_bytes += file_size
                except OSError as exc:
                    logger.warning("Failed to purge temp file %s: %s", entry.path, exc)
    except OSError as exc:
        logger.warning("Failed to scan temp directory %s: %s", temp_dir, exc)

    logger.info(
        "Temp file cleanup complete: purged %d file(s), freed %d byte(s).",
        purged_count,
        freed_bytes,
    )
    return purged_count


def get_temp_directory_size_bytes() -> int:
    """Calculate total disk space occupied by the active temp directory.

    Recursively walks through the system's active temporary directory
    (as returned by ``tempfile.gettempdir()``) and sums the byte sizes
    of all files found within it. This is useful for monitoring how
    much disk space is currently consumed by temporary work files,
    helping administrators identify potential disk space issues before
    they cause application failures.

    The function handles errors gracefully:
    - If the temp directory does not exist or is inaccessible, returns 0.
    - If individual files cannot be stat'd (e.g., permission denied),
      they are skipped and logged as warnings.
    - Symlinks are followed to count the target file size, but broken
      symlinks are skipped.

    Returns:
        Total size in bytes of all files inside the active temp directory.
        Returns 0 if the directory does not exist or is empty.

    Examples:
        >>> size = get_temp_directory_size_bytes()
        >>> print(f"Temp directory uses {size / (1024*1024):.1f} MB")
        Temp directory uses 45.3 MB
    """
    temp_dir = tempfile.gettempdir()
    total_size = 0

    # If the temp directory does not exist (very rare), return 0
    if not os.path.exists(temp_dir):
        logger.debug(
            "get_temp_directory_size_bytes: temp directory does not exist: %s",
            temp_dir,
        )
        return 0

    if not os.path.isdir(temp_dir):
        logger.warning(
            "get_temp_directory_size_bytes: temp path is not a directory: %s",
            temp_dir,
        )
        return 0

    try:
        # Use os.walk for recursive traversal of the entire temp directory tree.
        # This counts files in all subdirectories, not just the top level.
        for dirpath, dirnames, filenames in os.walk(temp_dir):
            # Process each file in the current directory
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                try:
                    # Use os.stat to get file size.
                    # follow_symlinks=True means broken symlinks will raise
                    # OSError and be caught below.
                    file_stat = os.stat(file_path, follow_symlinks=True)
                    total_size += file_stat.st_size

                except (OSError, ValueError) as exc:
                    # Skip files we cannot stat (permission denied,
                    # broken symlink, file deleted during walk, etc.)
                    logger.debug(
                        "get_temp_directory_size_bytes: skipping %s: %s",
                        file_path,
                        exc,
                    )
                    continue

    except OSError as exc:
        # If os.walk itself fails (e.g., permission denied on temp dir),
        # log the error and return whatever was accumulated so far.
        logger.warning(
            "get_temp_directory_size_bytes: failed to walk temp directory %s: %s",
            temp_dir,
            exc,
        )

    logger.debug(
        "get_temp_directory_size_bytes: total size is %d bytes (%.2f MB) in %s",
        total_size,
        total_size / (1024 * 1024),
        temp_dir,
    )

    return total_size

def verify_available_temp_space(required_bytes: int) -> bool:
    """
    Verify that the system temporary directory has enough free disk space.

    Args:
        required_bytes: Minimum number of free bytes required.

    Returns:
        True if sufficient free space is available.

    Raises:
        OSError: If the temporary directory does not have enough free space.
    """
    temp_dir = tempfile.gettempdir()
    _, _, free = shutil.disk_usage(temp_dir)

    if free < required_bytes:
        raise OSError("Insufficient free disk space in temp directory")

    return True

def rotate_backup_files(backup_dir: Path, keep_count: int = 5) -> int:
    """Enforce retention policies on backup directories by keeping only the N most recent files.

    Database backup files created in temp or backup directories accumulate over time
    and can exhaust disk space. This function scans the specified directory, sorts
    all files by their creation/modification time (newest first), and deletes older
    backup files that exceed the `keep_count` threshold.

    Only regular files are considered for deletion. Subdirectories and symlinks are
    ignored to prevent accidental data loss.

    Args:
        backup_dir: Path to the directory containing backup files. Must exist and
                    be a directory.
        keep_count: Number of most recent backup files to retain. Files beyond this
                    count (ordered by age) will be deleted. Must be >= 0.
                    Defaults to 5.

    Returns:
        The total number of backup files successfully deleted.

    Raises:
        FileNotFoundError: If the backup directory does not exist.
        NotADirectoryError: If the path exists but is not a directory.
        ValueError: If keep_count is negative.

    Examples:
        >>> from pathlib import Path
        >>> deleted = rotate_backup_files(Path("backups/"), keep_count=3)
        >>> print(f"Deleted {deleted} old backup files.")
        Deleted 7 old backup files.
    """
    # Validate inputs
    if keep_count < 0:
        raise ValueError(f"keep_count must be >= 0, received {keep_count}")

    resolved_dir = Path(backup_dir).expanduser().resolve()

    if not resolved_dir.exists():
        logger.error(
            "rotate_backup_files: backup directory does not exist: %s",
            resolved_dir,
        )
        raise FileNotFoundError(f"Backup directory not found: {resolved_dir}")

    if not resolved_dir.is_dir():
        logger.error(
            "rotate_backup_files: path is not a directory: %s",
            resolved_dir,
        )
        raise NotADirectoryError(f"Path is not a directory: {resolved_dir}")

    # Collect all regular files in the directory (non-recursive for safety)
    # Using os.scandir for efficient file system traversal
    backup_files = []
    try:
        with os.scandir(resolved_dir) as entries:
            for entry in entries:
                # Only consider regular files, skip directories and symlinks
                if entry.is_file(follow_symlinks=False):
                    try:
                        # Use modification time for sorting (most reliable cross-platform)
                        # st_mtime is preferred over st_ctime as ctime means "metadata change"
                        # on Unix and "creation time" on Windows, causing inconsistencies.
                        mtime = entry.stat().st_mtime
                        backup_files.append((entry.path, mtime))
                    except OSError as exc:
                        logger.warning(
                            "rotate_backup_files: failed to stat file %s: %s",
                            entry.path,
                            exc,
                        )
    except OSError as exc:
        logger.error(
            "rotate_backup_files: failed to scan directory %s: %s",
            resolved_dir,
            exc,
        )
        return 0

    # If we have fewer files than keep_count, nothing to delete
    if len(backup_files) <= keep_count:
        logger.debug(
            "rotate_backup_files: only %d file(s) found, keep_count=%d. No deletions needed.",
            len(backup_files),
            keep_count,
        )
        return 0

    # Sort files by modification time, newest first (descending order)
    backup_files.sort(key=lambda x: x[1], reverse=True)

    # Files to keep are the first `keep_count` entries
    files_to_delete = backup_files[keep_count:]
    
    deleted_count = 0
    freed_bytes = 0

    for file_path, mtime in files_to_delete:
        try:
            # Get file size before deletion for logging purposes
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            freed_bytes += file_size
            
            logger.info(
                "rotate_backup_files: deleted old backup %s (age: %.1f days, size: %.2f MB)",
                os.path.basename(file_path),
                (time.time() - mtime) / 86400,
                file_size / (1024 * 1024),
            )
        except OSError as exc:
            logger.warning(
                "rotate_backup_files: failed to delete %s: %s",
                file_path,
                exc,
            )

    logger.info(
        "rotate_backup_files: rotation complete for %s. Deleted %d file(s), "
        "freed %d byte(s). Retained %d file(s).",
        resolved_dir,
        deleted_count,
        freed_bytes,
        keep_count,
    )

    return deleted_count
