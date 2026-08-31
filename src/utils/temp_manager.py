"""
src/utils/temp_manager.py
-------------------------
Utility for tracking and automatically cleaning up temporary files and directories
on application exit using Python's atexit module and tempfile utilities.

Provides functions to register temporary paths for automatic cleanup,
create managed temp files/directories, purge expired files safely (with symlink protection),
calculate total disk space consumed by temporary work files, and rotate backup files.

Recent Additions (Issue #3179):
- Hardened `purge_expired_temp_files` against symlink traversal attacks by passing
  follow_symlinks=False explicitly and safely unlinking symbolic links.
"""

import atexit
from contextlib import contextmanager
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Generator

# Global list of registered temporary paths to clean up
_REGISTERED_TEMP_PATHS: list[str] = []
_lock = threading.Lock()

logger = logging.getLogger(__name__)


def register_temp_path(path: str) -> str:
    """Registers a file or directory path for automatic cleanup on process exit."""
    if path:
        with _lock:
            if path not in _REGISTERED_TEMP_PATHS:
                _REGISTERED_TEMP_PATHS.append(path)
    return path


def unregister_temp_path(path: str) -> None:
    """Removes a path from the cleanup tracking list if manually deleted earlier."""
    with _lock:
        if path in _REGISTERED_TEMP_PATHS:
            _REGISTERED_TEMP_PATHS.remove(path)


def cleanup_registered_temp_paths() -> None:
    """
    Cleans up all registered temporary files and directories.
    Registered as an atexit hook.
    """
    with _lock:
        paths = list(_REGISTERED_TEMP_PATHS)

    for path in paths:
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except OSError as exc:
            logger.warning("Failed to clean up temp file %s: %s", path, exc)
        finally:
            with _lock:
                if path in _REGISTERED_TEMP_PATHS:
                    _REGISTERED_TEMP_PATHS.remove(path)


# Register the exit handler automatically on module import
atexit.register(cleanup_registered_temp_paths)


def get_default_temp_file_retention_hours() -> float:
    """Read TEMP_FILE_RETENTION_HOURS from environment variable, falling back to 1.0 if not set or invalid (Issue #3182)."""
    env_val = os.getenv("TEMP_FILE_RETENTION_HOURS")
    if env_val is not None:
        try:
            val = float(env_val)
            if val > 0:
                return val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid TEMP_FILE_RETENTION_HOURS value '%s'. Falling back to default 1.0 hours.",
                env_val,
            )
    return 1.0


def cleanup_temp_files(retention_hours: Optional[float] = None) -> None:
    """
    Cleans up registered temporary files and directories that are older than the specified retention hours.

    Args:
        retention_hours: Max age in hours. Defaults to TEMP_FILE_RETENTION_HOURS env var (or 1.0 if not set).
    """
    if retention_hours is None:
        retention_hours = get_default_temp_file_retention_hours()

    now = time.time()
    retention_seconds = retention_hours * 3600.0

    with _lock:
        paths = list(_REGISTERED_TEMP_PATHS)

    for path in paths:
        try:
            if os.path.exists(path) or os.path.islink(path):
                mtime = os.path.getmtime(path)
                if now - mtime > retention_seconds:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    with _lock:
                        if path in _REGISTERED_TEMP_PATHS:
                            _REGISTERED_TEMP_PATHS.remove(path)
            else:
                with _lock:
                    if path in _REGISTERED_TEMP_PATHS:
                        _REGISTERED_TEMP_PATHS.remove(path)
        except OSError as exc:
            logger.warning("Failed to clean up temp path %s: %s", path, exc)


def create_managed_temp_file(
    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> str:
    """Creates a temporary file on disk and registers it for automatic deletion on exit."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    try:
        os.close(fd)
        register_temp_path(temp_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return temp_path


@contextmanager
def managed_temp_file(
    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> Generator[str, None, None]:
    """Context manager for a temp file that is cleaned up immediately on exit.

    Unlike ``create_managed_temp_file``, which relies on the ``atexit``
    handler to delete the file at process shutdown, this yields the path
    inside a ``with`` block and unregisters + unlinks the file as soon as
    the block exits (even if an exception was raised), so short-lived
    tasks don't have to wait for process exit to free disk space.
    """
    temp_path = create_managed_temp_file(suffix=suffix, prefix=prefix)
    try:
        yield temp_path
    finally:
        unregister_temp_path(temp_path)
        try:
            if os.path.exists(temp_path) or os.path.islink(temp_path):
                os.remove(temp_path)
        except OSError as exc:
            logger.warning("Failed to clean up temp file %s: %s", temp_path, exc)


def create_managed_temp_dir(    suffix: Optional[str] = None, prefix: Optional[str] = None
) -> str:
    """Creates a temporary directory on disk and registers it for automatic deletion on exit."""
    temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    register_temp_path(temp_dir)
    return temp_dir


def purge_expired_temp_files(max_age_seconds: int = 7200) -> int:
    """
    Scans the system temp directory and removes files whose last modification
    time is older than max_age_seconds (default: 2 hours). Hardened against 
    symlink traversal attacks (Issue #3179).

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
                    # Hardening: Check and safely handle symlinks without following them
                    if entry.is_symlink():
                        entry.unlink()
                        purged_count += 1
                        continue

                    if not entry.is_file(follow_symlinks=False):
                        continue

                    # Hardening: Pass follow_symlinks=False explicitly to stat()
                    file_stat = entry.stat(follow_symlinks=False)
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
    """Calculate total disk space occupied by the active temp directory recursively."""
    temp_dir = tempfile.gettempdir()
    total_size = 0

    if not os.path.exists(temp_dir) or not os.path.isdir(temp_dir):
        return 0

    try:
        for dirpath, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    file_stat = os.stat(file_path, follow_symlinks=True)
                    total_size += file_stat.st_size
                except (OSError, ValueError) as exc:
                    logger.debug("get_temp_directory_size_bytes: skipping %s: %s", file_path, exc)
                    continue
    except OSError as exc:
        logger.warning("get_temp_directory_size_bytes: failed to walk temp directory %s: %s", temp_dir, exc)

    return total_size


def verify_available_temp_space(required_bytes: int) -> bool:
    """Verify that the system temporary directory has enough free disk space."""
    temp_dir = tempfile.gettempdir()
    _, _, free = shutil.disk_usage(temp_dir)

    if free < required_bytes:
        raise OSError("Insufficient free disk space in temp directory")

    return True


def check_temp_disk_space(min_free_mb: int = 100) -> bool:
    """Verify that available disk space in the system temporary directory exceeds the minimum threshold."""
    temp_dir = tempfile.gettempdir()
    _, _, free = shutil.disk_usage(temp_dir)

    if free < min_free_mb * 1024 * 1024:
        raise OSError("Disk space in temp directory below safety threshold")

    return True


def rotate_backup_files(backup_dir: Path, keep_count: int = 5) -> int:
    """Enforce retention policies on backup directories by keeping only the N most recent files."""
    if keep_count < 0:
        raise ValueError(f"keep_count must be >= 0, received {keep_count}")

    resolved_dir = Path(backup_dir).expanduser().resolve()

    if not resolved_dir.exists():
        raise FileNotFoundError(f"Backup directory not found: {resolved_dir}")

    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {resolved_dir}")

    backup_files = []
    try:
        with os.scandir(resolved_dir) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    try:
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                        backup_files.append((entry.path, mtime))
                    except OSError as exc:
                        logger.warning("rotate_backup_files: failed to stat file %s: %s", entry.path, exc)
    except OSError as exc:
        logger.error("rotate_backup_files: failed to scan directory %s: %s", resolved_dir, exc)
        return 0

    if len(backup_files) <= keep_count:
        return 0

    backup_files.sort(key=lambda x: x[1], reverse=True)
    files_to_delete = backup_files[keep_count:]

    deleted_count = 0
    freed_bytes = 0

    for file_path, mtime in files_to_delete:
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            freed_bytes += file_size
        except OSError as exc:
            logger.warning("rotate_backup_files: failed to delete %s: %s", file_path, exc)

    return deleted_count


@contextmanager
def managed_ocr_temp_dir(prefix: str = "tesseract_ocr_") -> Generator[str, None, None]:
    """Context manager for OCR processing that creates a dedicated temporary directory."""
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir:
        old_temp = tempfile.tempdir
        tempfile.tempdir = tmp_dir
        old_env = {k: os.environ.get(k) for k in ("TMPDIR", "TEMP", "TMP")}
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TEMP"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        try:
            yield tmp_dir
        finally:
            tempfile.tempdir = old_temp
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
