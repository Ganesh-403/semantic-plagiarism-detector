import functools
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Callable

logger = logging.getLogger(__name__)

class ConcurrencyTimeoutError(Exception):
    """Raised when the FAISS lock cannot be acquired within the timeout threshold."""
    pass

class FAISSLock:
    """
    A robust, multi-process safe file-locking mechanism to protect the FAISS index
    from race conditions during concurrent document uploads or deletions.

    In a Streamlit environment, multiple sessions may attempt to write to the SQLite
    database and rebuild the FAISS index simultaneously. If two threads call save_index
    simultaneously, the .index file will corrupt.
    """

    def __init__(self, lock_file: str = "faiss_rebuild.lock", timeout: int = None):
        if timeout is None:
            from src.core.app_config import get_lock_timeout
            timeout = get_lock_timeout()

        self.lock_file = lock_file
        self.timeout = timeout

    def _is_stale(self) -> bool:
        """
        Checks if an existing lock file is stale (older than the timeout threshold).
        This protects against application crashes that leave phantom locks behind.
        """
        try:
            if not os.path.exists(self.lock_file):
                return False
            mtime = os.path.getmtime(self.lock_file)
            age = time.time() - mtime
            return age > self.timeout
        except OSError:
            # If we can't read the mtime, assume it's not stale to be safe
            return False

    def _clear_stale_lock(self):
        """Attempts to aggressively clear a lock file if it is deemed stale."""
        try:
            logger.warning(f"Detected stale FAISS lock: {self.lock_file}. Attempting aggressive clear.")
            os.remove(self.lock_file)
        except OSError as e:
            logger.error(f"Failed to clear stale FAISS lock: {e}")

    def acquire(self):
        """Attempts to acquire the atomic file lock."""
        start_time = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL ensures atomic creation. If file exists, raises FileExistsError.
                # This is process-safe and thread-safe at the OS level.
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, b"locked")
                os.close(fd)
                logger.debug(f"Acquired FAISS index lock: {self.lock_file}")
                return
            except FileExistsError:
                if self._is_stale():
                    self._clear_stale_lock()
                    continue  # Retry acquisition immediately

                if time.time() - start_time >= self.timeout:
                    logger.error(f"Timeout ({self.timeout}s) waiting for FAISS lock.")
                    raise ConcurrencyTimeoutError("Failed to acquire FAISS lock.")
                time.sleep(0.1)  # Spin wait

    def release(self):
        """Releases the atomic file lock."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                logger.debug(f"Released FAISS index lock: {self.lock_file}")
        except OSError as e:
            logger.warning(f"Failed to release FAISS lock gracefully: {e}")

@contextmanager
def faiss_write_lock(lock_path: str = "corpus.index.lock", timeout: int = None):
    """
    Context manager for safely locking FAISS I/O operations.

    Usage:
        with faiss_write_lock():
            build_index()
            save_index()
    """
    lock = FAISSLock(lock_file=lock_path, timeout=timeout)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

def with_sqlite_retry(
    fn: Callable | None = None,
    *,
    max_retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
) -> Callable:
    """
    Decorator that retries SQLite operations when a sqlite3.OperationalError occurs
    due to a locked or busy database ("database is locked" / "database is busy").

    Applies exponential backoff on subsequent retry attempts. Usable both as a
    bare decorator (``@with_sqlite_retry``) and with parameters
    (``@with_sqlite_retry(max_retries=5)``).

    Args:
        fn: Function being decorated when used as ``@with_sqlite_retry``.
        max_retries: Maximum number of retry attempts (default: 3).
        delay: Initial delay in seconds before the first retry (default: 0.1).
        backoff: Multiplier for exponential backoff (default: 2.0).

    Returns:
        Callable: Wrapped function with SQLite lock retry logic.
    """
    if fn is not None and callable(fn):
        return _make_sqlite_retry_wrapper(fn, max_retries=3, delay=0.1, backoff=2.0)

    def decorator(func: Callable) -> Callable:
        return _make_sqlite_retry_wrapper(
            func,
            max_retries=max_retries,
            delay=delay,
            backoff=backoff,
        )

    return decorator


def _make_sqlite_retry_wrapper(
    func: Callable,
    max_retries: int,
    delay: float,
    backoff: float,
) -> Callable:
    """Build a retry wrapper for the given function with exponential backoff."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        current_delay = delay
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                is_locked_err = "locked" in err_msg or "busy" in err_msg
                if is_locked_err and attempt < max_retries:
                    func_name = getattr(func, "__name__", str(func))
                    logger.warning(
                        f"SQLite database locked/busy in '{func_name}' "
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in "
                        f"{current_delay:.2f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                else:
                    raise

    return wrapper
