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

import logging
import os
import random
import time
from contextlib import contextmanager

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
            return age > max(self.timeout * 2, 5.0)
        except OSError:
            # If we can't read the mtime, assume it's not stale to be safe
            return False

    def _clear_stale_lock(self):
        """Attempts to aggressively clear a lock file if it is deemed stale."""
        try:
            logger.warning(
                f"Detected stale FAISS lock: {self.lock_file}. Attempting aggressive clear."
            )
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
                time.sleep(
                    0.1 + random.uniform(0, 0.05)
                )  # Spin wait with randomized jitter

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


# with_sqlite_retry now lives in src/db/common.py — re-exported here so
# existing callers (`from src.core.concurrency import with_sqlite_retry`
# and `from src.core import with_sqlite_retry`) keep working without a
# second, drifting copy of the same retry logic. A lazy re-export avoids a
# circular import when src.db is imported first (src.db -> src.core ->
# src.db.common).
def __getattr__(name):
    if name == "with_sqlite_retry":
        from src.db.common import with_sqlite_retry

        return with_sqlite_retry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
