"""Shared SQLite connection helpers."""

from __future__ import annotations

import functools
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


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

    Applies exponential backoff on subsequent retry attempts.

    Args:
        fn (Callable, optional): Function being decorated when used as @with_sqlite_retry.
        max_retries (int): Maximum number of retry attempts (default: 3).
        delay (float): Initial delay in seconds before the first retry (default: 0.1).
        backoff (float): Multiplier for exponential backoff (default: 2.0).

    Returns:
        Callable: Wrapped function with SQLite lock retry logic.
    """
    if fn is not None and callable(fn):
        return _make_wrapper(fn, max_retries=3, delay=0.1, backoff=2.0)

    def decorator(func: Callable) -> Callable:
        return _make_wrapper(func, max_retries=max_retries, delay=delay, backoff=backoff)

    return decorator


def _make_wrapper(func: Callable, max_retries: int, delay: float, backoff: float) -> Callable:
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
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in {current_delay:.2f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                else:
                    raise

    return wrapper


def get_read_connection(
    db_path: Path,
) -> sqlite3.Connection:
    """Open an existing SQLite database in read-only mode.

    The database path is converted to a platform-safe ``file:`` URI
    and opened with ``mode=ro``. SQLite therefore refuses
    ``INSERT``, ``UPDATE``, ``DELETE``, schema changes, and attempts
    to create a database that does not already exist.

    Args:
        db_path: Path to an existing SQLite database file.

    Returns:
        A read-only SQLite connection configured with
        :class:`sqlite3.Row` as its row factory.

    Raises:
        TypeError: If ``db_path`` is not a :class:`pathlib.Path`.
        FileNotFoundError: If the database file does not exist.
        IsADirectoryError: If ``db_path`` points to a directory.
        sqlite3.Error: If SQLite cannot open the file.
    """
    if not isinstance(db_path, Path):
        raise TypeError("db_path must be a pathlib.Path.")

    resolved_path = db_path.expanduser().resolve(strict=False)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"SQLite database does not exist: {resolved_path}"
        )
    if not resolved_path.is_file():
        raise IsADirectoryError(
            f"SQLite database path is not a file: {resolved_path}"
        )

    # Path.as_uri() handles Windows drive letters, spaces, Unicode,
    # and other characters that require URI escaping.
    database_uri = f"{resolved_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(
        database_uri,
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
