"""Shared SQLite connection helpers."""

from __future__ import annotations

import functools
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


def get_read_connection(
    db_path: Path,
) -> sqlite3.Connection:
    """Open an existing SQLite database in read-only mode."""
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

    database_uri = f"{resolved_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(
        database_uri,
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def with_sqlite_retry(
    fn: Callable | None = None,
    *,
    max_retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
) -> Callable:
    """Decorator that retries SQLite operations when a sqlite3.OperationalError occurs."""
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


@contextmanager
def managed_connection(db_path: str | os.PathLike) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for SQLite connections that guarantees conn.close() on exit,
    preventing unclosed connection handle leaks (Issue #1707).
    """
    conn = sqlite3.connect(db_path, timeout=15.0, check_same_thread=False)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
