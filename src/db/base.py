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

"""src/db/base.py - Common BaseRepository interface for SQLite database access."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

from src.db.common import with_sqlite_retry
from src.db.connection import create_connection, get_connection

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception class for database access errors."""

    pass


class BaseRepository:
    """Encapsulates SQLite connection usage, transactions, retries, and query execution."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path: Path = Path(os.path.abspath(str(db_path)))

    @property
    def db_path(self) -> Path:
        """Return the current database file path as a Path object."""
        return self._db_path

    def configure_db_path(self, db_path: str | Path) -> None:
        """Update the database file path."""
        self._db_path = Path(os.path.abspath(str(db_path)))

    @contextmanager
    def connection(
        self, read_only: bool = False
    ) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a managed SQLite connection."""
        with get_connection(self._db_path, read_only=read_only) as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing transactional execution with automatic commit and rollback."""
        conn = create_connection(self._db_path)
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception as rollback_exc:
                logger.error(f"[BaseRepository] Rollback failed: {rollback_exc}")
            raise exc
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @with_sqlite_retry
    def execute(self, sql: str, params: tuple | dict = ()) -> Any:
        """Execute a single DML statement within a transaction."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.lastrowid

    @with_sqlite_retry
    def executemany(self, sql: str, seq_of_params: Iterable) -> int:
        """Execute a DML statement across a sequence of parameters."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, seq_of_params)
            return cursor.rowcount

    @with_sqlite_retry
    def fetch_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        """Fetch a single matching row."""
        with self.connection(read_only=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchone()

    @with_sqlite_retry
    def fetch_all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """Fetch all matching rows."""
        with self.connection(read_only=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database schema."""
        row = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists within a specific database table."""
        if not self.table_exists(table_name):
            return False
        with self.connection(read_only=True) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = [
                r["name"] if isinstance(r, sqlite3.Row) else r[1]
                for r in cursor.fetchall()
            ]
            return column_name in columns
