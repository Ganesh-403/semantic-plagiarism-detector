"""
Unit tests for managed SQLite connections and cleanup (Issue #1707).
"""

import sqlite3
import unittest
from src.db.common import managed_connection
from src.db.auth import _connect as auth_connect


class TestConnectionCleanup(unittest.TestCase):
    """Test suite for SQLite connection cleanup and leak prevention."""

    def test_managed_connection_closes_on_exit(self):
        """Verify managed_connection contextmanager closes SQLite connection on exit."""
        db_file = ":memory:"
        captured_conn = None
        with managed_connection(db_file) as conn:
            captured_conn = conn
            conn.execute("CREATE TABLE t (id INT)")

        # Connection should be closed after exiting context manager block
        with self.assertRaises(sqlite3.ProgrammingError):
            captured_conn.execute("SELECT * FROM t")

    def test_auth_connect_closes_on_exit(self):
        """Verify src.db.auth._connect closes SQLite connection on exit."""
        captured_conn = None
        with auth_connect() as conn:
            captured_conn = conn
            conn.execute("SELECT 1")

        with self.assertRaises(sqlite3.ProgrammingError):
            captured_conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
