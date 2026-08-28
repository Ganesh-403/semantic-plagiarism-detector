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

"""
Unit tests for managed SQLite connections and cleanup (Issue #1707).
"""

import sqlite3
import unittest

from src.db.auth import _connect as auth_connect
from src.db.common import managed_connection


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
