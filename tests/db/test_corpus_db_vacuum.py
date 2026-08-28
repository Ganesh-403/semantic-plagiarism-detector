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

"""Tests for corpus database VACUUM maintenance (Issue #3417)."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.db.corpus_db import vacuum_corpus_database


class TestVacuumCorpusDatabase(unittest.TestCase):
    """Test corpus SQLite VACUUM maintenance and connection cleanup."""

    def test_vacuum_closes_existing_connections_before_running(self):
        """VACUUM closes pooled connections before opening its maintenance connection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "corpus.db"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE documents (id INTEGER)")
            conn.commit()
            conn.close()

            with patch(
                "src.db.corpus_db.get_corpus_db_path", return_value=db_path
            ), patch("src.db.corpus_db.close_connections") as close_connections:
                vacuum_corpus_database()

            close_connections.assert_called_once_with(all_threads=True)

            verify_conn = sqlite3.connect(db_path)
            try:
                verify_conn.execute("SELECT * FROM documents")
            finally:
                verify_conn.close()

    def test_vacuum_uses_autocommit_and_closes_connection(self):
        """VACUUM runs with isolation_level=None and always closes its connection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "corpus.db"
            sqlite3.connect(db_path).close()
            connections = []

            real_connect = sqlite3.connect

            def connect(*args, **kwargs):
                conn = real_connect(*args, **kwargs)
                connections.append(conn)
                return conn

            with patch(
                "src.db.corpus_db.get_corpus_db_path", return_value=db_path
            ), patch("src.db.corpus_db.close_connections"), patch(
                "src.db.corpus_db.sqlite3.connect", side_effect=connect
            ):
                vacuum_corpus_database()

            self.assertEqual(len(connections), 1)
            self.assertIsNone(connections[0].isolation_level)
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[0].execute("SELECT 1")

    def test_vacuum_closes_connection_when_execution_fails(self):
        """The maintenance connection is closed even when VACUUM raises."""

        class FakeConnection:
            def __init__(self):
                self.isolation_level = ""
                self.closed = False

            def execute(self, sql):
                self.assert_sql = sql
                raise sqlite3.DatabaseError("vacuum failed")

            def close(self):
                self.closed = True

        fake = FakeConnection()
        with patch(
            "src.db.corpus_db.get_corpus_db_path", return_value=Path("/tmp/corpus.db")
        ), patch("src.db.corpus_db.close_connections"), patch(
            "src.db.corpus_db.sqlite3.connect", return_value=fake
        ):
            with self.assertRaises(sqlite3.DatabaseError):
                vacuum_corpus_database()

        self.assertTrue(fake.closed)
        self.assertEqual(fake.assert_sql, "VACUUM")
        self.assertIsNone(fake.isolation_level)


if __name__ == "__main__":
    unittest.main()
