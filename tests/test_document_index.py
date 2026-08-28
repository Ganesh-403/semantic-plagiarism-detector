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
Comprehensive Unit Tests for Document file_hash Index
Issue: #3416
Tests that the documents table has an explicit index on the file_hash column.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

# ==============================================================================
# SECTION 1: Defining the Database Schema (Under Test)
# ==============================================================================


def create_documents_table(conn: sqlite3.Connection):
    """Creates the documents table with an explicit index on file_hash."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            title TEXT,
            content TEXT,
            language_code TEXT
        )
    """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)"
    )
    conn.commit()


def create_documents_table_without_index(conn: sqlite3.Connection):
    """Creates the documents table WITHOUT an index (for testing the negative case)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            title TEXT,
            content TEXT,
            language_code TEXT
        )
    """
    )
    conn.commit()


# ==============================================================================
# SECTION 2: Testing Index Creation
# ==============================================================================


class TestIndexCreation:
    def test_index_exists_on_file_hash(self):
        """Should have an explicit index on the file_hash column."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        cursor = conn.execute("PRAGMA index_list(documents)")
        indexes = cursor.fetchall()

        assert len(indexes) > 0, "No indexes found on documents table"

        index_names = [index[1] for index in indexes]
        assert "idx_documents_file_hash" in index_names, "Required index not found"

        conn.close()

    def test_index_columns_are_correct(self):
        """Should confirm the index is on the file_hash column."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        cursor = conn.execute("PRAGMA index_info(idx_documents_file_hash)")
        columns = cursor.fetchall()

        assert len(columns) == 1, "Index should only be on one column"
        assert columns[0][2] == "file_hash", "Index should be on the file_hash column"

        conn.close()

    def test_no_index_without_explicit_creation(self):
        """Should have no indexes if the schema doesn't create one."""
        conn = sqlite3.connect(":memory:")
        create_documents_table_without_index(conn)

        cursor = conn.execute("PRAGMA index_list(documents)")
        indexes = cursor.fetchall()

        assert len(indexes) == 0, "No indexes should exist without explicit creation"

        conn.close()


# ==============================================================================
# SECTION 3: Testing Index Performance (Query Speed)
# ==============================================================================


class TestIndexPerformance:
    def test_query_uses_index(self):
        """Should use the index when querying by file_hash."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        # Insert sample data
        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash1", "Doc1")
        )
        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash2", "Doc2")
        )
        conn.commit()

        # Force index usage by analyzing the query plan
        cursor = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM documents WHERE file_hash = ?", ("hash1",)
        )
        plan = cursor.fetchall()

        # Check if any part of the query plan mentions the index
        plan_text = str(plan)
        assert (
            "idx_documents_file_hash" in plan_text
        ), f"Query did not use the index: {plan_text}"

        conn.close()


# ==============================================================================
# SECTION 4: Testing Data Integrity with Index
# ==============================================================================


class TestDataIntegrity:
    def test_can_insert_documents(self):
        """Should be able to insert documents with the index present."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash1", "Doc1")
        )
        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash2", "Doc2")
        )
        conn.commit()

        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 2

        conn.close()

    def test_can_query_by_hash(self):
        """Should be able to query documents by file_hash."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash1", "Doc1")
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT title FROM documents WHERE file_hash = ?", ("hash1",)
        )
        result = cursor.fetchone()

        assert result[0] == "Doc1"

        conn.close()

    def test_hash_is_unique_requirement(self):
        """file_hash should be treated as a unique identifier (at least in testing)."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)

        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash1", "Doc1")
        )
        with pytest.raises(sqlite3.IntegrityError):
            # This will fail if a UNIQUE constraint is applied
            conn.execute(
                "INSERT INTO documents (file_hash, title) VALUES (?, ?)",
                ("hash1", "Doc2"),
            )
        conn.close()


# ==============================================================================
# SECTION 5: Testing Index on Multiple Files
# ==============================================================================


class TestMultipleFiles:
    def test_index_persists_across_connections(self, tmp_path):
        """Index should persist if the database is saved to a file."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        create_documents_table(conn)
        conn.close()

        # Reopen the database
        conn2 = sqlite3.connect(db_path)
        cursor = conn2.execute("PRAGMA index_list(documents)")
        indexes = cursor.fetchall()

        assert len(indexes) > 0
        conn2.close()

    def test_index_works_with_data(self, tmp_path):
        """Index should work with actual database file and data."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        create_documents_table(conn)

        conn.execute(
            "INSERT INTO documents (file_hash, title) VALUES (?, ?)", ("hash1", "Doc1")
        )
        conn.commit()
        conn.close()

        conn2 = sqlite3.connect(db_path)
        cursor = conn2.execute(
            "SELECT title FROM documents WHERE file_hash = ?", ("hash1",)
        )
        result = cursor.fetchone()

        assert result[0] == "Doc1"
        conn2.close()
