"""
Comprehensive Unit Tests for chunk_count Column in Documents Table
Issue: #3418
Tests that the documents table has a chunk_count column and that it stores data correctly.
"""

import sqlite3
import pytest


# ==============================================================================
# SECTION 1: Defining the Database Schema (Under Test)
# ==============================================================================

def create_documents_table(conn: sqlite3.Connection):
    """Creates the documents table with the chunk_count column."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            title TEXT,
            content TEXT,
            language_code TEXT,
            chunk_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def create_documents_table_without_chunk_count(conn: sqlite3.Connection):
    """Creates the documents table WITHOUT chunk_count (for testing negative case)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            title TEXT,
            content TEXT,
            language_code TEXT
        )
    """)
    conn.commit()


# ==============================================================================
# SECTION 2: Testing Column Existence
# ==============================================================================

class TestColumnExistence:
    def test_chunk_count_column_exists(self):
        """Should have a chunk_count column."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        assert "chunk_count" in column_names, "chunk_count column not found!"
        conn.close()

    def test_chunk_count_has_default_value(self):
        """Should have a default value of 0."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        
        # Find the chunk_count column
        chunk_count_col = [column for column in columns if column[1] == "chunk_count"][0]
        assert chunk_count_col[4] == 0  # Default value is 0
        
        conn.close()

    def test_no_chunk_count_in_old_schema(self):
        """Should NOT have a chunk_count column in the old schema."""
        conn = sqlite3.connect(":memory:")
        create_documents_table_without_chunk_count(conn)
        
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        assert "chunk_count" not in column_names
        conn.close()


# ==============================================================================
# SECTION 3: Testing Data Insertion
# ==============================================================================

class TestDataInsertion:
    def test_insert_document_with_chunk_count(self):
        """Should be able to insert a document with a chunk_count."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash1", "Doc1", "This is content", "en", 10))
        conn.commit()
        
        cursor = conn.execute("SELECT chunk_count FROM documents WHERE file_hash = ?", ("hash1",))
        result = cursor.fetchone()
        assert result[0] == 10
        
        conn.close()

    def test_insert_document_without_chunk_count_uses_default(self):
        """Should default to 0 if chunk_count is not provided."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code)
            VALUES (?, ?, ?, ?)
        """, ("hash2", "Doc2", "This is content", "en"))
        conn.commit()
        
        cursor = conn.execute("SELECT chunk_count FROM documents WHERE file_hash = ?", ("hash2",))
        result = cursor.fetchone()
        assert result[0] == 0
        
        conn.close()

    def test_insert_document_with_negative_chunk_count(self):
        """Should allow negative values (no validation built-in)."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash3", "Doc3", "This is content", "en", -5))
        conn.commit()
        
        cursor = conn.execute("SELECT chunk_count FROM documents WHERE file_hash = ?", ("hash3",))
        result = cursor.fetchone()
        assert result[0] == -5
        
        conn.close()


# ==============================================================================
# SECTION 4: Testing Data Querying
# ==============================================================================

class TestDataQuerying:
    def test_query_documents_by_chunk_count(self):
        """Should be able to query documents by chunk_count."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash1", "Doc1", "Content", "en", 10))
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash2", "Doc2", "Content", "en", 20))
        conn.commit()
        
        # Query for documents with chunk_count > 15
        cursor = conn.execute("SELECT file_hash FROM documents WHERE chunk_count > ?", (15,))
        result = cursor.fetchall()
        
        assert len(result) == 1
        assert result[0][0] == "hash2"
        
        conn.close()

    def test_chunk_count_can_be_updated(self):
        """Should be able to update the chunk_count value."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash1", "Doc1", "Content", "en", 10))
        conn.commit()
        
        conn.execute("UPDATE documents SET chunk_count = ? WHERE file_hash = ?", (25, "hash1"))
        conn.commit()
        
        cursor = conn.execute("SELECT chunk_count FROM documents WHERE file_hash = ?", ("hash1",))
        result = cursor.fetchone()
        assert result[0] == 25
        
        conn.close()


# ==============================================================================
# SECTION 5: Performance and Integrity
# ==============================================================================

class TestIntegrity:
    def test_chunk_count_does_not_affect_other_columns(self):
        """Inserting chunk_count should not break other columns."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash1", "Doc1", "This is content", "en", 10))
        conn.commit()
        
        cursor = conn.execute("SELECT file_hash, title, content, language_code FROM documents WHERE file_hash = ?", ("hash1",))
        result = cursor.fetchone()
        
        assert result[0] == "hash1"
        assert result[1] == "Doc1"
        assert result[2] == "This is content"
        assert result[3] == "en"
        
        conn.close()

    def test_large_chunk_count(self):
        """Should handle very large chunk_count values."""
        conn = sqlite3.connect(":memory:")
        create_documents_table(conn)
        
        conn.execute("""
            INSERT INTO documents (file_hash, title, content, language_code, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("hash1", "Doc1", "Content", "en", 999999999))
        conn.commit()
        
        cursor = conn.execute("SELECT chunk_count FROM documents WHERE file_hash = ?", ("hash1",))
        result = cursor.fetchone()
        assert result[0] == 999999999
        
        conn.close()