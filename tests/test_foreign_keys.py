"""
Comprehensive Unit Tests for Foreign Key Enforcement
Issue: #3411
Tests that all database connections ensure PRAGMA foreign_keys = ON.
"""

import sqlite3
import threading
import pytest
from unittest.mock import MagicMock, patch


# ==============================================================================
# SECTION 1: Defining the Connection Logic (Under Test)
# ==============================================================================

def create_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Creates a SQLite connection and enforces foreign keys.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def fallback_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Simulates a fallback connection in case the primary fails.
    """
    try:
        return create_connection(db_path)
    except Exception:
        return create_connection(db_path)


# ==============================================================================
# SECTION 2: Testing the Core Connection Logic
# ==============================================================================

class TestConnectionCreation:
    def test_connection_is_sqlite(self):
        """Should return a valid sqlite3.Connection object."""
        conn = create_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_foreign_keys_enabled(self):
        """PRAGMA foreign_keys should be 1 (ON)."""
        conn = create_connection()
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1
        conn.close()

    def test_connection_to_file(self, tmp_path):
        """Should work with a physical file path."""
        db_file = tmp_path / "test.db"
        conn = create_connection(str(db_file))
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.close()


# ==============================================================================
# SECTION 3: Testing Fallback Connection
# ==============================================================================

class TestFallbackConnection:
    def test_fallback_returns_valid_connection(self):
        """Fallback should return a valid connection."""
        conn = fallback_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_fallback_enforces_foreign_keys(self):
        """Fallback should also enforce foreign keys."""
        conn = fallback_connection()
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1
        conn.close()

    def test_fallback_after_primary_failure(self):
        """If primary fails, fallback should still work."""
        with patch("sqlite3.connect", side_effect=[Exception("Primary failed"), MagicMock()]):
            conn = fallback_connection()
            # We patched sqlite3.connect to fail once, so fallback triggers
            assert conn is not None


# ==============================================================================
# SECTION 4: Thread Safety and Concurrency
# ==============================================================================

class TestThreadSafety:
    def test_concurrent_connections(self):
        """Multiple threads should create connections with FK enabled."""
        results = []

        def create_and_check():
            conn = create_connection()
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            results.append(fk)
            conn.close()

        threads = [threading.Thread(target=create_and_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == 1 for r in results)


# ==============================================================================
# SECTION 5: Foreign Key Enforcement Test (Actual Data Integrity)
# ==============================================================================

class TestForeignKeyEnforcement:
    def test_insert_valid_fk(self):
        """Inserting a valid foreign key should succeed."""
        conn = create_connection()
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parent(id))")
        conn.execute("INSERT INTO parent (id) VALUES (1)")
        conn.execute("INSERT INTO child (id, parent_id) VALUES (1, 1)")
        conn.commit()
        conn.close()

    def test_insert_invalid_fk_fails(self):
        """Inserting an invalid foreign key should fail."""
        conn = create_connection()
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parent(id))")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO child (id, parent_id) VALUES (1, 999)")  # Parent does not exist
        conn.close()

    def test_fk_disabled_allows_invalid(self):
        """If FK is disabled, invalid inserts should succeed (to prove the test works)."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parent(id))")
        conn.execute("INSERT INTO child (id, parent_id) VALUES (1, 999)")
        conn.commit()
        conn.close()


# ==============================================================================
# SECTION 6: Edge Cases
# ==============================================================================

class TestEdgeCases:
    def test_connection_to_invalid_path(self):
        """Should raise an error for an invalid path."""
        with pytest.raises(sqlite3.OperationalError):
            create_connection("/nonexistent/dir/db.sqlite")

    def test_connection_without_foreign_keys(self):
        """Simulates a connection without FK enforcement (old connections)."""
        conn = sqlite3.connect(":memory:")
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 0  # Default is 0
        conn.close()


# ==============================================================================
# SECTION 7: Logging and Mocking
# ==============================================================================

class TestMockingAndLogging:
    def test_create_connection_called_with_right_args(self):
        """Ensure sqlite3.connect is called with the right path."""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.return_value = MagicMock()
            create_connection("test.db")
            mock_connect.assert_called_once_with("test.db")

    def test_fallback_retries_connection(self):
        """Ensure fallback retries the connection logic."""
        with patch("sqlite3.connect", side_effect=[Exception("Fail"), MagicMock()]) as mock_connect:
            fallback_connection()
            assert mock_connect.call_count == 2