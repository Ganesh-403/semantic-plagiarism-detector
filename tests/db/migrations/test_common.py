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
Unit tests for src.db.migrations.common helpers.
"""

import sqlite3
from unittest.mock import patch

import pytest

from src.db.migrations.common import (
    column_exists,
    enable_wal_mode,
    get_journal_mode,
    get_user_version,
    index_exists,
    perform_wal_checkpoint,
    quote_identifier,
    set_user_version,
    table_exists,
)


class TestWalModeHelpers:
    """Test suite for WAL mode optimization helpers."""

    @pytest.fixture
    def in_memory_db(self):
        """Provide a clean in-memory SQLite database for each test."""
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def test_enable_wal_mode_success(self, in_memory_db):
        """Test that enable_wal_mode successfully sets WAL and NORMAL synchronous."""
        result = enable_wal_mode(in_memory_db)
        assert result.lower() == "wal"

        # Verify synchronous mode
        cursor = in_memory_db.cursor()
        cursor.execute("PRAGMA synchronous;")
        assert cursor.fetchone()[0] == 1  # 1 is NORMAL

    def test_enable_wal_mode_logs_status(self, in_memory_db):
        """Test that enable_wal_mode logs the journal mode status."""
        with patch("src.db.migrations.common.logger.info") as mock_logger:
            enable_wal_mode(in_memory_db)
            mock_logger.assert_called_once()
            assert "SQLite WAL mode enabled" in mock_logger.call_args[0][0]
            assert "Journal mode: wal" in mock_logger.call_args[0][0]

    def test_enable_wal_mode_handles_error(self, in_memory_db):
        """Test that enable_wal_mode raises sqlite3.Error on failure."""
        with patch.object(in_memory_db, "cursor") as mock_cursor:
            mock_cursor.return_value.execute.side_effect = sqlite3.Error("DB locked")
            with pytest.raises(sqlite3.Error):
                enable_wal_mode(in_memory_db)

    def test_get_journal_mode_returns_current_mode(self, in_memory_db):
        """Test that get_journal_mode returns the correct current mode."""
        # Default is usually 'memory' or 'delete' for :memory:
        mode = get_journal_mode(in_memory_db)
        assert mode in ["memory", "delete", "wal"]

    def test_get_journal_mode_handles_error(self, in_memory_db):
        """Test that get_journal_mode returns 'unknown' on error."""
        with patch.object(in_memory_db, "cursor") as mock_cursor:
            mock_cursor.return_value.execute.side_effect = sqlite3.Error("DB error")
            assert get_journal_mode(in_memory_db) == "unknown"

    @pytest.mark.parametrize("mode", ["PASSIVE", "FULL", "RESTART", "TRUNCATE"])
    def test_perform_wal_checkpoint_valid_modes(self, in_memory_db, mode):
        """Test perform_wal_checkpoint with all valid modes."""
        # First enable WAL to make checkpoint meaningful
        enable_wal_mode(in_memory_db)
        result = perform_wal_checkpoint(in_memory_db, mode=mode)
        assert result["mode"] == mode.upper()
        assert isinstance(result["result"], tuple)

    def test_perform_wal_checkpoint_invalid_mode(self, in_memory_db):
        """Test perform_wal_checkpoint raises ValueError for invalid mode."""
        with pytest.raises(ValueError, match="Invalid checkpoint mode"):
            perform_wal_checkpoint(in_memory_db, mode="INVALID")


class TestSchemaHelpers:
    """Test suite for general schema inspection helpers."""

    @pytest.fixture
    def sample_db(self):
        """Provide a database with a sample table and index."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE INDEX idx_users_name ON users(name)")
        yield conn
        conn.close()

    def test_quote_identifier_escapes_quotes(self):
        """Test that quote_identifier properly escapes double quotes."""
        assert quote_identifier('test"table') == '"test""table"'

    def test_quote_identifier_rejects_null(self):
        """Test that quote_identifier rejects NUL bytes."""
        with pytest.raises(ValueError, match="contain no NUL"):
            quote_identifier("table\x00name")

    def test_table_exists_true(self, sample_db):
        """Test table_exists returns True for existing table."""
        assert table_exists(sample_db, "users") is True

    def test_table_exists_false(self, sample_db):
        """Test table_exists returns False for non-existing table."""
        assert table_exists(sample_db, "nonexistent") is False

    def test_column_exists_true(self, sample_db):
        """Test column_exists returns True for existing column."""
        assert column_exists(sample_db, "users", "name") is True

    def test_column_exists_false(self, sample_db):
        """Test column_exists returns False for non-existing column."""
        assert column_exists(sample_db, "users", "email") is False

    def test_index_exists_true(self, sample_db):
        """Test index_exists returns True for existing index."""
        assert index_exists(sample_db, "idx_users_name") is True

    def test_index_exists_false(self, sample_db):
        """Test index_exists returns False for non-existing index."""
        assert index_exists(sample_db, "idx_nonexistent") is False

    def test_user_version_get_and_set(self, sample_db):
        """Test getting and setting user_version."""
        assert get_user_version(sample_db) == 0
        set_user_version(sample_db, 5)
        assert get_user_version(sample_db) == 5

    def test_set_user_version_rejects_negative(self, sample_db):
        """Test that set_user_version rejects negative values."""
        with pytest.raises(ValueError, match="cannot be negative"):
            set_user_version(sample_db, -1)
