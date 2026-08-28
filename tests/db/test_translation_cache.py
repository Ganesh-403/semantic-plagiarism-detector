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
tests/db/test_translation_cache.py
-----------------------------------
Tests for translation caching system and TTL expiration helpers.
"""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from src.db import translation_cache
from src.db.translation_cache import (
    DB_PATH,
    cache_translation,
    get_cache_performance_summary,
    get_cached_translation,
    get_translation_cache_hit_ratio,
    reset_translation_cache_counters,
)


def test_translation_cache_miss():
    assert get_cached_translation("Texto no guardado") is None


def test_translation_cache_hit_and_store():
    foreign_text = "Bonjour le monde"
    expected_english = "Hello world"

    cache_translation(
        foreign_text, expected_english, source_lang="fr", target_lang="en"
    )

    cached_result = get_cached_translation(
        foreign_text, source_lang="fr", target_lang="en"
    )
    assert cached_result == expected_english


def test_translation_cache_index_exists():
    # Make sure DB is initialized
    get_cached_translation("Some text")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='translation_cache'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_translation_cache_created_at" in indexes
    finally:
        conn.close()


class TestTranslationCacheTTL:
    """Test suite for TTL expiration and purge helpers."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary database path for isolated testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            yield tmp.name
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    @pytest.fixture(autouse=True)
    def override_db_path(self, temp_db_path):
        """Override the module-level DB_PATH for the duration of the test."""
        original_path = translation_cache.DB_PATH
        translation_cache.DB_PATH = temp_db_path
        yield
        translation_cache.DB_PATH = original_path

    def _seed_cache_with_dates(self, conn, days_ago_list):
        """Helper to insert cache entries with specific historical dates."""
        cursor = conn.cursor()
        for days_ago in days_ago_list:
            cursor.execute(
                """
                INSERT INTO translation_cache
                (text_hash, foreign_text, translated_text, source_lang, target_lang, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now', '-' || ? || ' days'))
                """,
                (
                    f"hash_{days_ago}",
                    f"foreign_{days_ago}",
                    f"translated_{days_ago}",
                    "auto",
                    "en",
                    days_ago,
                ),
            )
        conn.commit()

    def test_purge_expired_translation_cache_default_days(self, temp_db_path):
        """Test purging with the default 60 days threshold."""
        conn = sqlite3.connect(temp_db_path)
        # Ensure schema initialization on temp DB
        translation_cache.get_cached_translation("init")
        # Seed: 2 entries at 30 days (should stay), 2 entries at 90 days (should be purged)
        self._seed_cache_with_dates(conn, [30, 30, 90, 90])

        deleted_count = translation_cache.purge_expired_translation_cache()

        assert deleted_count == 2
        conn.execute("SELECT COUNT(*) FROM translation_cache")
        assert conn.fetchone()[0] == 2

    def test_purge_expired_translation_cache_custom_days(self, temp_db_path):
        """Test purging with a custom days_old threshold."""
        conn = sqlite3.connect(temp_db_path)
        translation_cache.get_cached_translation("init")
        # Seed: 1 entry at 10 days, 1 entry at 20 days, 1 entry at 30 days
        self._seed_cache_with_dates(conn, [10, 20, 30])

        deleted_count = translation_cache.purge_expired_translation_cache(days_old=15)

        assert deleted_count == 2  # 20 and 30 days old are purged
        conn.execute("SELECT COUNT(*) FROM translation_cache")
        assert conn.fetchone()[0] == 1  # Only the 10-day-old entry remains

    def test_purge_expired_translation_cache_zero_days(self, temp_db_path):
        """Test purging with 0 days (should purge nothing inserted 'now')."""
        conn = sqlite3.connect(temp_db_path)
        translation_cache.get_cached_translation("init")
        conn.execute(
            """
            INSERT INTO translation_cache
            (text_hash, foreign_text, translated_text, source_lang, target_lang)
            VALUES ('hash_now', 'foreign', 'translated', 'auto', 'en')
            """
        )
        conn.commit()

        deleted_count = translation_cache.purge_expired_translation_cache(days_old=0)

        # Entries created at 'now' are not strictly less than 'now', so 0 deleted
        assert deleted_count == 0

    def test_purge_expired_translation_cache_negative_days_raises_error(self):
        """Test that negative days_old raises a ValueError."""
        with pytest.raises(ValueError, match="days_old must be a non-negative integer"):
            translation_cache.purge_expired_translation_cache(days_old=-5)

    def test_purge_expired_translation_cache_handles_db_error(self, temp_db_path):
        """Test that database errors during purge are logged and return 0."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = mock_connect.return_value.__enter__.return_value
            mock_conn.cursor.return_value.execute.side_effect = sqlite3.Error(
                "DB locked"
            )

            deleted_count = translation_cache.purge_expired_translation_cache()

            assert deleted_count == 0

    def test_purge_translation_cache_older_than_default_days(self, temp_db_path):
        """Test purging with the default 30 days threshold."""
        conn = sqlite3.connect(temp_db_path)
        # Ensure schema initialization on temp DB
        translation_cache.get_cached_translation("init")
        # Seed: 2 entries at 15 days (should stay), 2 entries at 45 days (should be purged)
        self._seed_cache_with_dates(conn, [15, 15, 45, 45])

        deleted_count = translation_cache.purge_translation_cache_older_than()

        assert deleted_count == 2
        conn.execute("SELECT COUNT(*) FROM translation_cache")
        assert conn.fetchone()[0] == 2

    def test_purge_translation_cache_older_than_custom_days(self, temp_db_path):
        """Test purging with a custom days threshold."""
        conn = sqlite3.connect(temp_db_path)
        translation_cache.get_cached_translation("init")
        # Seed: 1 entry at 10 days, 1 entry at 20 days, 1 entry at 30 days
        self._seed_cache_with_dates(conn, [10, 20, 30])

        deleted_count = translation_cache.purge_translation_cache_older_than(days=15)

        assert deleted_count == 2  # 20 and 30 days old are purged
        conn.execute("SELECT COUNT(*) FROM translation_cache")
        assert conn.fetchone()[0] == 1  # Only the 10-day-old entry remains

    def test_purge_translation_cache_older_than_negative_days_raises_error(self):
        """Test that negative days raises a ValueError."""
        with pytest.raises(ValueError, match="days must be a non-negative integer"):
            translation_cache.purge_translation_cache_older_than(days=-5)

    def test_purge_translation_cache_older_than_handles_db_error(self, temp_db_path):
        """Test that database errors during purge are logged and return 0."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = mock_connect.return_value.__enter__.return_value
            mock_conn.cursor.return_value.execute.side_effect = sqlite3.Error(
                "DB locked"
            )

            deleted_count = translation_cache.purge_translation_cache_older_than()

            assert deleted_count == 0

    def test_get_translation_cache_stats(self, temp_db_path):
        """Test retrieving accurate cache statistics."""
        conn = sqlite3.connect(temp_db_path)
        translation_cache.get_cached_translation("init")
        self._seed_cache_with_dates(conn, [10, 50, 100])

        stats = translation_cache.get_translation_cache_stats()

        assert stats["total_entries"] == 3
        assert stats["oldest_entry_days"] == 100

    def test_get_translation_cache_stats_empty(self, temp_db_path):
        """Test stats retrieval on an empty cache."""
        stats = translation_cache.get_translation_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["oldest_entry_days"] == 0

    def test_get_translation_cache_hit_ratio_initial(self):
        translation_cache.cache_hits = 0
        translation_cache.cache_misses = 0
        assert get_translation_cache_hit_ratio() == 0.0

    def test_get_translation_cache_hit_ratio_all_hits(self):
        translation_cache.cache_hits = 1
        translation_cache.cache_misses = 0
        assert get_translation_cache_hit_ratio() == 1.0

    def test_get_translation_cache_hit_ratio_all_misses(self):
        translation_cache.cache_hits = 0
        translation_cache.cache_misses = 1
        assert get_translation_cache_hit_ratio() == 0.0

    def test_get_translation_cache_hit_ratio_mixed(self):
        translation_cache.cache_hits = 3
        translation_cache.cache_misses = 1
        assert get_translation_cache_hit_ratio() == 0.75

    def test_get_translation_cache_hit_ratio_accumulates(self, temp_db_path):
        # Reset counters
        translation_cache.cache_hits = 0
        translation_cache.cache_misses = 0

        # First lookup: not found -> 0 hits, 1 miss
        get_cached_translation("hola")
        assert translation_cache.cache_hits == 0
        assert translation_cache.cache_misses == 1
        assert get_translation_cache_hit_ratio() == 0.0

        # Insert a translation
        cache_translation("hola", "hello")

        # Second lookup: found -> 1 hit, 1 miss
        res2 = get_cached_translation("hola")
        assert res2 == "hello"
        assert translation_cache.cache_hits == 1
        assert translation_cache.cache_misses == 1
        assert get_translation_cache_hit_ratio() == 0.5

        # Third lookup: found -> 2 hits, 1 miss
        res3 = get_cached_translation("hola")
        assert res3 == "hello"
        assert translation_cache.cache_hits == 2
        assert translation_cache.cache_misses == 1
        assert get_translation_cache_hit_ratio() == 2 / 3


def test_init_db_closes_connection():
    """Verify that _init_db() explicitly closes the database connection."""
    from unittest.mock import MagicMock

    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        translation_cache._init_db()

        # Verify close() was called on mock_conn
        mock_conn.close.assert_called()


def test_get_cache_performance_summary():
    """Test retrieving query cache performance summary telemetry."""
    # 1. Reset counters
    reset_translation_cache_counters()

    # 2. Check summary with zero requests
    summary = get_cache_performance_summary()
    assert summary["total_requests"] == 0
    assert summary["hits"] == 0
    assert summary["misses"] == 0
    assert summary["hit_ratio_percentage"] == 0.0

    # 3. Cache and perform lookups to generate hits and misses
    get_cached_translation("non-existent-text-xyz")  # Miss
    cache_translation("Bonjour", "Hello", "fr", "en")
    get_cached_translation("Bonjour", "fr", "en")  # Hit
    get_cached_translation("Bonjour", "fr", "en")  # Hit

    # 4. Check summary telemetry
    summary = get_cache_performance_summary()
    assert summary["total_requests"] == 3
    assert summary["hits"] == 2
    assert summary["misses"] == 1
    assert abs(summary["hit_ratio_percentage"] - 66.6666666) < 0.1


def test_get_translation_cache_stats(self, temp_db_path):
    """Test retrieving accurate cache statistics."""
    conn = sqlite3.connect(temp_db_path)
    translation_cache.get_cached_translation("init")
    self._seed_cache_with_dates(conn, [10, 50, 100])

    stats = translation_cache.get_translation_cache_stats()

    assert stats == {"total_entries": 3}


def test_get_translation_cache_stats_empty(self, temp_db_path):
    """Test stats retrieval on an empty cache."""
    stats = translation_cache.get_translation_cache_stats()
    assert stats == {"total_entries": 0}


def test_get_cached_translation_recovers_from_malformed_database(tmp_path, caplog):
    """A malformed SQLite cache is replaced and its schema is recreated."""
    cache_path = tmp_path / "translation_cache.db"
    with sqlite3.connect(cache_path) as conn:
        conn.execute("CREATE TABLE marker (value TEXT)")
        conn.execute("INSERT INTO marker VALUES ('keep schema test')")

    corrupted_bytes = bytearray(cache_path.read_bytes())
    corrupted_bytes[100:120] = b"X" * 20
    cache_path.write_bytes(corrupted_bytes)

    with patch.object(translation_cache, "_CACHE_DB_PATH", cache_path):
        with caplog.at_level("CRITICAL", logger=translation_cache.logger.name):
            result = get_cached_translation("malformed-cache", "en", "fr")

        assert result is None
        assert "corrupted" in caplog.text.lower()
        assert "deleting it and recreating the schema" in caplog.text

        with sqlite3.connect(cache_path) as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='translation_cache'"
            ).fetchone()

        assert table == ("translation_cache",)


def test_get_cached_translation_handles_non_corruption_database_error(tmp_path, caplog):
    """Non-corruption DatabaseError instances remain ordinary cache misses."""
    cache_path = tmp_path / "translation_cache.db"
    cache_path.write_bytes(b"placeholder")

    with patch.object(translation_cache, "_CACHE_DB_PATH", cache_path):
        with patch.object(
            translation_cache,
            "_connect",
            side_effect=sqlite3.DatabaseError("database is locked"),
        ):
            with caplog.at_level("ERROR", logger=translation_cache.logger.name):
                result = get_cached_translation("locked-cache")

    assert result is None
    assert "Failed to query translation cache" in caplog.text
    assert cache_path.read_bytes() == b"placeholder"


def test_translation_cache_uses_lock_for_get_and_save():
    """Verify that get_cached_translation and save_translation acquire the global _lock."""
    from unittest.mock import MagicMock

    from src.db.translation_cache import _lock, save_translation

    mock_lock = MagicMock()
    with patch("src.db.translation_cache._lock", mock_lock):
        # Trigger get_cached_translation
        get_cached_translation("test lock query text", "en", "es")
        # Trigger save_translation
        save_translation("test lock save text", "en", "es", "translation text")

    # Assert that _lock was acquired (enter) and released (exit)
    assert mock_lock.__enter__.call_count >= 2
    assert mock_lock.__exit__.call_count >= 2


def test_translation_cache_uses_provided_connection():
    """Verify that get_cached_translation and save_translation use the provided connection and do not open a new one."""
    from unittest.mock import MagicMock

    from src.db.translation_cache import get_cached_translation, save_translation

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"translated_text": "cached_val"}
    mock_conn.execute.return_value = mock_cursor

    with patch("src.db.translation_cache._connect") as mock_connect, patch(
        "src.db.translation_cache._get_connection"
    ) as mock_get_conn:
        # Call get_cached_translation with provided connection
        result = get_cached_translation("source", "fr", "en", conn=mock_conn)
        assert result == "cached_val"

        # Call save_translation with provided connection
        save_result = save_translation("source", "fr", "en", "target", conn=mock_conn)
        assert save_result is True

        # Assert that no new connections were spawned
        mock_connect.assert_not_called()
        mock_get_conn.assert_not_called()

        # Assert that the provided connection was used
        assert mock_conn.execute.call_count >= 2


def test_migrate_legacy_cache_rehashes_and_inserts(tmp_path):
    """Legacy rows are copied using the modern language-aware cache hash."""
    legacy_db = tmp_path / "legacy.db"
    modern_db = tmp_path / "translation_cache.db"

    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            """
            CREATE TABLE legacy_translation_cache (
                text_hash TEXT PRIMARY KEY,
                foreign_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_lang TEXT,
                target_lang TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO legacy_translation_cache
            (text_hash, foreign_text, translated_text, source_lang, target_lang)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("legacy-hash", " Bonjour ", "Hello", "FR", "EN"),
        )

    stats = translation_cache.migrate_legacy_cache(legacy_db, modern_db)

    assert stats == {"scanned": 1, "migrated": 1, "skipped": 0, "errors": 0}

    with sqlite3.connect(modern_db) as conn:
        row = conn.execute(
            """
            SELECT source_hash, source_text, source_lang, target_lang,
                   translated_text
            FROM translation_cache
            """
        ).fetchone()

    assert row is not None
    assert row[0] == translation_cache._generate_hash(" Bonjour ", "fr", "en")
    assert row[0] != "legacy-hash"
    assert row[1:] == (" Bonjour ", "fr", "en", "Hello")


def test_migrate_legacy_cache_is_idempotent(tmp_path):
    """Running the migration twice does not overwrite existing modern rows."""
    legacy_db = tmp_path / "legacy.db"
    modern_db = tmp_path / "translation_cache.db"

    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            """
            CREATE TABLE legacy_translation_cache (
                text_hash TEXT PRIMARY KEY,
                foreign_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_lang TEXT,
                target_lang TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO legacy_translation_cache
            (text_hash, foreign_text, translated_text, source_lang, target_lang)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("legacy-hash", "hola", "hello", "es", "en"),
        )

    first = translation_cache.migrate_legacy_cache(legacy_db, modern_db)
    second = translation_cache.migrate_legacy_cache(legacy_db, modern_db)

    assert first["migrated"] == 1
    assert second["migrated"] == 0
    assert second["skipped"] == 1

    with sqlite3.connect(modern_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM translation_cache").fetchone()[0] == 1
