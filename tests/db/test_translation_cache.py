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
    get_cached_translation,
    reset_translation_cache_counters,
    get_cache_performance_summary,
    get_translation_cache_hit_ratio,
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
            mock_conn.cursor.return_value.execute.side_effect = sqlite3.Error("DB locked")

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
            mock_conn.cursor.return_value.execute.side_effect = sqlite3.Error("DB locked")

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

