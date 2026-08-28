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
tests/db/test_translation_cache_purge.py
----------------------------------------
Comprehensive unit tests for the translation cache TTL cleanup mechanism.

Verifies that purge_old_translations correctly identifies and deletes
stale entries based on the created_at timestamp (Issue #2985).
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from src.db.translation_cache import (
    get_cache_stats,
    initialize_cache_db,
    purge_old_translations,
    save_translation,
)


@pytest.fixture
def temp_cache_db(tmp_path):
    """Create a temporary translation cache database for testing."""
    db_path = tmp_path / "test_cache.db"
    initialize_cache_db(db_path)
    yield db_path
    # Cleanup is handled by tmp_path fixture


class TestPurgeOldTranslations:
    """Test suite for the purge_old_translations function."""

    def test_purge_deletes_old_entries(self, temp_cache_db):
        """Verify entries older than the threshold are deleted."""
        # Insert a translation with a fake old timestamp
        old_date = (datetime.utcnow() - timedelta(days=40)).isoformat()

        with sqlite3.connect(temp_cache_db) as conn:
            conn.execute(
                """
                INSERT INTO translations
                (source_hash, source_text, source_lang, target_lang, translated_text, created_at, last_accessed_at)
                VALUES ('hash1', 'old text', 'es', 'en', 'old translated', ?, ?)
                """,
                (old_date, old_date),
            )
            conn.commit()

        # Insert a recent translation
        save_translation(
            "new text", "es", "en", "new translated", db_path=temp_cache_db
        )

        # Purge entries older than 30 days
        deleted = purge_old_translations(days=30, db_path=temp_cache_db)

        assert deleted == 1

        # Verify only the new entry remains
        stats = get_cache_stats(db_path=temp_cache_db)
        assert stats["total_entries"] == 1

    def test_purge_keeps_recent_entries(self, temp_cache_db):
        """Verify entries newer than the threshold are kept."""
        # Insert translations from 5, 10, and 15 days ago
        for days_ago in [5, 10, 15]:
            old_date = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
            with sqlite3.connect(temp_cache_db) as conn:
                conn.execute(
                    """
                    INSERT INTO translations
                    (source_hash, source_text, source_lang, target_lang, translated_text, created_at, last_accessed_at)
                    VALUES (?, 'text', 'es', 'en', 'translated', ?, ?)
                    """,
                    (f"hash_{days_ago}", old_date, old_date),
                )
                conn.commit()

        # Purge entries older than 30 days
        deleted = purge_old_translations(days=30, db_path=temp_cache_db)

        assert deleted == 0

        stats = get_cache_stats(db_path=temp_cache_db)
        assert stats["total_entries"] == 3

    def test_purge_exact_boundary(self, temp_cache_db):
        """Verify entries exactly at the boundary are handled correctly."""
        # Insert entry exactly 30 days ago
        boundary_date = (datetime.utcnow() - timedelta(days=30)).isoformat()

        with sqlite3.connect(temp_cache_db) as conn:
            conn.execute(
                """
                INSERT INTO translations
                (source_hash, source_text, source_lang, target_lang, translated_text, created_at, last_accessed_at)
                VALUES ('hash_boundary', 'text', 'es', 'en', 'translated', ?, ?)
                """,
                (boundary_date, boundary_date),
            )
            conn.commit()

        # Purge entries older than 30 days (strictly less than)
        deleted = purge_old_translations(days=30, db_path=temp_cache_db)

        # The entry is exactly 30 days old, so it should NOT be deleted
        # (created_at < cutoff_date, where cutoff is now - 30 days)
        assert deleted == 0

    def test_purge_zero_days_deletes_all(self, temp_cache_db):
        """Verify purge with days=0 deletes all entries."""
        save_translation("text1", "es", "en", "trans1", db_path=temp_cache_db)
        save_translation("text2", "es", "en", "trans2", db_path=temp_cache_db)

        deleted = purge_old_translations(days=0, db_path=temp_cache_db)

        # Since cutoff is "now", and created_at is slightly in the past,
        # all entries should be deleted.
        assert deleted == 2

        stats = get_cache_stats(db_path=temp_cache_db)
        assert stats["total_entries"] == 0

    def test_purge_negative_days_raises_error(self, temp_cache_db):
        """Verify negative days parameter raises ValueError."""
        with pytest.raises(ValueError, match="days must be >= 0"):
            purge_old_translations(days=-5, db_path=temp_cache_db)

    def test_purge_empty_database(self, temp_cache_db):
        """Verify purging an empty database returns 0."""
        deleted = purge_old_translations(days=30, db_path=temp_cache_db)
        assert deleted == 0


class TestCacheStats:
    """Test suite for the get_cache_stats function."""

    def test_stats_empty_database(self, temp_cache_db):
        """Verify stats for an empty database."""
        stats = get_cache_stats(db_path=temp_cache_db)

        assert stats["total_entries"] == 0
        assert stats["oldest_entry"] is None
        assert stats["newest_entry"] is None

    def test_stats_populated_database(self, temp_cache_db):
        """Verify stats for a populated database."""
        save_translation("text1", "es", "en", "trans1", db_path=temp_cache_db)
        save_translation("text2", "fr", "en", "trans2", db_path=temp_cache_db)

        stats = get_cache_stats(db_path=temp_cache_db)

        assert stats["total_entries"] == 2
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None
