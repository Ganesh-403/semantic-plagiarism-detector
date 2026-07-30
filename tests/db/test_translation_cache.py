"""
tests/db/test_translation_cache.py
-----------------------------------
Tests for translation caching system.
"""

from src.db.translation_cache import cache_translation, get_cached_translation


def test_translation_cache_miss():
    assert get_cached_translation("Texto no guardado") is None


def test_translation_cache_hit_and_store():
    foreign_text = "Bonjour le monde"
    expected_english = "Hello world"

    cache_translation(foreign_text, expected_english, source_lang="fr", target_lang="en")

    cached_result = get_cached_translation(foreign_text, source_lang="fr", target_lang="en")
    assert cached_result == expected_english


def test_translation_cache_index_exists():
    # Make sure DB is initialized
    get_cached_translation("Some text")

    from src.db.translation_cache import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='translation_cache'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_translation_cache_created_at" in indexes
    finally:
        conn.close()