"""
src/db/translation_cache.py
---------------------------
SQLite cache for translation API requests to preserve API quota.
Maps SHA-256 hash of (foreign_text, source_lang, target_lang) -> cached_text.
"""

import hashlib
import logging
import os
import sqlite3
from typing import Optional

from src.core.app_config import CORPUS_DB_PATH, FALLBACK_CORPUS_DB_PATH

logger = logging.getLogger(__name__)

# Seed the translation cache DB path from the centralized app_config.
# ``DB_PATH`` is intentionally kept as a module-level string so that tests
# importing ``src.db.translation_cache.DB_PATH`` continue to work.
DB_PATH = str(CORPUS_DB_PATH)

# In-memory counters for lookup hits and misses
_cache_hits = 0
_cache_misses = 0


def _init_db() -> None:
    """
    Initializes the translation cache table and indexes if they do not exist.

    The table includes a `created_at` timestamp column to support TTL-based
    expiration and purging of stale cache entries.
    """
    path = DB_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
    except (sqlite3.OperationalError, OSError, PermissionError):
        # Centralized temp-dir fallback (matches corpus_db.py and incidents.py)
        path = str(FALLBACK_CORPUS_DB_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_cache (
                    text_hash TEXT PRIMARY KEY,
                    foreign_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    source_lang TEXT,
                    target_lang TEXT DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translation_cache_created_at
                ON translation_cache(created_at)
                """
            )
            conn.commit()
    finally:
        conn.close()


def _hash_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Generates a unique SHA-256 hash for a given text and language pair.

    Args:
        text: The foreign text to be translated.
        source_lang: The source language code.
        target_lang: The target language code.

    Returns:
        str: A hexadecimal SHA-256 hash string.
    """
    key = f"{source_lang}:{target_lang}:{text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_cached_translation(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> Optional[str]:
    """
    Retrieves a cached translation if available.

    Args:
        text: The foreign text to look up.
        source_lang: The source language code.
        target_lang: The target language code.

    Returns:
        Optional[str]: The cached translated text, or None if not found.
    """
    _init_db()
    if not text or not text.strip():
        return None

    text_hash = _hash_text(text, source_lang, target_lang)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT translated_text FROM translation_cache WHERE text_hash = ?",
            (text_hash,),
        )
        row = cursor.fetchone()
        global _cache_hits, _cache_misses
        if row:
            _cache_hits += 1
            return row[0]
        else:
            _cache_misses += 1
            return None


def cache_translation(
    foreign_text: str,
    translated_text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
) -> None:
    """
    Stores a new translation in the SQLite cache.

    Args:
        foreign_text: The original foreign text.
        translated_text: The translated text.
        source_lang: The source language code.
        target_lang: The target language code.
    """
    _init_db()
    if not foreign_text or not translated_text:
        return

    text_hash = _hash_text(foreign_text, source_lang, target_lang)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO translation_cache
            (text_hash, foreign_text, translated_text, source_lang, target_lang)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text_hash, foreign_text, translated_text, source_lang, target_lang),
        )
        conn.commit()


def purge_expired_translation_cache(days_old: int = 60) -> int:
    """
    Purge translation cache entries older than the specified number of days.

    This prevents unbounded database growth by removing stale translation
    pairs that are unlikely to be requested again.

    Args:
        days_old: The age in days after which a cache entry is considered expired.
                  Defaults to 60 days.

    Returns:
        int: The number of rows successfully deleted from the cache.
    """
    _init_db()
    if days_old < 0:
        raise ValueError("days_old must be a non-negative integer.")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM translation_cache
                WHERE created_at < datetime('now', '-' || ? || ' days')
                """,
                (days_old,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(
                f"Purged {deleted_count} expired translation cache entries "
                f"older than {days_old} days."
            )
            return deleted_count
    except sqlite3.Error as e:
        logger.error(f"Failed to purge expired translation cache: {e}")
        return 0


def get_translation_cache_stats() -> dict:
    """
    Get statistics about the current translation cache.

    Returns:
        dict: A dictionary containing total entries and oldest entry age in days.
    """
    _init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translation_cache")
            total_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT CAST(JULIANDAY('now') - JULIANDAY(MIN(created_at)) AS INTEGER) "
                "FROM translation_cache"
            )
            oldest_days = cursor.fetchone()[0]

            return {
                "total_entries": total_count,
                "oldest_entry_days": oldest_days if oldest_days else 0,
            }
    except sqlite3.Error as e:
        logger.error(f"Failed to get translation cache stats: {e}")
        return {"total_entries": 0, "oldest_entry_days": 0}


def get_translation_cache_hit_rate() -> float:
    """Returns the translation cache hit rate (hits / (hits + misses)).

    If there have been no cache lookups, returns 0.0.
    """
    total = _cache_hits + _cache_misses
    if total == 0:
        return 0.0
    return _cache_hits / total


def reset_translation_cache_counters() -> None:
    """Reset the cache hits and misses counters to zero."""
    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0
