"""
src/db/translation_cache.py
---------------------------
SQLite-backed persistent cache for cross-lingual back-translations.

This module manages the storage and retrieval of translated text chunks
to avoid redundant API calls to translation services (e.g., Google Translate,
DeepL, or local MarianMT models). By caching translations, we significantly
reduce latency and API costs during cross-lingual plagiarism detection.

Features:
- Persistent SQLite storage for source/target language pairs.
- SHA-256 hashing for efficient lookups.
- Automated TTL (Time-To-Live) cleanup to prevent unbounded database growth (Issue #2985).
- Backward compatibility with legacy cache structures.

Recent Additions:
- Issue #1956: Created translation_cache table schema with source_hash primary key.
- Issue #2985: Missing TTL cleanup for Translation Cache. Added automated purge
  mechanism to delete translations older than a specified threshold (default 30 days).
"""

import hashlib
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.app_config import _REPO_ROOT, CORPUS_DB_PATH, FALLBACK_CORPUS_DB_PATH

logger = logging.getLogger(__name__)

# ── Configuration & Paths ────────────────────────────────────────────────────

# Seed the translation cache DB path from the centralized app_config.
# ``DB_PATH`` is intentionally kept as a module-level string so that tests
# importing ``src.db.translation_cache.DB_PATH`` continue to work.
DB_PATH = str(CORPUS_DB_PATH)

# In-memory counters for lookup hits and misses (Legacy)
cache_hits = 0
cache_misses = 0

# Issue #1956 & #2985 Cache DB Path
_CACHE_DB_PATH = _REPO_ROOT / "data" / "translation_cache.db"
_lock = threading.Lock()
_CORRUPTION_MESSAGE = "database disk image is malformed"

# Default TTL for cached translations in days (Issue #2985)
DEFAULT_TTL_DAYS = 30


# ── Connection Managers ──────────────────────────────────────────────────────


@contextmanager
def _connect(db_path: Optional[Path] = None):
    """Borrow a reusable SQLite connection for the translation cache (Issue #1956)."""
    path = db_path or _CACHE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _get_connection(db_path: Optional[Path] = None):
    """Context manager for acquiring and releasing SQLite connections.

    Ensures that connections are properly closed after use, even if
    an exception occurs during database operations.

    Args:
        db_path: Path to the SQLite database file. If None, uses _CACHE_DB_PATH.

    Yields:
        An active sqlite3.Connection object.
    """
    path = db_path or _CACHE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialization ───────────────────────────────────────────────────────────


def _ensure_translation_cache_schema(db_path: Path) -> None:
    """Create the modern translation-cache schema at ``db_path`` if needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_cache (
                source_hash TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_translation_langs
            ON translation_cache(source_lang, target_lang)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_translation_created
            ON translation_cache(created_at)
            """
        )


def init_translation_cache() -> None:
    """Create the translation cache table if it does not exist (Issue #1956 & #2985)."""
    _ensure_translation_cache_schema(_CACHE_DB_PATH)
    logger.info("Translation cache initialized at %s", _CACHE_DB_PATH)


def initialize_cache_db(db_path: Optional[Path] = None) -> None:
    """Initialize the modern cache schema at the requested database path."""
    _ensure_translation_cache_schema(Path(db_path) if db_path is not None else _CACHE_DB_PATH)


# ── Hashing Helpers ──────────────────────────────────────────────────────────


def _hash_text_simple(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> str:
    """Generate a SHA-256 hash of the text and language pair for Issue #1956 cache key lookup.

    Includes source_lang and target_lang in the hash payload (f"{source_lang}:{target_lang}:{text}")
    to prevent primary key collisions when the same source text is translated to different target languages (Issue #2983).
    """
    payload = f"{source_lang}:{target_lang}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_hash(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a deterministic SHA-256 hash for a translation request.

    The hash uniquely identifies a translation based on the source text
    and the language pair. This ensures that the same text translated
    to different languages gets separate cache entries.

    Args:
        text: The source text to translate.
        source_lang: The source language code (e.g., 'es').
        target_lang: The target language code (e.g., 'en').

    Returns:
        A hex-encoded SHA-256 hash string.
    """
    # Normalize inputs to ensure consistent hashing
    normalized_text = text.strip().lower()
    normalized_source = source_lang.strip().lower()
    normalized_target = target_lang.strip().lower()

    # Combine into a single payload
    payload = f"{normalized_source}|{normalized_target}|{normalized_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def migrate_legacy_cache(
    legacy_db_path: Optional[Path] = None,
    cache_db_path: Optional[Path] = None,
) -> dict[str, int]:
    """Migrate entries from ``legacy_translation_cache`` into ``translation_cache``.

    The legacy cache uses ``_hash_text()`` and stores its source text in
    ``foreign_text``. The modern cache uses ``_generate_hash()``, which includes
    normalized source/target language codes in the cache key. This migration
    deliberately recomputes the modern hash instead of reusing ``text_hash``.

    Existing modern entries are left untouched so that a migration cannot
    overwrite newer translations. The legacy table is preserved for rollback
    and backward compatibility.

    Args:
        legacy_db_path: SQLite database containing ``legacy_translation_cache``.
            Defaults to the configured corpus database.
        cache_db_path: SQLite database containing ``translation_cache``.
            Defaults to the modern translation-cache database.

    Returns:
        A summary with ``scanned``, ``migrated``, ``skipped``, and ``errors``
        counts.
    """
    source_path = Path(legacy_db_path) if legacy_db_path is not None else Path(DB_PATH)
    target_path = (
        Path(cache_db_path) if cache_db_path is not None else _CACHE_DB_PATH
    )

    if not source_path.exists():
        logger.info("Legacy translation cache database does not exist: %s", source_path)
        return {"scanned": 0, "migrated": 0, "skipped": 0, "errors": 0}

    _ensure_translation_cache_schema(target_path)

    try:
        with sqlite3.connect(str(source_path)) as legacy_conn:
            legacy_conn.row_factory = sqlite3.Row
            try:
                rows = legacy_conn.execute(
                    """
                    SELECT text_hash, foreign_text, translated_text,
                           source_lang, target_lang, created_at
                    FROM legacy_translation_cache
                    """
                ).fetchall()
            except sqlite3.Error as exc:
                logger.error("Failed to read legacy translation cache: %s", exc)
                return {"scanned": 0, "migrated": 0, "skipped": 0, "errors": 1}

        stats = {"scanned": len(rows), "migrated": 0, "skipped": 0, "errors": 0}
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(str(target_path)) as cache_conn:
            for row in rows:
                source_text = row["foreign_text"]
                translated_text = row["translated_text"]
                source_lang = (row["source_lang"] or "auto").strip().lower()
                target_lang = (row["target_lang"] or "en").strip().lower()

                if not source_text or not translated_text:
                    stats["skipped"] += 1
                    continue

                source_hash = _generate_hash(source_text, source_lang, target_lang)
                created_at = row["created_at"] or now
                try:
                    cursor = cache_conn.execute(
                        """
                        INSERT OR IGNORE INTO translation_cache
                        (source_hash, source_text, source_lang, target_lang,
                         translated_text, created_at, last_accessed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_hash,
                            source_text,
                            source_lang,
                            target_lang,
                            translated_text,
                            str(created_at),
                            str(created_at),
                        ),
                    )
                    if cursor.rowcount == 1:
                        stats["migrated"] += 1
                    else:
                        stats["skipped"] += 1
                except sqlite3.Error as exc:
                    stats["errors"] += 1
                    logger.error(
                        "Failed to migrate legacy translation entry %s: %s",
                        row["text_hash"],
                        exc,
                    )

        logger.info(
            "Legacy translation cache migration complete: scanned=%d migrated=%d "
            "skipped=%d errors=%d",
            stats["scanned"],
            stats["migrated"],
            stats["skipped"],
            stats["errors"],
        )
        return stats
    except sqlite3.Error as exc:
        logger.error("Translation cache migration failed: %s", exc)
        return {"scanned": 0, "migrated": 0, "skipped": 0, "errors": 1}


# ── Core Cache Operations (Issue #1956 & #2985) ─────────────────────────────
def _recover_corrupted_cache() -> None:
    """Delete a corrupted cache database and recreate its schema."""
    logger.critical(
        "Translation cache database is corrupted at %s; deleting it and "
        "recreating the schema.",
        _CACHE_DB_PATH,
    )

    with _lock:
        for suffix in ("", "-wal", "-shm"):
            cache_path = (
                _CACHE_DB_PATH
                if not suffix
                else _CACHE_DB_PATH.with_name(_CACHE_DB_PATH.name + suffix)
            )
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.critical(
                    "Unable to remove corrupted translation cache file %s: %s",
                    cache_path,
                    exc,
                )
                raise

        init_translation_cache()
        logger.info("Recreated translation cache schema at %s", _CACHE_DB_PATH)


def get_cached_translation(
    source_text: str,
    source_lang: str,
    target_lang: str,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """Retrieve a cached translation if it exists.

    Args:
        source_text: The original text.
        source_lang: Source language code.
        target_lang: Target language code.
        db_path: Optional path to the database file.
        conn: Optional active SQLite database connection.

    Returns:
        The translated text string, or None if not cached.
    """
    if not source_text:
        return None

    source_hash = _generate_hash(source_text, source_lang, target_lang)

    try:
        with _lock:
            if conn is not None:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT translated_text FROM translation_cache
                    WHERE source_hash = ?
                    """,
                    (source_hash,),
                )
                row = cursor.fetchone()

                if row:
                    # Update last_accessed_at for potential LRU tracking
                    conn.execute(
                        """
                        UPDATE translation_cache 
                        SET last_accessed_at = ? 
                        WHERE source_hash = ?
                        """,
                        (datetime.utcnow().isoformat(), source_hash),
                    )
                    logger.debug(
                        "Cache hit for translation: %s -> %s", source_lang, target_lang
                    )
                    return row["translated_text"]
            else:
                with _connect(db_path) as new_conn:
                    new_conn.row_factory = sqlite3.Row
                    cursor = new_conn.execute(
                        """
                        SELECT translated_text FROM translation_cache
                        WHERE source_hash = ?
                        """,
                        (source_hash,),
                    )
                    row = cursor.fetchone()

                    if row:
                        # Update last_accessed_at for potential LRU tracking
                        new_conn.execute(
                            """
                            UPDATE translation_cache 
                            SET last_accessed_at = ? 
                            WHERE source_hash = ?
                            """,
                            (datetime.utcnow().isoformat(), source_hash),
                        )
                        logger.debug(
                            "Cache hit for translation: %s -> %s",
                            source_lang,
                            target_lang,
                        )
                        return row["translated_text"]

            return None

    except sqlite3.Error as exc:
        logger.error("Failed to query translation cache: %s", exc)
        return None
    except sqlite3.DatabaseError as exc:
        if _CORRUPTION_MESSAGE in str(exc).lower():
            _recover_corrupted_cache()
        else:
            logger.error("Failed to query translation cache: %s", exc)
        return None


def save_translation(
    source_text: str,
    source_lang: str,
    target_lang: str,
    translated_text: str,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Save a new translation to the cache.

    Args:
        source_text: The original text.
        source_lang: Source language code.
        target_lang: Target language code.
        translated_text: The resulting translation.
        db_path: Optional path to the database file.
        conn: Optional active SQLite database connection.

    Returns:
        True if saved successfully, False otherwise.
    """
    if not source_text or not translated_text:
        return False

    source_hash = _generate_hash(source_text, source_lang, target_lang)
    now = datetime.utcnow().isoformat()

    try:
        with _lock:
            if conn is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO translation_cache
                    (source_hash, source_text, source_lang, target_lang, translated_text, created_at, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_hash,
                        source_text,
                        source_lang.strip().lower(),
                        target_lang.strip().lower(),
                        translated_text.strip(),
                        now,
                        now,
                    ),
                )
            else:
                with _connect(db_path) as new_conn:
                    new_conn.execute(
                        """
                        INSERT OR REPLACE INTO translation_cache
                        (source_hash, source_text, source_lang, target_lang, translated_text, created_at, last_accessed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_hash,
                            source_text,
                            source_lang.strip().lower(),
                            target_lang.strip().lower(),
                            translated_text.strip(),
                            now,
                            now,
                        ),
                    )
        logger.debug("Saved translation to cache: %s -> %s", source_lang, target_lang)
        return True
    except sqlite3.Error as exc:
        logger.error("Failed to save translation to cache: %s", exc)
        return False


def clear_translation_cache(db_path: Optional[Path] = None) -> int:
    """Delete all entries from the translation cache.

    Returns:
        The number of rows deleted.
    """
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute("DELETE FROM translation_cache")
            deleted_count = cursor.rowcount
            logger.info("Cleared translation cache. Deleted %d rows.", deleted_count)
            return deleted_count
    except sqlite3.Error as exc:
        logger.error("Failed to clear translation cache: %s", exc)
        return 0


def purge_old_translations(
    days: int = DEFAULT_TTL_DAYS, db_path: Optional[Path] = None
) -> int:
    """Delete cached translations older than the specified threshold.

    This function implements the automated TTL cleanup mechanism required
    by Issue #2985. It prevents the translation cache database from growing
    indefinitely by removing stale entries.

    Args:
        days: The age threshold in days. Translations older than this
              will be deleted. Defaults to 30 days.
        db_path: Optional path to the database file.

    Returns:
        The number of rows deleted.
    """
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")

    cutoff_date = datetime.utcnow() - timedelta(days=days)
    cutoff_iso = cutoff_date.isoformat()

    logger.info("Purging translations older than %d days (before %s)", days, cutoff_iso)

    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM translation_cache 
                WHERE created_at < ?
                """,
                (cutoff_iso,),
            )
            deleted_count = cursor.rowcount

            logger.info("Purge complete. Deleted %d stale translations.", deleted_count)
            return deleted_count

    except sqlite3.Error as e:
        logger.error("Failed to purge old translations: %s", e)
        return 0


def get_cache_stats(db_path: Optional[Path] = None) -> dict[str, Any]:
    """Retrieve statistics about the translation cache.

    Returns:
        A dictionary containing:
        - total_entries: Total number of cached translations.
        - oldest_entry: ISO timestamp of the oldest cached translation.
        - newest_entry: ISO timestamp of the newest cached translation.
    """
    stats = {"total_entries": 0, "oldest_entry": None, "newest_entry": None}

    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM translation_cache")
            stats["total_entries"] = cursor.fetchone()["count"]

            cursor = conn.execute(
                "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM translation_cache"
            )
            row = cursor.fetchone()
            if row:
                stats["oldest_entry"] = row["oldest"]
                stats["newest_entry"] = row["newest"]

    except sqlite3.Error as e:
        logger.error("Failed to retrieve cache stats: %s", e)

    return stats


# ── Legacy Cache Functions (Backward Compatibility) ──────────────────────────


def _init_db() -> None:
    """
    Initializes the legacy translation cache table and indexes if they do not exist.

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
                CREATE TABLE IF NOT EXISTS legacy_translation_cache (
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
                CREATE INDEX IF NOT EXISTS idx_legacy_translation_cache_created_at
                ON legacy_translation_cache(created_at)
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


def get_legacy_cached_translation(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> Optional[str]:
    """
    Retrieves a cached translation from the legacy cache if available.

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
            "SELECT translated_text FROM legacy_translation_cache WHERE text_hash = ?",
            (text_hash,),
        )
        row = cursor.fetchone()
        global cache_hits, cache_misses
        if row:
            cache_hits += 1
            return row[0]
        else:
            cache_misses += 1
            return None


def cache_translation(
    foreign_text: str,
    translated_text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
) -> None:
    """
    Stores a new translation in the legacy SQLite cache.

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
            INSERT OR REPLACE INTO legacy_translation_cache
            (text_hash, foreign_text, translated_text, source_lang, target_lang)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text_hash, foreign_text, translated_text, source_lang, target_lang),
        )
        conn.commit()


def purge_expired_translation_cache(days_old: int = 60) -> int:
    """
    Purge legacy translation cache entries older than the specified number of days.

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
                DELETE FROM legacy_translation_cache
                WHERE created_at < datetime('now', '-' || ? || ' days')
                """,
                (days_old,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(
                "Purged %d expired legacy translation cache entries older than %d days.",
                deleted_count,
                days_old,
            )
            return deleted_count
    except sqlite3.Error as e:
        logger.error("Failed to purge expired translation cache: %s", e)
        return 0


def purge_translation_cache_older_than(days: int = 30) -> int:
    """
    Purge legacy translation cache entries older than the specified number of days.

    Args:
        days: The age in days after which a cache entry is considered expired.
              Defaults to 30 days.

    Returns:
        int: The number of rows successfully deleted from the cache.
    """
    _init_db()
    if days < 0:
        raise ValueError("days must be a non-negative integer.")

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    # SQLite CURRENT_TIMESTAMP defaults to UTC string format YYYY-MM-DD HH:MM:SS
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM legacy_translation_cache WHERE created_at < ?",
                (cutoff_str,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(
                "Purged %d legacy translation cache entries older than %d days.",
                deleted_count,
                days,
            )
            return deleted_count
    except sqlite3.Error as e:
        logger.error("Failed to purge translation cache: %s", e)
        return 0


def get_translation_cache_stats() -> dict[str, int]:
    """
    Get statistics about the current legacy translation cache.

    Returns:
        dict[str, int]: A dictionary containing total entries count.
    """
    _init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translation_cache")
            row = cursor.fetchone()
            total_count = row[0] if row else 0

            return {
                "total_entries": int(total_count),
            }
    except sqlite3.Error as e:
        logger.error(f"Failed to get translation cache stats: {e}")
        return {"total_entries": 0}


# Fix: Original code referenced undefined `_cache_hits` / `_cache_misses`.
# Map to the existing module-level `cache_hits` / `cache_misses` counters.
_cache_hits = cache_hits
_cache_misses = cache_misses


def get_translation_cache_hit_ratio() -> float:
    """
    Computes the translation cache hit ratio.
    """
    total = cache_hits + cache_misses
    if total == 0:
        return 0.0
    return cache_hits / total


def reset_translation_cache_counters() -> None:
    """Reset the cache hits and misses counters to zero."""
    global cache_hits, cache_misses
    cache_hits = 0
    cache_misses = 0


def get_cache_performance_summary() -> dict[str, Any]:
    """Retrieves cache lookup telemetry, including total requests, hits, misses, and hit ratio percentage.

    Returns:
        dict[str, Any]: A dictionary summary of cache performance statistics.
    """
    total = cache_hits + cache_misses
    ratio = (float(cache_hits) / total * 100.0) if total > 0 else 0.0
    return {
        "total_requests": total,
        "hits": cache_hits,
        "misses": cache_misses,
        "hit_ratio_percentage": ratio,
    }
