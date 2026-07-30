"""
src/db/translation_cache.py
---------------------------
SQLite cache for translation API requests to preserve API quota.
Maps SHA-256 hash of (foreign_text, source_lang, target_lang) -> cached_text.
"""

import hashlib
import os
import sqlite3
import tempfile
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "corpus.db")


def _init_db():
    """Initializes the translation cache table if it does not exist."""
    path = DB_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
    except (sqlite3.OperationalError, OSError, PermissionError):
        path = os.path.join(tempfile.gettempdir(), "semantic_plagiarism_detector", "data", "corpus.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)

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




def _hash_text(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> str:
    """Generates a unique SHA-256 hash for a given text and language pair."""
    key = f"{source_lang}:{target_lang}:{text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_cached_translation(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> Optional[str]:
    """Retrieves cached translation if available."""
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
        return row[0] if row else None


def cache_translation(
    foreign_text: str,
    translated_text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
) -> None:
    """Stores a new translation in the SQLite cache."""
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
        