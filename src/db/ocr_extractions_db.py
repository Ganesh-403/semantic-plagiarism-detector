"""
src/db/ocr_extractions_db.py
----------------------------
SQLite database manager for OCR Extraction Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/ocr_extractions.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    path = db_path or DEFAULT_DB_PATH
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


def initialize_ocr_extractions_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_extraction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                block_count INTEGER NOT NULL,
                layout_coherence REAL NOT NULL,
                extracted_text_hash TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "OCR extractions database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_ocr_extraction(
    document_id: str,
    image_hash: str,
    block_count: int,
    layout_coherence: float,
    extracted_text: str,
    db_path: Optional[Path] = None,
) -> bool:
    import hashlib

    text_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO ocr_extraction_logs (document_id, image_hash, block_count, layout_coherence, extracted_text_hash, analyzed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    image_hash,
                    block_count,
                    layout_coherence,
                    text_hash,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log OCR extraction: %s", e)
        return False
