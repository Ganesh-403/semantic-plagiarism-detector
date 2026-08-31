"""
src/db/multimedia_forensics_db.py
---------------------------------
SQLite database manager for Multimedia Forensics Logs.
"""

import sqlite3, logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/multimedia_forensics.db")


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


def initialize_multimedia_forensics_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS multimedia_forensics_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, media_a_id TEXT NOT NULL, media_b_id TEXT NOT NULL,
            dubbing_probability REAL NOT NULL, is_dubbed INTEGER NOT NULL, analyzed_at TEXT NOT NULL)"""
        )


def log_av_forensics(
    media_a_id: str,
    media_b_id: str,
    prob: float,
    is_dubbed: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO multimedia_forensics_logs (media_a_id, media_b_id, dubbing_probability, is_dubbed, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    media_a_id,
                    media_b_id,
                    prob,
                    1 if is_dubbed else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log AV forensics: %s", e)
        return False
