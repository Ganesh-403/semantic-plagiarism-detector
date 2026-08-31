"""
src/db/federation_registry_db.py
--------------------------------
SQLite database manager for Federated Plagiarism Detection Registry.

Manages trusted institutional nodes and persists imported LSH bands
for cross-institutional plagiarism queries.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/federation_registry.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for SQLite connections."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_federation_db(db_path: Optional[Path] = None) -> None:
    """Create the federation registry database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trusted_nodes (
                institution_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                public_key_hash TEXT,
                added_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS federated_signatures (
                document_id TEXT PRIMARY KEY,
                institution_id TEXT NOT NULL,
                lsh_bands_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (institution_id) REFERENCES trusted_nodes(institution_id)
            )
        """
        )

    logger.info(
        "Federation registry database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def register_trusted_node(
    institution_id: str, name: str, public_key_hash: str, db_path: Optional[Path] = None
) -> bool:
    """Register a new trusted institutional node."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trusted_nodes 
                (institution_id, name, public_key_hash, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (institution_id, name, public_key_hash, datetime.utcnow().isoformat()),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to register node %s: %s", institution_id, e)
        return False


def store_federated_signature(
    document_id: str,
    institution_id: str,
    lsh_bands: list[str],
    db_path: Optional[Path] = None,
) -> bool:
    """Store imported LSH bands from a trusted node."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO federated_signatures 
                (document_id, institution_id, lsh_bands_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_id,
                    institution_id,
                    json.dumps(lsh_bands),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to store federated signature for %s: %s", document_id, e)
        return False
