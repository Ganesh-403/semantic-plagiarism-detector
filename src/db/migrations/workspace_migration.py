from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def migrate_to_multitenancy(db_path: str) -> None:
    """Idempotent migration adding workspaces table and scoping existing tables by workspace_id."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Create workspaces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)

        # 2. Insert default workspace if not present
        default_ws_id = "default-workspace-id"
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT OR IGNORE INTO workspaces (id, name, created_at, is_active)
            VALUES (?, ?, ?, 1)
        """,
            (default_ws_id, "Default Organization", now),
        )

        # 3. Add workspace_id columns to existing tables if missing
        tables = ["document_corpus", "incidents", "users", "translation_cache"]
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            if "workspace_id" not in columns:
                logger.info("Adding workspace_id column to %s", table)
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN workspace_id TEXT DEFAULT '{default_ws_id}'"
                )

        # 4. Backfill existing null records into default workspace
        for table in tables:
            cursor.execute(
                f"UPDATE {table} SET workspace_id = ? WHERE workspace_id IS NULL",
                (default_ws_id,),
            )

        conn.commit()
        logger.info("Multi-tenancy migration completed successfully.")
    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()
