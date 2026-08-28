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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """
        )

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
            )  # nosec

        conn.commit()
        logger.info("Multi-tenancy migration completed successfully.")
    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()
