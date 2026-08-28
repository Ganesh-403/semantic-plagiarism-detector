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
Migration 006: Add password expiration support to the users table.

Adds a `password_expires_at` column to track when a user's password
will naturally expire, supporting 90-day rotation policies common
in academic IT environments (Issue #2716).
"""

import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default password lifetime in days
DEFAULT_PASSWORD_LIFETIME_DAYS = 90


def migrate(connection: sqlite3.Connection) -> None:
    """Execute the migration to add password expiration column.

    Args:
        connection: Active SQLite database connection.
    """
    logger.info("Running migration 006: Add password_expires_at column")

    # Check if column already exists to make migration idempotent
    cursor = connection.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]

    if "password_expires_at" not in columns:
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN password_expires_at TEXT
        """
        )
        logger.info("Added password_expires_at column to users table")

        # Set expiration for existing users to 90 days from now
        # This prevents immediate lockout of all existing users upon migration
        expiration_date = (
            datetime.utcnow() + timedelta(days=DEFAULT_PASSWORD_LIFETIME_DAYS)
        ).isoformat()

        connection.execute(
            """
            UPDATE users
            SET password_expires_at = ?
            WHERE password_expires_at IS NULL
        """,
            (expiration_date,),
        )

        logger.info("Set password_expires_at to %s for existing users", expiration_date)
    else:
        logger.info("password_expires_at column already exists, skipping")

    connection.commit()


def rollback(connection: sqlite3.Connection) -> None:
    """Rollback the migration (SQLite doesn't support DROP COLUMN easily).

    Note: SQLite versions prior to 3.35.0 do not support DROP COLUMN.
    For older versions, this would require recreating the table.
    For simplicity, we assume modern SQLite or accept the column remains.
    """
    logger.warning(
        "Rollback for migration 006 is not fully supported in older SQLite versions. "
        "The password_expires_at column may remain in the schema."
    )
    # In SQLite 3.35.0+:
    # connection.execute("ALTER TABLE users DROP COLUMN password_expires_at")
