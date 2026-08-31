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
database_backup.py
------------------
Consistent SQLite database download helpers and retention management.

This module provides utilities for creating transactionally consistent
snapshots of SQLite databases and managing the lifecycle of backup files
to prevent disk space exhaustion.

Recent Additions (Issue #465):
- Added `cleanup_old_backups` function to enforce retention policies
  (max backups count and max age in days).

Recent Additions (Issue #468):
- Added `create_password_protected_backup` function to wrap a snapshot
  in an optionally AES-256-encrypted ZIP archive.

Recent Additions (Issue #932):
- Added `optimize_database` function to run PRAGMA optimize, VACUUM, and ANALYZE.

Recent Additions (Issue #1047):
- Added `get_database_size_bytes` helper returning the on-disk size of a
  SQLite database file (0 if the file does not exist).
- Added `get_total_database_size_bytes` convenience that sums the size of
  every path in `src.core.app_config.HEALTHZ_DB_PATHS` (corpus.db +
  users.db).

Recent Additions (Issue #1156):
- Added `get_database_table_stats` function returning a dictionary mapping
  each table name to its row count, plus a special '_table_count' key.

Recent Additions (Issue #1885):
- Added explicit file existence check in `create_database_backup` before
  copying or compressing.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Optional, Union

from src.core.app_config import get_backup_dir
from src.db.connection import DEFAULT_SQLITE_TIMEOUT, apply_busy_timeout
from src.db.corpus_db import get_corpus_db_path

# ── Logger Configuration ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SQLITE_HEADER = b"SQLite format 3\x00"
DEFAULT_BACKUP_DIRECTORY = get_backup_dir()

# Maintenance operations open their own connections rather than going through
# src.db.connection.create_connection(), because they need isolation_level=None
# for VACUUM. They still share the busy-timeout helper so that the timeout a
# connection is opened with is the timeout SQLite actually enforces.
OPTIMIZE_TIMEOUT_SECONDS: float = 5.0
CHECKPOINT_TIMEOUT_SECONDS: float = 10.0


class BackupRestoreSecurityError(Exception):
    """Raised when a backup fails pre-restore security validation.

    Inherits from Exception (not ValueError) so callers catching ValueError
    do not accidentally suppress security errors.
    """


_ALLOWED_DB_DIR = Path(__file__).parent.parent.parent.resolve()


def _resolve_safe_path(db_path: str | Path) -> Path:
    """Resolve path and reject anything outside the project root."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_relative_to(_ALLOWED_DB_DIR):
        raise ValueError(f"db_path is outside the allowed directory: {path}")
    return path


def iter_sqlite_snapshot_chunks(
    database_path: str | Path,
    chunk_size: int = 64 * 1024,
) -> Generator[bytes, None, None]:
    """
    Yield transactionally consistent SQLite snapshot bytes in chunks directly from a temporary file on disk.

    SQLite's online backup API is used instead of reading a live database
    file directly into memory. This includes committed pages correctly even when the
    source database uses WAL journaling, while preventing high RAM usage spikes.

    Args:
        database_path: Path to the source SQLite database.
        chunk_size: Chunk size in bytes (default: 64KB).

    Yields:
        bytes: Chunks of raw SQLite snapshot bytes.

    Raises:
        FileNotFoundError: If the source database does not exist.
        IsADirectoryError: If the source path is a directory.
        sqlite3.DatabaseError: If the generated backup is invalid.
    """
    source_path = Path(database_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {source_path}")
    if not source_path.is_file():
        raise IsADirectoryError(f"SQLite database path is not a file: {source_path}")

    with tempfile.TemporaryDirectory(
        prefix="semantic-plagiarism-backup-"
    ) as temporary_directory:
        snapshot_path = Path(temporary_directory) / source_path.name
        source_uri = f"{source_path.as_uri()}?mode=ro"

        with closing(
            sqlite3.connect(
                source_uri,
                uri=True,
                check_same_thread=False,
            )
        ) as source_connection:
            apply_busy_timeout(source_connection, DEFAULT_SQLITE_TIMEOUT)
            with closing(sqlite3.connect(snapshot_path)) as destination:
                source_connection.backup(destination)

        with open(snapshot_path, "rb") as f:
            header = f.read(len(SQLITE_HEADER))
            if header != SQLITE_HEADER:
                raise sqlite3.DatabaseError(
                    "Generated backup is not a valid SQLite database."
                )
            f.seek(0)
            while chunk := f.read(chunk_size):
                yield chunk


def create_sqlite_snapshot(
    database_path: str | Path,
    check_integrity: bool = False,
) -> bytes:
    """
    Return a transactionally consistent SQLite snapshot.

    SQLite's online backup API is used instead of reading a live database
    file directly. This includes committed pages correctly even when the
    source database uses WAL journaling.

    Args:
        database_path: Path to the source SQLite database.
        check_integrity: If True, checks the integrity of the source database
                         using PRAGMA quick_check before creating a snapshot.

    Returns:
        bytes: The raw bytes of the SQLite snapshot.

    Raises:
        FileNotFoundError: If the source database does not exist.
        IsADirectoryError: If the source path is a directory.
        sqlite3.DatabaseError: If the integrity check fails or the generated backup is invalid.
    """
    if check_integrity:
        source_path = Path(database_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"SQLite database does not exist: {source_path}")
        if not source_path.is_file():
            raise IsADirectoryError(
                f"SQLite database path is not a file: {source_path}"
            )

        source_uri = f"{source_path.as_uri()}?mode=ro"
        with closing(
            sqlite3.connect(
                source_uri,
                uri=True,
                check_same_thread=False,
            )
        ) as source_connection:
            apply_busy_timeout(source_connection, DEFAULT_SQLITE_TIMEOUT)
            cursor = source_connection.cursor()
            try:
                cursor.execute("PRAGMA quick_check;")
                result = cursor.fetchone()
                if not result or result[0] != "ok":
                    details = result[0] if result else "Unknown error"
                    raise sqlite3.DatabaseError(
                        f"Database integrity check failed: {details}"
                    )
            except sqlite3.DatabaseError as exc:
                if "Database integrity check failed" not in str(exc):
                    raise sqlite3.DatabaseError(
                        f"Database integrity check failed: {exc}"
                    ) from exc
                raise
            finally:
                cursor.close()

    return b"".join(iter_sqlite_snapshot_chunks(database_path))


def get_database_file_size_bytes(db_path: str | Path) -> int:
    """Return the file size in bytes, or 0 if the file does not exist."""
    path = _resolve_safe_path(db_path)
    return path.stat().st_size if path.is_file() else 0


def create_corpus_database_snapshot() -> bytes:
    """Return a downloadable snapshot of the configured corpus DB."""
    return create_sqlite_snapshot(get_corpus_db_path())


def iter_corpus_database_snapshot_chunks(
    chunk_size: int = 64 * 1024,
) -> Generator[bytes, None, None]:
    """Yield chunks of a downloadable snapshot of the configured corpus DB."""
    return iter_sqlite_snapshot_chunks(get_corpus_db_path(), chunk_size=chunk_size)


def create_database_backup(
    database_path: str | Path,
    *,
    backup_dir: str | Path = DEFAULT_BACKUP_DIRECTORY,
    compress_backup: bool = True,
) -> Path:
    """Write an on-disk backup file for the given SQLite database.

    When ``compress_backup`` is True (default), the snapshot bytes are
    streamed through ``gzip.GzipFile`` and written as a ``.db.gz`` file,
    cutting backup storage footprint by roughly 70%. When False, a plain
    ``.db`` copy is written instead (issue #1488).

    Raises:
        FileNotFoundError: If the source database file does not exist on disk.
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Source database file does not exist: {database_path}")

    source_name = Path(database_path).name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    destination_dir = Path(backup_dir).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)

    backup_path = None
    try:
        if compress_backup:
            backup_path = destination_dir / f"{source_name}.{timestamp}.db.gz"
            compression_level = int(os.getenv("BACKUP_GZIP_COMPRESSION_LEVEL", "6"))
            with gzip.GzipFile(
                backup_path, "wb", compresslevel=compression_level
            ) as gz_file:
                for chunk in iter_sqlite_snapshot_chunks(database_path):
                    gz_file.write(chunk)
        else:
            backup_path = destination_dir / f"{source_name}.{timestamp}.db"
            with open(backup_path, "wb") as f:
                for chunk in iter_sqlite_snapshot_chunks(database_path):
                    f.write(chunk)
    except Exception:
        if backup_path and os.path.exists(backup_path):
            try:
                os.unlink(backup_path)
            except OSError:
                pass
        raise

    try:
        os.chmod(backup_path, 0o600)
    except OSError:
        pass

    return backup_path


def get_database_size_bytes(db_path: str | Path) -> int:
    """Return the size of a SQLite database file in bytes.

    Acceptance criteria (issue #1047):
    - Returns the on-disk file size in bytes for an existing database.
    - Returns ``0`` when the database file does not exist.

    The function resolves the path with :meth:`Path.expanduser` /
    :meth:`Path.resolve` so ``~`` and relative paths behave predictably.
    It intentionally does **not** raise for missing files — admin dashboards
    and ``/healthz``-style probes need a numeric value they can render
    without try/except noise. A non-existent DB simply contributes ``0``
    to any aggregate total.

    Args:
        db_path: Path to the SQLite database file (``corpus.db`` or
            ``users.db``). Accepts ``str`` or :class:`~pathlib.Path`.

    Returns:
        File size in bytes, or ``0`` if the file does not exist.

    Raises:
        OSError: Propagated only for genuine filesystem errors that are
            *not* "file not found" (e.g. permission denied on a parent
            directory). ``FileNotFoundError`` is swallowed and mapped to
            ``0``.

    Example:
        >>> from src.db.database_backup import get_database_size_bytes
        >>> from src.db.corpus_db import get_corpus_db_path
        >>> size = get_database_size_bytes(get_corpus_db_path())
        >>> print(f"corpus.db is {size / 1024:.1f} KB")
    """
    resolved_path = Path(db_path).expanduser()

    try:
        return resolved_path.stat().st_size
    except FileNotFoundError:
        logger.debug(
            "Database file does not exist (size reported as 0): %s",
            resolved_path,
        )
        return 0


def get_total_database_size_bytes() -> int:
    """Return the combined on-disk size of all production SQLite databases.

    Sums the file sizes of every path in
    :data:`src.core.app_config.HEALTHZ_DB_PATHS` (currently ``corpus.db``
    and ``users.db``) using :func:`get_database_size_bytes`. Missing
    files contribute ``0``, so this never raises for an unprovisioned
    environment.

    Returns:
        Total bytes consumed by all configured SQLite databases.
    """
    from src.core.app_config import HEALTHZ_DB_PATHS

    return sum(get_database_size_bytes(path) for path in HEALTHZ_DB_PATHS)


def get_database_table_stats(db_path: str | Path) -> dict[str, int]:
    """Return a dictionary mapping each table name to its row count.

    Inspects the SQLite database at the given path, queries
    ``sqlite_master`` for all user-defined table names, then counts the
    rows in each table. The result is useful for health reporting and
    monitoring database utilization across deployments.

    Args:
        db_path: Path to the SQLite database file. Accepts ``str`` or
            :class:`~pathlib.Path`. Relative paths and ``~`` are
            expanded automatically.

    Returns:
        A dictionary where:
        - Each key is a table name (string) and its value is the row
          count (int) for that table.
        - The special key ``'_table_count'`` holds the total number of
          user-defined tables found in the database.
        - If the database does not exist or cannot be read, returns
          ``{'_table_count': 0}``.

    Examples:
        >>> stats = get_database_table_stats("data/corpus.db")
        >>> print(stats["_table_count"])
        4
        >>> print(stats["documents"])
        42

        >>> get_database_table_stats("/nonexistent.db")
        {'_table_count': 0}

    Issue traceability:
        Originally added under issue #1156. Issue #1773 requests the same
        helper with the same acceptance criteria; regression tests in
        ``TestGetDatabaseTableStatsIssue1773`` lock in the contract.
    """
    resolved_path = Path(db_path).expanduser().resolve()

    if not resolved_path.exists():
        logger.debug(
            "get_database_table_stats: database does not exist at %s, "
            "returning empty stats.",
            resolved_path,
        )
        return {"_table_count": 0}

    if not resolved_path.is_file():
        logger.warning(
            "get_database_table_stats: path is not a file: %s, returning empty stats.",
            resolved_path,
        )
        return {"_table_count": 0}

    stats: dict[str, int] = {}

    try:
        with closing(
            sqlite3.connect(
                str(resolved_path),
                check_same_thread=False,
            )
        ) as connection:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                try:
                    count_cursor = connection.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'  # nosec
                    )
                    row = count_cursor.fetchone()
                    row_count = int(row[0]) if row else 0
                    stats[table_name] = row_count

                except sqlite3.Error as exc:
                    logger.warning(
                        "get_database_table_stats: failed to count rows "
                        "for table '%s': %s",
                        table_name,
                        exc,
                    )
                    stats[table_name] = 0

    except sqlite3.Error as exc:
        logger.error(
            "get_database_table_stats: failed to open database at %s: %s",
            resolved_path,
            exc,
        )
        return {"_table_count": 0}

    except OSError as exc:
        logger.error(
            "get_database_table_stats: filesystem error reading %s: %s",
            resolved_path,
            exc,
        )
        return {"_table_count": 0}

    stats["_table_count"] = len(table_names)

    logger.debug(
        "get_database_table_stats: found %d table(s) in %s",
        len(table_names),
        resolved_path,
    )

    return stats


def get_table_schema_info(db_path: str | Path, table_name: str) -> list[dict]:
    """Return column metadata for the given table in a SQLite database.

    Executes ``PRAGMA table_info([table_name])`` and returns one dictionary
    per column with the keys ``name``, ``type``, ``notnull``, ``dflt_value``
    and ``pk``. The ``notnull`` and ``pk`` values are ``0``/``1`` flags as
    reported by SQLite.

    Args:
        db_path: Path to the SQLite database file. Accepts ``str`` or
            :class:`~pathlib.Path`. Relative paths and ``~`` are expanded
            automatically.
        table_name: Name of the table to inspect. Must be a plain SQL
            identifier (letters, digits and underscores) to prevent SQL
            injection.

    Returns:
        A list of dictionaries describing each column of the table. Returns
        an empty list when the database file does not exist, the path is not
        a file, the table name is unsafe, or the table has no columns.

    Example:
        >>> from src.db.database_backup import get_table_schema_info
        >>> get_table_schema_info("data/corpus.db", "documents")
        [{'name': 'id', 'type': 'INTEGER', 'notnull': 0, 'dflt_value': None, 'pk': 1}]
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
        logger.warning(
            "get_table_schema_info: refusing unsafe table name %r.",
            table_name,
        )
        return []

    resolved_path = Path(db_path).expanduser().resolve()

    if not resolved_path.exists() or not resolved_path.is_file():
        logger.debug(
            "get_table_schema_info: database file not found at %s.",
            resolved_path,
        )
        return []

    try:
        with closing(
            sqlite3.connect(str(resolved_path), check_same_thread=False)
        ) as connection:
            cursor = connection.execute(f"PRAGMA table_info([{table_name}])")
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error(
            "get_table_schema_info: failed to inspect table %r in %s: %s",
            table_name,
            resolved_path,
            exc,
        )
        return []

    return [
        {
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "dflt_value": row[4],
            "pk": row[5],
        }
        for row in rows
    ]


def create_password_protected_backup(
    snapshot_bytes: bytes,
    password: str | None = None,
    *,
    archive_name: str = "corpus.db",
) -> bytes:
    """Wrap snapshot bytes in a ZIP archive, optionally AES-256-encrypted.

    When *password* is provided the archive uses AES-256 encryption via
    ``pyzipper``. Without a password a standard (unencrypted) ZIP is
    created with ``zipfile``.

    Args:
        snapshot_bytes: Raw bytes of the SQLite snapshot.
        password: Optional encryption password. ``None`` or empty string
            produces an unencrypted ZIP.
        archive_name: Filename used for the entry inside the ZIP.

    Returns:
        The complete ZIP archive as raw bytes.
    """
    buf = io.BytesIO()
    if password:
        import pyzipper

        with pyzipper.AESZipFile(
            buf,
            "w",
            compression=pyzipper.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES,
        ) as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.writestr(archive_name, snapshot_bytes)
    else:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(archive_name, snapshot_bytes)
    return buf.getvalue()


def _resolve_authorized_backup(
    source: str | Path,
    backup_dir: str | Path,
) -> Path:
    """Resolve and validate a backup source before restoration.

    The resolved source must remain inside the resolved backup
    directory. This blocks absolute-path injection, ``..`` traversal,
    and symlinks that escape the authorized directory.
    """
    authorized_directory = Path(backup_dir).expanduser().resolve(strict=True)
    if not authorized_directory.is_dir():
        raise NotADirectoryError(
            f"Designated backup path is not a directory: {authorized_directory}"
        )

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = authorized_directory / candidate

    try:
        resolved_source = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Backup file does not exist: {candidate}") from None

    try:
        resolved_source.relative_to(authorized_directory)
    except ValueError as exc:
        raise BackupRestoreSecurityError(
            "Backup source must be inside the designated backup "
            f"directory: {authorized_directory}"
        ) from exc

    source_stat = os.stat(resolved_source, follow_symlinks=True)

    if not stat.S_ISREG(source_stat.st_mode):
        raise BackupRestoreSecurityError("Backup source must be a regular file.")

    if os.name != "nt" and (source_stat.st_mode & stat.S_IWOTH):
        raise BackupRestoreSecurityError(
            "Refusing to restore a world-writable backup file."
        )

    return resolved_source


def _validate_sqlite_backup(source: Path) -> None:
    """Verify the SQLite header and integrity before replacement."""
    with source.open("rb") as backup_file:
        header = backup_file.read(len(SQLITE_HEADER))

    if header != SQLITE_HEADER:
        raise sqlite3.DatabaseError("Backup file is not a valid SQLite database.")

    source_uri = f"{source.as_uri()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()

    if result is None or result[0] != "ok":
        details = result[0] if result else "unknown failure"
        raise sqlite3.DatabaseError(f"SQLite backup integrity check failed: {details}")


def restore(
    source: str | Path,
    *,
    backup_dir: str | Path = DEFAULT_BACKUP_DIRECTORY,
    destination: str | Path | None = None,
) -> Path:
    """Securely restore an authorized SQLite backup.

    Security validation happens before any destination file is
    modified. The source must resolve inside ``backup_dir`` and must
    not be world-writable. A valid backup is copied to a temporary
    file in the destination directory and atomically installed with
    ``os.replace``.

    Args:
        source: Backup filename or path. Relative names are resolved
            beneath ``backup_dir``.
        backup_dir: Authorized directory containing restore sources.
        destination: Live database path. Defaults to the configured
            corpus database.

    Returns:
        The resolved destination path.

    Raises:
        BackupRestoreSecurityError: For unauthorized paths, unsafe
            file types, or world-writable source files.
        FileNotFoundError: When the backup directory or source is
            missing.
        sqlite3.DatabaseError: When the backup is not a healthy SQLite
            database.
        OSError: When the atomic file replacement fails.
    """
    source_path = _resolve_authorized_backup(
        source,
        backup_dir,
    )
    _validate_sqlite_backup(source_path)

    destination_path = (
        Path(destination if destination is not None else get_corpus_db_path())
        .expanduser()
        .resolve()
    )
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if source_path == destination_path:
        raise BackupRestoreSecurityError(
            "Backup source and restore destination must differ."
        )

    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.restore-",
            suffix=".tmp",
            dir=destination_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)

        shutil.copyfile(source_path, temporary_path)
        _validate_sqlite_backup(temporary_path)

        with temporary_path.open("r+b") as restored_file:
            os.fsync(restored_file.fileno())

        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Database restored securely from %s to %s.",
        source_path,
        destination_path,
    )
    return destination_path


def restore_database_backup(
    source: str | Path,
    *,
    backup_dir: str | Path = DEFAULT_BACKUP_DIRECTORY,
    destination: str | Path | None = None,
) -> Path:
    """Descriptive alias for :func:`restore`."""
    return restore(
        source,
        backup_dir=backup_dir,
        destination=destination,
    )


def cleanup_old_backups(
    backup_dir: str | Path = DEFAULT_BACKUP_DIRECTORY,
    max_backups: int = 10,
    max_age_days: int = 30,
) -> dict[str, int]:
    """Remove stale ``.db`` backups using count and age limits."""
    backup_path = Path(backup_dir)

    if not backup_path.exists() or not backup_path.is_dir():
        logger.warning(
            "Backup directory does not exist: %s",
            backup_path,
        )
        return {
            "files_deleted": 0,
            "bytes_freed": 0,
        }

    db_files = [
        f
        for f in backup_path.iterdir()
        if f.is_file() and (f.name.endswith(".db") or f.name.endswith(".db.gz"))
    ]
    if not db_files:
        logger.info("No backup files (.db or .db.gz) found to clean up.")
        return {
            "files_deleted": 0,
            "bytes_freed": 0,
        }

    db_files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    files_deleted = 0
    bytes_freed = 0

    for index, file_path in enumerate(db_files):
        file_stat = file_path.stat()
        file_age_seconds = current_time - file_stat.st_mtime

        if index >= max_backups or file_age_seconds > max_age_seconds:
            try:
                file_path.unlink()
                files_deleted += 1
                bytes_freed += file_stat.st_size
                logger.info(
                    "Deleted stale backup: %s (age: %.1f days)",
                    file_path.name,
                    file_age_seconds / 86400,
                )
            except OSError as exception:
                logger.error(
                    "Failed to delete backup %s: %s",
                    file_path.name,
                    exception,
                )

    logger.info(
        "Backup cleanup complete. Deleted %s files, freed %s bytes.",
        files_deleted,
        bytes_freed,
    )
    return {
        "files_deleted": files_deleted,
        "bytes_freed": bytes_freed,
    }


def run_incremental_vacuum(conn: sqlite3.Connection) -> bool:
    """Execute SQLite incremental vacuum on an existing connection."""
    try:
        conn.execute("PRAGMA incremental_vacuum;")
        return True
    except sqlite3.Error as exc:
        logger.error("Incremental vacuum failed: %s", exc)
        return False


def optimize_database(db_path: str | Path) -> bool:
    """Reclaim unused SQLite pages and refresh query-planner statistics.

    A dedicated autocommit connection is opened for the maintenance task and
    always closed before this function returns. ``VACUUM`` cannot run inside
    an active transaction, so autocommit mode is used and the WAL is
    checkpointed before optimisation where possible.

    Args:
        db_path: Path to an existing SQLite database file.

    Returns:
        ``True`` when all maintenance commands complete successfully;
        otherwise ``False``. Failures are logged without deleting or replacing
        the database file.
    """
    target_path = Path(db_path).expanduser().resolve()

    if not target_path.exists():
        logger.warning(
            "Cannot optimize: database file not found at %s",
            target_path,
        )
        return False
    if not target_path.is_file():
        logger.warning(
            "Cannot optimize: database path is not a file: %s",
            target_path,
        )
        return False

    try:
        with target_path.open("rb") as database_file:
            if database_file.read(len(SQLITE_HEADER)) != SQLITE_HEADER:
                logger.error(
                    "Cannot optimize: file is not a valid SQLite database: %s",
                    target_path,
                )
                return False
    except OSError as exc:
        logger.error(
            "Cannot read database before optimization: %s",
            exc,
        )
        return False

    initial_size_bytes = target_path.stat().st_size
    logger.info(
        "Starting database optimization. Initial size: %.2f MB",
        initial_size_bytes / (1024 * 1024),
    )

    try:
        with closing(
            sqlite3.connect(
                str(target_path),
                timeout=OPTIMIZE_TIMEOUT_SECONDS,
                isolation_level=None,
            )
        ) as connection:
            apply_busy_timeout(connection, OPTIMIZE_TIMEOUT_SECONDS)
            connection.execute("PRAGMA auto_vacuum = INCREMENTAL")

            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                details = quick_check[0] if quick_check else "unknown failure"
                logger.error(
                    "Cannot optimize database because integrity check failed: %s",
                    details,
                )
                return False

            try:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.DatabaseError:
                logger.debug(
                    "WAL checkpoint was unavailable for %s",
                    target_path,
                    exc_info=True,
                )

            logger.info("Executing PRAGMA optimize...")
            connection.execute("PRAGMA optimize")

            logger.info("Executing VACUUM...")
            connection.execute("VACUUM")

            logger.info("Executing ANALYZE...")
            connection.execute("ANALYZE")

        final_size_bytes = target_path.stat().st_size
        reclaimed_bytes = max(0, initial_size_bytes - final_size_bytes)
        reduction_percentage = (
            reclaimed_bytes / initial_size_bytes * 100 if initial_size_bytes else 0.0
        )

        logger.info(
            "Database optimization completed successfully. "
            "Final size: %.2f MB. Space reclaimed: %.2f MB (%.1f%%)",
            final_size_bytes / (1024 * 1024),
            reclaimed_bytes / (1024 * 1024),
            reduction_percentage,
        )
        return True

    except sqlite3.Error as exc:
        logger.error(
            "SQLite optimization failed for %s: %s",
            target_path,
            exc,
        )
        return False
    except OSError as exc:
        logger.error(
            "File-system error during optimization of %s: %s",
            target_path,
            exc,
        )
        return False


def checkpoint_wal_log(db_path: str | Path) -> bool:
    """
    Execute PRAGMA wal_checkpoint(TRUNCATE) on the database connection and log WAL file size.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        bool: True if checkpoint was successful, False otherwise.
    """
    target_path = Path(db_path).expanduser().resolve()
    if not target_path.exists():
        logger.error("Cannot checkpoint: database does not exist: %s", target_path)
        return False
    if not target_path.is_file():
        logger.error("Cannot checkpoint: database path is not a file: %s", target_path)
        return False

    wal_path = Path(f"{target_path}-wal")

    wal_size_before = wal_path.stat().st_size if wal_path.exists() else 0
    logger.info(
        "WAL file size before checkpoint for %s: %.2f KB (%d bytes)",
        target_path.name,
        wal_size_before / 1024.0,
        wal_size_before,
    )

    try:
        with closing(
            sqlite3.connect(
                str(target_path),
                timeout=CHECKPOINT_TIMEOUT_SECONDS,
                isolation_level=None,
            )
        ) as connection:
            apply_busy_timeout(connection, CHECKPOINT_TIMEOUT_SECONDS)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        wal_size_after = wal_path.stat().st_size if wal_path.exists() else 0
        logger.info(
            "WAL file size after checkpoint for %s: %.2f KB (%d bytes)",
            target_path.name,
            wal_size_after / 1024.0,
            wal_size_after,
        )
        return True
    except sqlite3.Error as exc:
        logger.error(
            "SQLite WAL checkpoint failed for %s: %s",
            target_path,
            exc,
        )
        return False
    except OSError as exc:
        logger.error(
            "File-system error during WAL checkpoint of %s: %s",
            target_path,
            exc,
        )
        return False


def verify_backup_file(backup_path: str | Path) -> bool:
    """
    Verify the integrity of a database backup archive.

    This function checks if the backup file exists, and attempts to
    decompress/read the first 100 bytes of the file, asserting that
    the start of the decompressed content matches the SQLite file header.

    Args:
        backup_path: Path to the backup file (typically .db.gz or .db).

    Returns:
        bool: True if the file exists and is a valid SQLite database
              (or valid gzip archive of one), False otherwise.
    """
    path = Path(backup_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        logger.error(
            "Backup verification failed: file does not exist or is not a file: %s", path
        )
        return False

    try:
        with open(path, "rb") as f:
            magic = f.read(2)

        is_gzip = magic == b"\x1f\x8b"

        if is_gzip:
            with gzip.open(path, "rb") as gz:
                content = gz.read(100)
        else:
            with open(path, "rb") as f:
                content = f.read(100)

        return content.startswith(SQLITE_HEADER)

    except Exception as exc:
        logger.error("Error verifying backup file %s: %s", path, exc)
        return False
