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

"""src/utils/storage_metrics.py - Disk usage calculation for SQLite databases and FAISS index."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths by comparing their resolved absolute form."""
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(p)
        except Exception as e:
            logger.debug("Could not resolve path: %s", e)
    return unique_paths


def get_sqlite_db_paths() -> list[Path]:
    """Retrieve unique paths of SQLite database files in standard locations.

    Collects the three configured application databases (corpus, auth and
    incidents) plus any additional ``*.db`` files sitting in the repository
    root or in ``data/``.

    Each configured path is resolved independently and a failure to resolve
    one is logged at debug level and skipped, so a partially installed
    environment still reports usage for the databases it can see.

    Returns:
        List[Path]: Existing-or-not database paths, de-duplicated by their
        resolved absolute form. Paths are returned in discovery order.
    """
    paths: list[Path] = []

    # 1. Corpus DB path
    try:
        from src.db.corpus_db import get_corpus_db_path

        paths.append(get_corpus_db_path())
    except Exception as e:
        logger.debug("Could not resolve path: %s", e)

    # 2. Auth DB path
    try:
        from src.db.auth import get_auth_db_path

        paths.append(get_auth_db_path())
    except Exception as e:
        logger.debug("Could not resolve path: %s", e)

    # 3. Incidents DB path
    try:
        from src.db.incidents import DEFAULT_DB_PATH as incidents_db_path

        paths.append(Path(incidents_db_path))
    except Exception as e:
        logger.debug("Could not resolve path: %s", e)

    # 4. Search root and data directories for additional .db files
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    for folder in [base_dir, data_dir]:
        if folder.exists():
            for file_path in folder.glob("*.db"):
                paths.append(file_path)

    # Deduplicate resolved absolute paths
    return _deduplicate_paths(paths)


def get_faiss_index_paths() -> list[Path]:
    """Retrieve unique paths of FAISS index files in standard locations.

    Always includes the two default ``corpus.index`` locations (repository
    root and ``data/``) so a caller can report "0 bytes" for an index that has
    not been built yet, then adds any other ``*.index`` files found alongside
    them.

    Returns:
        List[Path]: Existing-or-not index paths, de-duplicated by their
        resolved absolute form. Paths are returned in discovery order.
    """
    paths: list[Path] = []

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    # Default corpus.index
    paths.append(base_dir / "corpus.index")
    paths.append(data_dir / "corpus.index")

    for folder in [base_dir, data_dir]:
        if folder.exists():
            for file_path in folder.glob("*.index"):
                paths.append(file_path)

    return _deduplicate_paths(paths)


def calculate_storage_usage(
    db_paths: Optional[list[Path]] = None,
    index_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """Calculate total SQLite + FAISS disk usage in bytes and formatted megabytes.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'sqlite_bytes': int (bytes used by SQLite files)
            - 'faiss_bytes': int (bytes used by FAISS index files)
            - 'total_bytes': int (combined bytes)
            - 'sqlite_mb': float (megabytes, rounded to 2 decimal places)
            - 'faiss_mb': float (megabytes, rounded to 2 decimal places)
            - 'total_mb': float (megabytes, rounded to 2 decimal places)
            - 'formatted_total': str (formatted total string e.g. "1.25 MB")
            - 'formatted_sqlite': str (formatted SQLite size)
            - 'formatted_faiss': str (formatted FAISS index size)
            - 'sqlite_file_count': int (number of SQLite files found)
            - 'faiss_file_count': int (number of FAISS index files found)
    """
    if db_paths is None:
        db_paths = get_sqlite_db_paths()
    if index_paths is None:
        index_paths = get_faiss_index_paths()

    sqlite_bytes = 0
    sqlite_file_count = 0
    for db_path in db_paths:
        try:
            if db_path.exists() and db_path.is_file():
                sqlite_bytes += db_path.stat().st_size
                sqlite_file_count += 1
        except OSError as e:
            logger.debug("Could not resolve path: %s", e)

    faiss_bytes = 0
    faiss_file_count = 0
    for idx_path in index_paths:
        try:
            if idx_path.exists() and idx_path.is_file():
                faiss_bytes += idx_path.stat().st_size
                faiss_file_count += 1
        except OSError as e:
            logger.debug("Could not resolve path: %s", e)

    total_bytes = sqlite_bytes + faiss_bytes

    sqlite_mb = round(sqlite_bytes / (1024 * 1024), 2)
    faiss_mb = round(faiss_bytes / (1024 * 1024), 2)
    total_mb = round(total_bytes / (1024 * 1024), 2)

    return {
        "sqlite_bytes": sqlite_bytes,
        "faiss_bytes": faiss_bytes,
        "total_bytes": total_bytes,
        "sqlite_mb": sqlite_mb,
        "faiss_mb": faiss_mb,
        "total_mb": total_mb,
        "formatted_total": f"{total_mb:.2f} MB",
        "formatted_sqlite": f"{sqlite_mb:.2f} MB",
        "formatted_faiss": f"{faiss_mb:.2f} MB",
        "sqlite_file_count": sqlite_file_count,
        "faiss_file_count": faiss_file_count,
    }


def get_directory_size_bytes(directory: Union[str, Path]) -> int:
    """Calculate total file size in bytes for a directory recursively.

    Accurately sums files in nested subdirectories while ignoring broken symlinks
    or unreadable files.

    Args:
        directory: Path or string path of directory to inspect.

    Returns:
        int: Total size of files in bytes.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return 0

    total_bytes = 0
    try:
        for file_path in dir_path.rglob("*"):
            try:
                if file_path.is_file():
                    total_bytes += file_path.stat().st_size
            except (OSError, ValueError) as e:
                logger.debug("Could not read size of %s: %s", file_path, e)
                continue
    except OSError as e:
        logger.debug("Error traversing directory %s: %s", dir_path, e)

    return total_bytes


def calculate_database_fragmentation(db_path: str) -> dict[str, float | int | str]:
    """
    Queries SQLite storage engine page allocations to evaluate structural
    fragmentation levels and identify if an analytical VACUUM routine is required.

    Returns:
        Dict detailing page counts, freelist counts, and calculated fragmentation ratio.
    """
    connection = None
    try:
        # Establish a read-only or direct cursor sequence into the target SQLite file
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # 1. Retrieve the count of empty, deleted, or unallocated database pages
        cursor.execute("PRAGMA freelist_count;")
        freelist_count: int = cursor.fetchone()[0]

        # 2. Retrieve the cumulative count of total structural database pages
        cursor.execute("PRAGMA page_count;")
        page_count: int = cursor.fetchone()[0]

        # Handle zero-allocation edge cases gracefully to avoid ZeroDivisionError logs
        if page_count == 0:
            return {
                "freelist_count": 0,
                "page_count": 0,
                "fragmentation_percentage": 0.0,
                "status": "EMPTY_DATABASE",
            }

        # Calculate fragmentation percentage based on space-recovery eligibility
        fragmentation_percentage: float = (freelist_count / page_count) * 100.0

        # Determine actionable optimization benchmarks
        # Standard administrative threshold sets optimization need at > 20% bloat
        needs_vacuum: bool = fragmentation_percentage > 20.0

        return {
            "freelist_count": freelist_count,
            "page_count": page_count,
            "fragmentation_percentage": round(fragmentation_percentage, 2),
            "status": "VACUUM_RECOMMENDED" if needs_vacuum else "OPTIMAL",
        }

    except sqlite3.Error as error:
        # Capture engine connectivity abnormalities safely
        return {
            "error": "SQLITE_QUERY_FAILURE",
            "details": str(error),
            "fragmentation_percentage": -1.0,
        }

    finally:
        if connection:
            connection.close()


def _storage_history_db_path() -> Path:
    """Default SQLite file used for daily storage snapshots."""
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "storage_history.db"


def _connect_storage_history(
    db_path: Optional[Path] = None,
) -> sqlite3.Connection:
    path = db_path or _storage_history_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_history (
            date TEXT PRIMARY KEY,
            db_size_bytes INTEGER,
            temp_size_bytes INTEGER
        )
        """
    )
    return conn


def record_storage_snapshot(db_path: Optional[Path] = None) -> None:
    """Record today's database and temp directory sizes in storage_history."""
    from src.utils.temp_manager import get_temp_directory_size_bytes

    usage = calculate_storage_usage()
    db_size_bytes = int(usage["sqlite_bytes"])
    temp_size_bytes = int(get_temp_directory_size_bytes())
    today = date.today().isoformat()

    conn = _connect_storage_history(db_path)
    try:
        conn.execute(
            """
            INSERT INTO storage_history (date, db_size_bytes, temp_size_bytes)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                db_size_bytes = excluded.db_size_bytes,
                temp_size_bytes = excluded.temp_size_bytes
            """,
            (today, db_size_bytes, temp_size_bytes),
        )
        conn.commit()
    finally:
        conn.close()


def get_projected_days_until_full(
    max_disk_bytes: int,
    db_path: Optional[Path] = None,
) -> float:
    """Estimate days until combined db+temp usage reaches max_disk_bytes.

    Uses a simple linear growth rate from the oldest to newest snapshot.
    Returns ``float('inf')`` when growth cannot be projected (too few points
    or non-positive growth). Returns ``0.0`` when usage already meets or
    exceeds the limit.
    """
    conn = _connect_storage_history(db_path)
    try:
        rows = conn.execute(
            """
            SELECT date, db_size_bytes, temp_size_bytes
            FROM storage_history
            ORDER BY date ASC
            """
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < 2:
        return float("inf")

    first_day = datetime.strptime(rows[0][0], "%Y-%m-%d").date()
    last_day = datetime.strptime(rows[-1][0], "%Y-%m-%d").date()
    elapsed_days = (last_day - first_day).days
    if elapsed_days <= 0:
        return float("inf")

    first_total = int(rows[0][1] or 0) + int(rows[0][2] or 0)
    last_total = int(rows[-1][1] or 0) + int(rows[-1][2] or 0)
    growth = last_total - first_total
    if growth <= 0:
        return float("inf")

    if last_total >= max_disk_bytes:
        return 0.0

    bytes_per_day = growth / elapsed_days
    remaining = max_disk_bytes - last_total
    return remaining / bytes_per_day


def get_storage_by_class() -> List[Dict[str, Any]]:
    """Return a per-class-section storage breakdown.

    Groups non-deleted documents by ``class_section`` (documents with a
    blank/NULL class_section are grouped under ``"Unassigned"``) and reports,
    for each group:
        - class_section: str
        - document_count: int (distinct documents in the class)
        - chunk_count: int (chunks belonging to those documents)
        - estimated_bytes: int (sum of chunk text + embedding blob sizes)

    Returns an empty list if the corpus database does not exist or cannot
    be queried.
    """
    from src.db.corpus_db import get_corpus_db_path

    db_path = get_corpus_db_path()
    results: list[dict[str, Any]] = []
    if not db_path.exists():
        return results

    try:
        connection = sqlite3.connect(str(db_path))
        try:
            rows = connection.execute(
                """
                SELECT
                    COALESCE(NULLIF(d.class_section, ''), 'Unassigned') AS class_section,
                    COUNT(DISTINCT d.filename) AS document_count,
                    COUNT(c.vector_id) AS chunk_count,
                    COALESCE(SUM(LENGTH(c.chunk_text) + LENGTH(c.embedding)), 0) AS estimated_bytes
                FROM documents d
                LEFT JOIN chunks c ON c.filename = d.filename
                WHERE d.is_deleted = 0 OR d.is_deleted IS NULL
                GROUP BY class_section
                ORDER BY estimated_bytes DESC
                """
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as e:
        logger.debug("Could not compute storage by class: %s", e)
        return results

    for class_section, document_count, chunk_count, estimated_bytes in rows:
        results.append(
            {
                "class_section": class_section,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "estimated_bytes": int(estimated_bytes),
            }
        )
    return results
