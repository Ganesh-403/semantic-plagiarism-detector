"""src/utils/storage_metrics.py - Disk usage calculation for SQLite databases and FAISS index."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_sqlite_db_paths() -> List[Path]:
    """Retrieve unique paths of SQLite database files in standard application locations."""
    paths: List[Path] = []

    # 1. Corpus DB path
    try:
        from src.db.corpus_db import get_corpus_db_path
        paths.append(get_corpus_db_path())
    except Exception:
        pass

    # 2. Auth DB path
    try:
        from src.db.auth import _DB_PATH as auth_db_path
        paths.append(Path(auth_db_path))
    except Exception:
        pass

    # 3. Incidents DB path
    try:
        from src.db.incidents import DEFAULT_DB_PATH as incidents_db_path
        paths.append(Path(incidents_db_path))
    except Exception:
        pass

    # 4. Search root and data directories for additional .db files
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    for folder in [base_dir, data_dir]:
        if folder.exists():
            for file_path in folder.glob("*.db"):
                paths.append(file_path)

    # Deduplicate resolved absolute paths
    unique_paths: List[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(p)
        except Exception:
            pass

    return unique_paths


def get_faiss_index_paths() -> List[Path]:
    """Retrieve unique paths of FAISS index files in standard application locations."""
    paths: List[Path] = []

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    # Default corpus.index
    paths.append(base_dir / "corpus.index")
    paths.append(data_dir / "corpus.index")

    for folder in [base_dir, data_dir]:
        if folder.exists():
            for file_path in folder.glob("*.index"):
                paths.append(file_path)

    unique_paths: List[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(p)
        except Exception:
            pass

    return unique_paths


def calculate_storage_usage(
    db_paths: Optional[List[Path]] = None,
    index_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
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
    """
    if db_paths is None:
        db_paths = get_sqlite_db_paths()
    if index_paths is None:
        index_paths = get_faiss_index_paths()

    sqlite_bytes = 0
    for db_path in db_paths:
        try:
            if db_path.exists() and db_path.is_file():
                sqlite_bytes += db_path.stat().st_size
        except OSError:
            pass

    faiss_bytes = 0
    for idx_path in index_paths:
        try:
            if idx_path.exists() and idx_path.is_file():
                faiss_bytes += idx_path.stat().st_size
        except OSError:
            pass

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
    }
