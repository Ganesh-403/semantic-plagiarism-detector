"""Unit tests for src/utils/storage_metrics.py."""

from pathlib import Path
from src.utils.storage_metrics import (
    calculate_storage_usage,
    get_faiss_index_paths,
    get_sqlite_db_paths,
)


def test_calculate_storage_usage_empty(tmp_path: Path) -> None:
    """Test calculate_storage_usage with empty file lists or non-existent files."""
    db_file = tmp_path / "nonexistent.db"
    index_file = tmp_path / "nonexistent.index"

    usage = calculate_storage_usage(db_paths=[db_file], index_paths=[index_file])

    assert usage["sqlite_bytes"] == 0
    assert usage["faiss_bytes"] == 0
    assert usage["total_bytes"] == 0
    assert usage["sqlite_mb"] == 0.0
    assert usage["faiss_mb"] == 0.0
    assert usage["total_mb"] == 0.0
    assert usage["formatted_total"] == "0.00 MB"
    assert usage["formatted_sqlite"] == "0.00 MB"
    assert usage["formatted_faiss"] == "0.00 MB"


def test_calculate_storage_usage_with_dummy_files(tmp_path: Path) -> None:
    """Test calculate_storage_usage with created files of known sizes."""
    db_file = tmp_path / "test_corpus.db"
    index_file = tmp_path / "test_corpus.index"

    # Write 1 MB (1024 * 1024 bytes) to db_file
    db_file.write_bytes(b"0" * (1024 * 1024))
    # Write 0.5 MB (512 * 1024 bytes) to index_file
    index_file.write_bytes(b"0" * (512 * 1024))

    usage = calculate_storage_usage(db_paths=[db_file], index_paths=[index_file])

    assert usage["sqlite_bytes"] == 1024 * 1024
    assert usage["faiss_bytes"] == 512 * 1024
    assert usage["total_bytes"] == (1024 + 512) * 1024
    assert usage["sqlite_mb"] == 1.0
    assert usage["faiss_mb"] == 0.5
    assert usage["total_mb"] == 1.5
    assert usage["formatted_sqlite"] == "1.00 MB"
    assert usage["formatted_faiss"] == "0.50 MB"
    assert usage["formatted_total"] == "1.50 MB"


def test_get_sqlite_db_paths() -> None:
    """Test get_sqlite_db_paths returns a list of Path objects."""
    paths = get_sqlite_db_paths()
    assert isinstance(paths, list)
    for p in paths:
        assert isinstance(p, Path)


def test_get_faiss_index_paths() -> None:
    """Test get_faiss_index_paths returns a list of Path objects."""
    paths = get_faiss_index_paths()
    assert isinstance(paths, list)
    for p in paths:
        assert isinstance(p, Path)
