import os
from unittest.mock import MagicMock, patch

from src.core.synchronization import _backup_corrupted_index, verify_and_repair_index

# ---------------------------------------------------------------------------
# Test Verification & Desync Scenarios
# ---------------------------------------------------------------------------


def test_verify_and_repair_index_perfect_match():
    """
    Test that when FAISS index exists and its count perfectly matches SQLite,
    no rebuild is triggered.
    """
    with (
        patch("os.path.exists", return_value=True),
        patch("src.core.synchronization.load_index") as mock_load,
        patch("src.core.synchronization.get_embedding_count", return_value=500),
        patch("src.core.synchronization._rebuild_index") as mock_rebuild,
    ):
        mock_index = MagicMock()
        mock_index.ntotal = 500
        mock_load.return_value = mock_index

        verify_and_repair_index("/fake/path.index")

        # Rebuild should NOT be called
        mock_rebuild.assert_not_called()


def test_verify_and_repair_index_missing_faiss():
    """
    Test that if the FAISS index file is entirely missing, a full rebuild is triggered.
    """
    with (
        patch("os.path.exists", return_value=False),
        patch("src.core.synchronization._rebuild_index") as mock_rebuild,
    ):
        verify_and_repair_index("/fake/path.index")

        # Rebuild MUST be called
        mock_rebuild.assert_called_once_with("/fake/path.index")


def test_verify_and_repair_index_desync():
    """
    Test that if FAISS count and DB count mismatch (e.g. server crash during ingestion),
    a full rebuild is triggered automatically.
    """
    with (
        patch("os.path.exists", return_value=True),
        patch("src.core.synchronization.load_index") as mock_load,
        patch("src.core.synchronization.get_embedding_count", return_value=1200),
        patch("src.core.synchronization._backup_corrupted_index") as mock_backup,
        patch("src.core.synchronization._rebuild_index") as mock_rebuild,
    ):
        mock_index = MagicMock()
        mock_index.ntotal = 1150  # 50 vectors lost during crash
        mock_load.return_value = mock_index

        verify_and_repair_index("/fake/path.index")

        # Backup and Rebuild MUST be called
        mock_backup.assert_called_once_with("/fake/path.index")
        mock_rebuild.assert_called_once_with("/fake/path.index")


def test_verify_and_repair_index_load_failure():
    """
    Test that if FAISS throws an exception during loading (e.g., corrupted file),
    the system catches it and forces a complete rebuild.
    """
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "src.core.synchronization.load_index",
            side_effect=Exception("Corrupted EOF"),
        ),
        patch("src.core.synchronization._backup_corrupted_index") as mock_backup,
        patch("src.core.synchronization._rebuild_index") as mock_rebuild,
    ):
        verify_and_repair_index("/fake/path.index")

        # Backup and Rebuild MUST be called due to exception
        mock_backup.assert_called_once_with("/fake/path.index")
        mock_rebuild.assert_called_once_with("/fake/path.index")


# ---------------------------------------------------------------------------
# Test Internal Mechanics
# ---------------------------------------------------------------------------


def test_rebuild_index_process():
    """
    Test the internal rebuild process accurately fetches SQLite blobs and generates a new FAISS index.
    """
    from src.core.synchronization import _rebuild_index

    with (
        patch("src.core.synchronization.get_all_embeddings") as mock_get_embs,
        patch("src.core.synchronization.build_index_from_matrix") as mock_build,
        patch("src.core.synchronization.save_index") as mock_save,
    ):
        mock_matrix = MagicMock()
        mock_get_embs.return_value = mock_matrix

        mock_index = MagicMock()
        mock_index.ntotal = 999
        mock_build.return_value = mock_index

        _rebuild_index("/fake/path.index")

        mock_get_embs.assert_called_once()
        mock_build.assert_called_once_with(mock_matrix)
        mock_save.assert_called_once_with(mock_index, "/fake/path.index")


def test_backup_corrupted_index_mechanics():
    """
    Ensure the backup mechanic copies the file to the 'backups' directory with a timestamp.
    """
    with (
        patch("os.path.exists", side_effect=[True, False]),
        patch("os.makedirs") as mock_makedirs,
        patch("shutil.copy2") as mock_copy,
        patch("os.listdir", return_value=[]),
        patch("src.core.synchronization.datetime") as mock_dt,
    ):
        # Mock datetime so we get a consistent timestamp
        mock_dt.now.return_value.strftime.return_value = "20240101_120000"

        _backup_corrupted_index("/fake/data/corpus.index")

        mock_makedirs.assert_called_once_with(os.path.normpath("/fake/data/backups"))
        expected_dest = os.path.normpath(
            "/fake/data/backups/corpus_20240101_120000.index.bak"
        )
        mock_copy.assert_called_once_with("/fake/data/corpus.index", expected_dest)


def test_backup_corrupted_index_retention_cap(tmp_path):
    """Keep at most 5 .index.bak files after a new backup is created."""
    import time

    index_path = tmp_path / "corpus.index"
    index_path.write_bytes(b"index-data")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    base = time.time() - 1000
    for i in range(5):
        path = backup_dir / f"corpus_old{i}.index.bak"
        path.write_bytes(b"old")
        os.utime(path, (base + i, base + i))

    _backup_corrupted_index(str(index_path))

    backups = sorted(backup_dir.glob("*.index.bak"))
    assert len(backups) == 5
    assert not (backup_dir / "corpus_old0.index.bak").exists()


def test_atexit_graceful_shutdown_registered():
    """Verify that background_tasks.shutdown is registered with atexit."""
    import atexit

    from src.core.synchronization import background_tasks

    found = False
    for handler in atexit._exithandlers:
        # atexit handlers are tuples of (func, args, kwargs)
        func, args, kwargs = handler[0], handler[1], handler[2]  # noqa: F841
        if func == background_tasks.shutdown:
            assert kwargs.get("wait") is True
            found = True
            break

    assert found, "Graceful shutdown callback was not registered with atexit"
