from datetime import datetime, timedelta

import numpy as np
import pytest

from src.db.corpus_db import (
    _connect,
    add_chunks,
    add_document,
    clear_all_data,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_deleted_documents,
    get_document_by_hash,
    get_document_chunks_count,
    get_document_count_by_user,
    get_document_count_fast,
    get_document_word_counts,
    get_documents_by_class,
    get_unique_class_sections,
    purge_stale_trash,
    restore_document,
    soft_delete_document,
    CorpusRepository,
)


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Uses the global mock_db fixture from conftest.py for complete DB isolation."""
    yield


def test_add_document_metadata():
    res1 = add_document("test1.pdf", "hash_abc_123")
    assert isinstance(res1, int)

    res2 = add_document("test2.pdf", "hash_abc_123")
    assert res2 == res1

    res3 = add_document("test1.pdf", "different_hash")
    assert res3 is None


def test_add_document_returns_existing_id_for_duplicate_hash(caplog):
    import logging

    hash_value = "abc1234_dup"

    with caplog.at_level(logging.INFO):
        first_id = add_document(
            filename="file1_dup.pdf",
            file_hash=hash_value,
        )

        second_id = add_document(
            filename="file2_dup.pdf",
            file_hash=hash_value,
        )

    assert second_id == first_id
    assert isinstance(first_id, int)

    with _connect() as conn:
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE file_hash = ?
            """,
            (hash_value,),
        ).fetchone()[0]

    assert count == 1
    assert "already exists in corpus; skipping insertion." in caplog.text


def test_get_document_by_hash():
    add_document("doc_alpha.txt", "hash_xyz_789")

    match = get_document_by_hash("hash_xyz_789")
    assert match == "doc_alpha.txt"

    no_match = get_document_by_hash("nonexistent_hash")
    assert no_match is None


def test_add_and_retrieve_chunks():
    add_document("doc1.pdf", "hash_1")

    dummy_emb_1 = np.ones(384, dtype=np.float32) * 0.5
    dummy_emb_2 = np.ones(384, dtype=np.float32) * 1.5

    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1 text", dummy_emb_1),
        (1, "doc1.pdf", 1, "Paragraph 2 text", dummy_emb_2),
    ]

    add_chunks(chunks)

    assert get_document_chunks_count("doc1.pdf") == 2

    registry = get_chunk_registry()
    assert len(registry) == 2
    assert registry[0].doc_name == "doc1.pdf"
    assert registry[0].chunk_text == "Paragraph 1 text"

    embs = get_all_embeddings()
    assert embs.shape == (2, 384)
    assert np.allclose(embs[0], dummy_emb_1)
    assert np.allclose(embs[1], dummy_emb_2)


def test_delete_document_cascades():
    add_document("doc1.pdf", "hash_1")
    add_document("doc2.pdf", "hash_2")

    dummy_emb = np.zeros(384, dtype=np.float32)

    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1", dummy_emb),
        (1, "doc2.pdf", 0, "Paragraph 2", dummy_emb),
    ]
    add_chunks(chunks)

    delete_document("doc1.pdf")

    all_docs = get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0]["filename"] == "doc2.pdf"

    registry = get_chunk_registry()
    assert len(registry) == 1
    assert registry[0].doc_name == "doc2.pdf"

    embs = get_all_embeddings()
    assert embs.shape == (1, 384)


def test_document_metadata_fields():
    res = add_document(
        "metadata_test.pdf",
        "hash_metadata_123",
        class_section="Class B",
        student_name="Alice Smith",
        assignment_title="Homework 1",
        detected_language="en",
    )
    assert isinstance(res, int)

    from src.db.schemas import Document

    docs = get_all_documents()
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc["filename"] == "metadata_test.pdf"
    assert doc["class_section"] == "Class B"
    assert doc["student_name"] == "Alice Smith"
    assert doc["assignment_title"] == "Homework 1"
    assert doc["detected_language"] == "en"


def test_class_queries():
    add_document(
        "doc_a.pdf",
        "hash_a",
        class_section="Class A",
        student_name="Student A",
        assignment_title="Title A",
    )
    add_document(
        "doc_b.pdf",
        "hash_b",
        class_section="Class B",
        student_name="Student B",
        assignment_title="Title B",
    )
    add_document(
        "doc_c.pdf",
        "hash_c",
        class_section="Class A",
        student_name="Student C",
        assignment_title="Title C",
    )
    add_document("doc_empty.pdf", "hash_empty")

    classes = get_unique_class_sections()
    assert "Class A" in classes
    assert "Class B" in classes
    assert len(classes) == 2

    class_a_docs = get_documents_by_class("Class A")
    assert "doc_a.pdf" in class_a_docs
    assert "doc_c.pdf" in class_a_docs
    assert len(class_a_docs) == 2

    class_b_docs = get_documents_by_class("Class B")
    assert "doc_b.pdf" in class_b_docs
    assert len(class_b_docs) == 1


def test_batch_soft_delete_documents():
    from src.db.corpus_db import _connect, batch_soft_delete_documents

    # Add some test documents
    add_document("doc_soft1.pdf", "hash_s1")
    add_document("doc_soft2.pdf", "hash_s2")
    add_document("doc_soft3.pdf", "hash_s3")

    # Fetch their IDs
    with _connect() as conn:
        rows = conn.execute("SELECT id, filename FROM documents ORDER BY id").fetchall()
        doc_ids = {row[1]: row[0] for row in rows}

    id1 = doc_ids["doc_soft1.pdf"]
    id2 = doc_ids["doc_soft2.pdf"]
    id3 = doc_ids["doc_soft3.pdf"]

    # 1. Multiple valid IDs
    count = batch_soft_delete_documents([id1, id2])
    assert count == 2

    # Check that they are deleted
    with _connect() as conn:
        deleted = conn.execute(
            "SELECT is_deleted FROM documents WHERE id IN (?, ?)", (id1, id2)
        ).fetchall()
        assert all(row[0] == 1 for row in deleted)
        not_deleted = conn.execute(
            "SELECT is_deleted FROM documents WHERE id = ?", (id3,)
        ).fetchone()
        assert not_deleted[0] == 0

    # 2. Empty list
    assert batch_soft_delete_documents([]) == 0

    # 3. Invalid/non-existing IDs
    count = batch_soft_delete_documents([9999, 10000])
    assert count == 0

    # 4. Already deleted documents
    count = batch_soft_delete_documents([id1])
    assert (
        count == 1
    )  # SQLite UPDATE rowcount still returns matched rows even if value didn't change


def test_batch_permanently_delete_documents():
    from src.db.corpus_db import _connect, batch_permanently_delete_documents

    add_document("doc_perm1.pdf", "hash_p1")
    add_document("doc_perm2.pdf", "hash_p2")
    add_document("doc_perm3.pdf", "hash_p3")

    with _connect() as conn:
        rows = conn.execute("SELECT id, filename FROM documents ORDER BY id").fetchall()
        doc_ids = {row[1]: row[0] for row in rows}

    id1 = doc_ids["doc_perm1.pdf"]
    id2 = doc_ids["doc_perm2.pdf"]
    id3 = doc_ids["doc_perm3.pdf"]

    # 1. Multiple valid IDs
    count = batch_permanently_delete_documents([id1, id2])
    assert count == 2

    # The targeted documents are hard-deleted, the rest are kept
    with _connect() as conn:
        remaining = conn.execute(
            "SELECT filename FROM documents WHERE id IN (?, ?)", (id1, id2)
        ).fetchall()
        assert remaining == []
        kept = conn.execute(
            "SELECT filename FROM documents WHERE id = ?", (id3,)
        ).fetchone()
        assert kept[0] == "doc_perm3.pdf"

    # 2. Empty list
    assert batch_permanently_delete_documents([]) == 0

    # 3. Invalid/non-existing IDs
    count = batch_permanently_delete_documents([9999, 10000])
    assert count == 0


def test_batch_permanently_delete_documents_purges_related_records():
    from src.db.corpus_db import _connect, batch_permanently_delete_documents

    add_document("doc_perm_soft.pdf", "hash_ps")
    add_document("doc_perm_other.pdf", "hash_po")

    dummy_emb = np.zeros(384, dtype=np.float32)
    add_chunks([(1, "doc_perm_soft.pdf", 0, "Paragraph 1", dummy_emb)])

    # Soft-delete moves chunks into deleted_chunks
    soft_delete_document("doc_perm_soft.pdf")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank, date_flagged, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "INC-PERM",
                "doc_perm_soft.pdf",
                "doc_perm_other.pdf",
                0.75,
                "Medium",
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ),
        )
        rows = conn.execute("SELECT id, filename FROM documents ORDER BY id").fetchall()
        doc_ids = {row[1]: row[0] for row in rows}

    soft_id = doc_ids["doc_perm_soft.pdf"]

    count = batch_permanently_delete_documents([soft_id])
    assert count == 1

    with _connect() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM documents WHERE id = ?", (soft_id,)
            ).fetchone()[0]
            == 0
        )
        # deleted_chunks has no cascade constraint, so it must be purged manually
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM deleted_chunks WHERE filename = ?",
                ("doc_perm_soft.pdf",),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM plagiarism_incidents WHERE incident_id = ?",
                ("INC-PERM",),
            ).fetchone()[0]
            == 0
        )
        # The unrelated document is untouched
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM documents WHERE filename = ?",
                ("doc_perm_other.pdf",),
            ).fetchone()[0]
            == 1
        )


def test_clear_all_data_clears_incidents(mock_db):
    # 1. Add mock documents
    add_document("doc1.pdf", "hash1")
    add_document("doc2.pdf", "hash2")

    # 2. Add mock incidents directly via SQL
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank, date_flagged, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "INC-1",
                "doc1.pdf",
                "doc2.pdf",
                0.85,
                "High",
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ),
        )

    # Verify incident exists
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM plagiarism_incidents").fetchone()[0]
        assert count == 1

    # 3. Clear all data
    clear_all_data()

    # Verify everything is cleared directly in SQLite
    assert len(get_all_documents()) == 0
    with _connect() as conn:
        count_after = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents"
        ).fetchone()[0]
        assert count_after == 0


def test_get_document_word_counts():
    clear_all_data()

    add_document("doc1.txt", "hash_doc1")
    add_document("doc2.txt", "hash_doc2")

    chunks = [
        (1, "doc1.txt", 0, "This is the first chunk.", np.zeros(384)),
        (2, "doc1.txt", 1, "And this is the second chunk of doc1.", np.zeros(384)),
        (3, "doc2.txt", 0, "Doc2 has only one single chunk.", np.zeros(384)),
    ]
    add_chunks(chunks)

    word_counts = get_document_word_counts()  # noqa: F821
    assert word_counts["doc1.txt"] == 13
    assert word_counts["doc2.txt"] == 6


def test_optimize_database_vacuum(mock_db):
    from src.db.corpus_db import optimize_database

    res = optimize_database()
    assert "size_before" in res
    assert "size_after" in res
    assert "reclaimed_bytes" in res
    assert "error" in res

    assert res["error"] is None
    assert res["size_before"] > 0
    assert res["size_after"] > 0
    assert res["reclaimed_bytes"] >= 0


def test_optimize_database_error_handling():
    from src.db.corpus_db import (
        configure_db_path,
        get_corpus_db_path,
        optimize_database,
    )

    original_path = get_corpus_db_path()
    try:
        configure_db_path("/invalid_dir_xyz_123/corpus.db")
        res = optimize_database()
        assert res["error"] is not None
        assert res["size_before"] == 0
        assert res["size_after"] == 0
        assert res["reclaimed_bytes"] == 0
    finally:
        configure_db_path(original_path)


# ==============================================================================
# NEW TESTS FOR ISSUE #834: Soft Delete and Document Restoration
# ==============================================================================


def test_soft_delete_document():
    """Verify soft-deleted documents are omitted from search/active queries and can be restored."""
    filename = "essay_student_a.pdf"
    file_hash = "hash_12345"

    inserted = add_document(
        filename=filename, file_hash=file_hash, student_name="Student A"
    )
    assert isinstance(inserted, int)

    dummy_embedding = np.random.rand(384).astype(np.float32)
    add_chunks([(0, filename, 0, "Paragraph 1 text content.", dummy_embedding)])

    active_docs = get_all_documents(include_deleted=False)
    assert len(active_docs) == 1
    assert active_docs[0]["filename"] == filename

    active_chunks = get_chunk_registry()
    assert len(active_chunks) == 1

    # Soft delete
    soft_delete_document(filename)

    # Excluded from active queries
    active_docs_after_delete = get_all_documents(include_deleted=False)
    assert len(active_docs_after_delete) == 0

    active_chunks_after_delete = get_chunk_registry()
    assert len(active_chunks_after_delete) == 0

    deleted_docs = get_deleted_documents()
    assert len(deleted_docs) == 1
    assert deleted_docs[0]["filename"] == filename

    all_docs_including_deleted = get_all_documents(include_deleted=True)
    assert len(all_docs_including_deleted) == 1
    assert all_docs_including_deleted[0]["filename"] == filename

    # Restore document
    restore_document(filename)

    active_docs_restored = get_all_documents(include_deleted=False)
    assert len(active_docs_restored) == 1
    assert active_docs_restored[0]["filename"] == filename

    restored_chunks = get_chunk_registry()
    assert len(restored_chunks) == 1
    assert restored_chunks[0].doc_name == filename
    assert restored_chunks[0].chunk_text == "Paragraph 1 text content."

    assert len(get_deleted_documents()) == 0


def test_purge_stale_trash_deletes_old_documents(mock_db):
    """Test that purge_stale_trash deletes documents older than the threshold."""
    add_document("old_trash_doc.pdf", "hash_old_trash")
    soft_delete_document("old_trash_doc.pdf")

    old_date = (datetime.now() - timedelta(days=40)).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET deleted_at = ? WHERE filename = ?",
            (old_date, "old_trash_doc.pdf"),
        )

    deleted_count = purge_stale_trash(days_in_trash=30)
    assert deleted_count == 1

    with _connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE filename = ?",
            ("old_trash_doc.pdf",),
        ).fetchone()
        assert row is None


def test_purge_stale_trash_retains_recently_deleted(mock_db):
    """Test that purge_stale_trash retains documents deleted recently."""
    add_document("recent_trash_doc.pdf", "hash_recent_trash")
    soft_delete_document("recent_trash_doc.pdf")

    deleted_count = purge_stale_trash(days_in_trash=30)
    assert deleted_count == 0

    with _connect() as conn:
        row = conn.execute(
            "SELECT is_deleted FROM documents WHERE filename = ?",
            ("recent_trash_doc.pdf",),
        ).fetchone()
        assert row is not None and row[0] == 1


def test_purge_stale_trash_ignores_active_documents(mock_db):
    """Test that purge_stale_trash does not affect active (is_deleted=0) documents."""
    add_document("active_old_doc.pdf", "hash_active_old")

    old_date = (datetime.now() - timedelta(days=100)).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET upload_date = ? WHERE filename = ?",
            (old_date, "active_old_doc.pdf"),
        )

    deleted_count = purge_stale_trash(days_in_trash=30)
    assert deleted_count == 0

    with _connect() as conn:
        row = conn.execute(
            "SELECT is_deleted FROM documents WHERE filename = ?",
            ("active_old_doc.pdf",),
        ).fetchone()
        assert row is not None and (row[0] == 0 or row[0] is None)


def test_add_chunks_logs_memory_usage(mock_db, caplog):
    """Test that add_chunks logs memory usage before and after insertions."""
    import logging

    add_document("doc_mem_test.pdf", "hash_mem_test")
    dummy_emb = np.ones(384, dtype=np.float32) * 0.5
    chunks = [(100, "doc_mem_test.pdf", 0, "Memory test chunk", dummy_emb)]

    with caplog.at_level(logging.INFO):
        add_chunks(chunks)

    log_messages = [record.message for record in caplog.records]
    assert any(
        "Memory usage before batch chunk insertion:" in msg for msg in log_messages
    )
    assert any(
        "Memory usage after batch chunk insertion:" in msg for msg in log_messages
    )


def test_get_document_count_by_user_returns_zero_for_unknown_user(mock_db):
    assert get_document_count_by_user("nobody") == 0


def test_get_document_count_by_user_counts_active_documents(mock_db):
    add_document("doc1.pdf", "hash_1", owner="alice")
    add_document("doc2.pdf", "hash_2", owner="alice")
    add_document("doc3.pdf", "hash_3", owner="bob")

    assert get_document_count_by_user("alice") == 2
    assert get_document_count_by_user("bob") == 1


def test_get_document_count_by_user_excludes_soft_deleted(mock_db):
    add_document("active.pdf", "hash_active", owner="alice")
    add_document("trashed.pdf", "hash_trashed", owner="alice")
    soft_delete_document("trashed.pdf")

    assert get_document_count_by_user("alice") == 1


def test_get_document_count_by_user_excludes_other_owners(mock_db):
    add_document("alice_doc.pdf", "hash_a", owner="alice")
    add_document("bob_doc.pdf", "hash_b", owner="bob")
    add_document("charlie_doc.pdf", "hash_c", owner="charlie")

    assert get_document_count_by_user("alice") == 1
    assert get_document_count_by_user("bob") == 1
    assert get_document_count_by_user("charlie") == 1


def test_get_document_count_by_user_handles_none_owner(mock_db):
    add_document("no_owner.pdf", "hash_none")
    add_document("alice_doc.pdf", "hash_alice", owner="alice")

    assert get_document_count_by_user("alice") == 1
    assert get_document_count_by_user("") == 0


def test_get_document_count_by_user_does_not_crash_with_many_null_owners(mock_db):
    """Regression test: NULL-owner documents (e.g. from add_document() calls
    that omit `owner`, still fully possible even after migration_010's
    DEFAULT 'system' backfill -- see migration_010_add_document_owner's
    docstring) must never cause get_document_count_by_user() to raise, and
    must never be miscounted against a real user."""
    for i in range(5):
        add_document(f"no_owner_{i}.pdf", f"hash_none_{i}")  # owner omitted -> NULL
    add_document("alice_doc.pdf", "hash_alice", owner="alice")
    add_document("bob_doc.pdf", "hash_bob", owner="bob")

    # No crash, and NULL-owner rows are excluded from every real user's count.
    assert get_document_count_by_user("alice") == 1
    assert get_document_count_by_user("bob") == 1
    assert get_document_count_by_user("system") == 0
    assert get_document_count_by_user("") == 0


def test_get_document_count_by_user_does_not_crash_when_queried_with_none(mock_db):
    """Calling the function itself with owner_username=None (SQL
    ``owner = NULL`` never matches, per SQL's NULL-comparison semantics)
    must not raise, and must correctly return 0 rather than matching
    NULL-owner rows."""
    from src.db.corpus_db import get_document_count_by_user

    add_document("no_owner.pdf", "hash_none_for_none_query")

    result = get_document_count_by_user(None)
    assert result == 0


def test_get_document_count_by_user_returns_int(mock_db):
    add_document("doc.pdf", "hash", owner="alice")
    result = get_document_count_by_user("alice")
    assert isinstance(result, int)
    assert result == 1


def test_get_document_count_fast(mock_db):
    """Verify that get_document_count_fast returns correct counts for active and deleted documents."""
    clear_all_data()

    # Initially count is 0
    assert get_document_count_fast(include_deleted=False) == 0
    assert get_document_count_fast(include_deleted=True) == 0

    # Add active documents
    add_document("doc1.pdf", "hash_1")
    add_document("doc2.pdf", "hash_2")

    # Add soft-deleted document
    add_document("doc3.pdf", "hash_3")
    soft_delete_document("doc3.pdf")

    # Verify counts
    assert get_document_count_fast(include_deleted=False) == 2
    assert get_document_count_fast(include_deleted=True) == 3


def test_get_document_count_by_user():
    from src.db.corpus_db import _connect, get_document_count_by_user

    with _connect() as db:
        db.execute(
            "INSERT INTO documents (filename, file_hash, upload_date, owner, is_deleted) VALUES (?, ?, ?, ?, ?)",
            ("1.pdf", "hash1", "date", "alice", 0),
        )
        db.execute(
            "INSERT INTO documents (filename, file_hash, upload_date, owner, is_deleted) VALUES (?, ?, ?, ?, ?)",
            ("2.pdf", "hash2", "date", "alice", 0),
        )
        db.execute(
            "INSERT INTO documents (filename, file_hash, upload_date, owner, is_deleted) VALUES (?, ?, ?, ?, ?)",
            ("3.pdf", "hash3", "date", "alice", 1),
        )

    assert get_document_count_by_user("alice") == 2


def test_get_document_count_by_user_empty():
    from src.db.corpus_db import get_document_count_by_user

    assert get_document_count_by_user("unknown-user") == 0


def test_deleted_chunks_has_deleted_at_column():
    """Verify deleted_chunks table has deleted_at column (#2342)."""
    import sqlite3

    import src.db.corpus_db as corpus_db

    conn = sqlite3.connect(corpus_db._DB_PATH)
    try:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(deleted_chunks)").fetchall()
        ]
        assert "deleted_at" in columns
    finally:
        conn.close()


def test_get_embedding_storage_footprint_empty(mock_db):
    """Test storage footprint on an empty database."""
    from src.db.corpus_db import get_embedding_storage_footprint

    # Ensure empty
    clear_all_data()

    res = get_embedding_storage_footprint()
    assert res["embedding_bytes"] == 0
    assert res["chunk_count"] == 0
    assert isinstance(res["database_bytes"], int)
    assert res["database_bytes"] > 0  # SQLite db file has overhead even if empty
    assert res["embedding_percentage"] == 0.0


def test_get_embedding_storage_footprint_normal(mock_db):
    """Test storage footprint with normal populated chunks."""
    from src.db.corpus_db import get_embedding_storage_footprint

    clear_all_data()
    add_document("doc1.pdf", "hash_footprint_1")
    
    # Add dummy embeddings of a known size (e.g. 384 floats = 1536 bytes each)
    dummy_emb_1 = np.ones(384, dtype=np.float32)
    dummy_emb_2 = np.ones(384, dtype=np.float32)
    
    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1 text", dummy_emb_1),
        (1, "doc1.pdf", 1, "Paragraph 2 text", dummy_emb_2),
    ]
    add_chunks(chunks)

    res = get_embedding_storage_footprint()
    assert res["chunk_count"] == 2
    # 2 * 384 * 4 = 3072 bytes
    assert res["embedding_bytes"] == 3072
    assert res["database_bytes"] > 0
    assert 0.0 < res["embedding_percentage"] <= 100.0


def test_get_embedding_storage_footprint_missing_file(monkeypatch, mock_db):
    """Test storage footprint handles OSError when checking DB file size."""
    from src.db.corpus_db import get_embedding_storage_footprint
    from pathlib import Path

    def mock_stat(*args, **kwargs):
        raise OSError("File not found mocked")

    monkeypatch.setattr(Path, "stat", mock_stat)

    res = get_embedding_storage_footprint()
    
    # database_bytes should fallback to 0 when stat raises OSError
    assert res["database_bytes"] == 0
    assert res["embedding_percentage"] == 0.0


def test_get_embedding_storage_footprint_null_values(mock_db):
    """Test storage footprint gracefully handles NULL returned from SUM() query."""
    from src.db.corpus_db import get_embedding_storage_footprint, _connect

    clear_all_data()

    # Manually insert a chunk with a NULL embedding to force SUM() behavior.
    # Note: the chunks table has a NOT NULL constraint on embedding, but 
    # we can bypass it by temporarily dropping the table or just querying an empty table
    # Wait, if table is empty, SUM() returns NULL.
    # We already test empty table, let's explicitly mock the cursor to return (None, 0)
    
    class MockCursor:
        def fetchone(self):
            return (None, 0)
            
    class MockConn:
        def execute(self, query):
            return MockCursor()
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import src.db.corpus_db
    original_connect = src.db.corpus_db._connect
    
    def mock_connect():
        return MockConn()

    src.db.corpus_db._connect = mock_connect
    try:
        res = get_embedding_storage_footprint()
        assert res["embedding_bytes"] == 0
        assert res["chunk_count"] == 0
    finally:
        src.db.corpus_db._connect = original_connect


def test_soft_delete_and_restore_document_returns_bool():
    add_document("doc_bool_1.pdf", "hash_b1")
    add_document("doc_bool_2.pdf", "hash_b2")
    add_document("doc_bool_3.pdf", "hash_b3")

    # Soft delete doc1
    assert soft_delete_document("doc_bool_1.pdf") is True
    # Attempting to soft-delete again returns False
    assert soft_delete_document("doc_bool_1.pdf") is False
    # Attempting to soft-delete non-existent doc returns False
    assert soft_delete_document("nonexistent.pdf") is False

    # Default get_all_documents excludes soft-deleted docs
    active_docs = get_all_documents()
    active_filenames = [d["filename"] for d in active_docs]
    assert "doc_bool_1.pdf" not in active_filenames
    assert "doc_bool_2.pdf" in active_filenames
    assert "doc_bool_3.pdf" in active_filenames

    # With include_deleted=True, soft-deleted doc is returned
    all_docs = get_all_documents(include_deleted=True)
    doc1_entry = next(d for d in all_docs if d["filename"] == "doc_bool_1.pdf")
    assert doc1_entry["deleted_at"] is not None

    doc2_entry = next(d for d in all_docs if d["filename"] == "doc_bool_2.pdf")
    assert doc2_entry["deleted_at"] is None

    # Restore doc1
    assert restore_document("doc_bool_1.pdf") is True
    # Attempting to restore already restored doc returns False
    assert restore_document("doc_bool_1.pdf") is False
    # Attempting to restore non-existent doc returns False
    assert restore_document("nonexistent.pdf") is False

    # Verify doc1 is back in default get_all_documents
    restored_docs = get_all_documents()
    restored_filenames = [d["filename"] for d in restored_docs]
    assert "doc_bool_1.pdf" in restored_filenames


def test_corpus_repository_class_interface():
    repo = CorpusRepository()
    add_document("repo_doc.pdf", "repo_hash_1")
    assert get_document_by_hash("repo_hash_1") == "repo_doc.pdf"

    docs = repo.get_all_documents()
    assert any(d["filename"] == "repo_doc.pdf" for d in docs)

    assert repo.soft_delete_document("repo_doc.pdf") is True
    assert not any(d["filename"] == "repo_doc.pdf" for d in repo.get_all_documents())
    assert any(d["filename"] == "repo_doc.pdf" for d in repo.get_all_documents(include_deleted=True))

    assert repo.restore_document("repo_doc.pdf") is True
    assert any(d["filename"] == "repo_doc.pdf" for d in repo.get_all_documents())

