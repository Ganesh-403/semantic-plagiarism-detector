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
    get_document_word_counts,
    get_documents_by_class,
    get_total_document_count,
    get_deleted_documents_count,
    get_unique_class_sections,
    purge_stale_trash,
    restore_document,
    soft_delete_document,
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
    from src.db.corpus_db import batch_soft_delete_documents, _connect

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
    from src.db.corpus_db import batch_permanently_delete_documents, _connect

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
    from src.db.corpus_db import batch_permanently_delete_documents, _connect

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
        rows = conn.execute(
            "SELECT id, filename FROM documents ORDER BY id"
        ).fetchall()
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
            ("INC-1", "doc1.pdf", "doc2.pdf", 0.85, "High", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
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

    word_counts = get_document_word_counts()
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


def test_get_document_count_by_user_returns_int(mock_db):
    add_document("doc.pdf", "hash", owner="alice")
    result = get_document_count_by_user("alice")
    assert isinstance(result, int)
    assert result == 1


def test_get_total_document_count(mock_db):
    assert get_total_document_count() == 0
    add_document("doc1.pdf", "hash_doc1")
    add_document("doc2.pdf", "hash_doc2")
    assert get_total_document_count() == 2
    soft_delete_document("doc1.pdf")
    assert get_total_document_count() == 1
    assert get_total_document_count(include_deleted=True) == 2


def test_documents_created_at_index_exists(mock_db):
    from src.db.corpus_db import get_corpus_db_path
    import sqlite3
    db_path = get_corpus_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='documents'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_documents_created_at" in indexes
    finally:
        conn.close()


def test_get_deleted_documents_count(mock_db):
    assert get_deleted_documents_count() == 0
    add_document("doc1.pdf", "hash_doc1")
    add_document("doc2.pdf", "hash_doc2")
    assert get_deleted_documents_count() == 0
    soft_delete_document("doc1.pdf")
    assert get_deleted_documents_count() == 1
    soft_delete_document("doc2.pdf")
    assert get_deleted_documents_count() == 2
    restore_document("doc1.pdf")
    assert get_deleted_documents_count() == 1
# ---------------------------------------------------------------------------
# Issue #1359 — FTS5 Full-Text Search tests
# ---------------------------------------------------------------------------


def test_search_documents_fts_empty_query():
    """Empty query must return an empty list."""
    from src.db.corpus_db import search_documents_fts
    assert search_documents_fts("") == []
    assert search_documents_fts("   ") == []
    assert search_documents_fts(None) == []


def test_search_documents_fts_no_matches():
    """A query that matches nothing must return an empty list."""
    from src.db.corpus_db import search_documents_fts
    add_document("test_fts_doc.pdf", "hash_fts_001")
    results = search_documents_fts("nonexistent_term_xyz")
    assert results == []


def test_search_documents_fts_finds_by_filename():
    """FTS search should find documents by filename."""
    from src.db.corpus_db import search_documents_fts
    add_document("machine_learning_essay.pdf", "hash_fts_002", student_name="Alice")
    results = search_documents_fts("machine")
    assert len(results) == 1
    assert results[0]["filename"] == "machine_learning_essay.pdf"
    assert results[0]["student_name"] == "Alice"


def test_search_documents_fts_finds_by_student_name():
    """FTS search should find documents by student name."""
    from src.db.corpus_db import search_documents_fts
    add_document("essay1.pdf", "hash_fts_003", student_name="Bob Smith")
    results = search_documents_fts("Bob")
    assert len(results) == 1
    assert results[0]["student_name"] == "Bob Smith"


def test_search_documents_fts_finds_by_assignment_title():
    """FTS search should find documents by assignment title."""
    from src.db.corpus_db import search_documents_fts
    add_document("lab_report.pdf", "hash_fts_004", assignment_title="Final Lab Report")
    results = search_documents_fts("Final")
    assert len(results) == 1
    assert results[0]["assignment_title"] == "Final Lab Report"


def test_search_documents_fts_returns_correct_fields():
    """The result dict must contain all expected fields."""
    from src.db.corpus_db import search_documents_fts
    add_document("data_science.pdf", "hash_fts_005", student_name="Carol", assignment_title="ML Project")
    results = search_documents_fts("data")
    assert len(results) == 1
    result = results[0]
    assert "id" in result
    assert "filename" in result
    assert "student_name" in result
    assert "assignment_title" in result
    assert "upload_date" in result
    assert "snippet" in result


def test_search_documents_fts_excludes_deleted():
    """Soft-deleted documents must not appear in FTS results."""
    from src.db.corpus_db import search_documents_fts, soft_delete_document
    add_document("active_doc.pdf", "hash_fts_006", student_name="Active User")
    add_document("deleted_doc.pdf", "hash_fts_007", student_name="Deleted User")
    soft_delete_document("deleted_doc.pdf")
    results = search_documents_fts("User")
    # Only the non-deleted doc should appear
    filenames = [r["filename"] for r in results]
    assert "active_doc.pdf" in filenames
    assert "deleted_doc.pdf" not in filenames


def test_search_documents_fts_multiple_results():
    """FTS search should return all matching documents, ranked by relevance."""
    from src.db.corpus_db import search_documents_fts
    add_document("plagiarism_detection.pdf", "hash_fts_008", student_name="Alice")
    add_document("plagiarism_essay.pdf", "hash_fts_009", student_name="Bob")
    add_document("unrelated_topic.pdf", "hash_fts_010", student_name="Carol")
    results = search_documents_fts("plagiarism")
    assert len(results) == 2
    filenames = {r["filename"] for r in results}
    assert "plagiarism_detection.pdf" in filenames
    assert "plagiarism_essay.pdf" in filenames


def test_search_documents_fts_trigger_sync_on_delete():
    """Hard-deleting a document must remove it from the FTS index."""
    from src.db.corpus_db import search_documents_fts, delete_document
    add_document("to_delete.pdf", "hash_fts_011", student_name="Delete Me")
    results_before = search_documents_fts("Delete")
    assert len(results_before) == 1
    delete_document("to_delete.pdf")
    results_after = search_documents_fts("Delete")
    assert len(results_after) == 0


def test_fts5_virtual_table_exists():
    """The documents_fts virtual table must exist after migration."""
    from src.db.corpus_db import get_corpus_db_path
    import sqlite3
    conn = sqlite3.connect(str(get_corpus_db_path()))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert cursor.fetchone() is not None
    finally:
        conn.close()


def test_fts5_triggers_exist():
    """The FTS sync triggers must exist after migration."""
    from src.db.corpus_db import get_corpus_db_path
    import sqlite3
    conn = sqlite3.connect(str(get_corpus_db_path()))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('documents_ai', 'documents_ad', 'documents_au')"
        )
        trigger_names = {row[0] for row in cursor.fetchall()}
        assert "documents_ai" in trigger_names
        assert "documents_ad" in trigger_names
        assert "documents_au" in trigger_names
    finally:
        conn.close()


def test_corpus_soft_delete_lifecycle():
    """Verify document soft-deletion and soft-delete recovery workflows."""
    filename = "lifecycle_test.pdf"
    file_hash = "lifecycle_hash_999"

    # 1. Add document and chunks
    added = add_document(
        filename=filename,
        file_hash=file_hash,
        student_name="Lifecycle Tester",
        class_section="CS 101",
        assignment_title="Lifecycle Assignment",
    )
    assert isinstance(added, int)

    dummy_embedding = np.random.rand(384).astype(np.float32)
    add_chunks([(1000, filename, 0, "Initial paragraph content", dummy_embedding)])

    # Verify present in active queries initially
    active_docs = get_all_documents(include_deleted=False)
    assert len(active_docs) == 1
    assert active_docs[0]["filename"] == filename

    active_chunks = get_chunk_registry()
    assert len(active_chunks) == 1
    assert active_chunks[0].doc_name == filename

    assert get_total_document_count() == 1
    assert get_deleted_documents_count() == 0

    # 2. Soft delete the document
    soft_delete_document(filename)

    # Verify excluded from active queries
    assert len(get_all_documents(include_deleted=False)) == 0
    assert len(get_chunk_registry()) == 0
    assert get_total_document_count() == 0
    assert get_deleted_documents_count() == 1

    # Verify present in get_deleted_documents
    deleted_docs = get_deleted_documents()
    assert len(deleted_docs) == 1
    assert deleted_docs[0]["filename"] == filename

    # 3. Restore the document
    restore_document(filename)

    # Verify included in active queries again
    active_docs_after = get_all_documents(include_deleted=False)
    assert len(active_docs_after) == 1
    assert active_docs_after[0]["filename"] == filename

    active_chunks_after = get_chunk_registry()
    assert len(active_chunks_after) == 1
    assert active_chunks_after[0].doc_name == filename
    assert active_chunks_after[0].chunk_text == "Initial paragraph content"

    assert get_total_document_count() == 1
    assert get_deleted_documents_count() == 0


def test_soft_delete_and_restore_document():
    """Verify document soft-deletion and subsequent restoration flow (#1284)."""
    filename = "delete_restore_test.pdf"
    file_hash = "delete_restore_hash_123"

    # Insert document
    added = add_document(
        filename=filename,
        file_hash=file_hash,
        student_name="Test Student",
        class_section="Section A",
        assignment_title="Test Assignment",
    )
    assert added is True

    dummy_embedding = np.random.rand(384).astype(np.float32)
    add_chunks([(999, filename, 0, "Test chunk content", dummy_embedding)])

    # Verify visible initially
    active_docs = get_all_documents(include_deleted=False)
    assert any(doc["filename"] == filename for doc in active_docs)

    # Call soft_delete_document()
    soft_delete_document(filename)

    # Verify hidden from queries
    active_docs_after_delete = get_all_documents(include_deleted=False)
    assert not any(doc["filename"] == filename for doc in active_docs_after_delete)

    # Call restore_document()
    restore_document(filename)

    # Verify visible in queries again
    active_docs_after_restore = get_all_documents(include_deleted=False)
    assert any(doc["filename"] == filename for doc in active_docs_after_restore)


