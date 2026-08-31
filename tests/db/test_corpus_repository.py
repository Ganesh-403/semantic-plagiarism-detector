from __future__ import annotations

import sqlite3

from src.db import corpus_db
from src.db.corpus_repository import CorpusRepository


def _seed(tmp_path):
    db = tmp_path / "corpus.db"
    corpus_db.configure_db_path(db)
    with corpus_db._connect() as conn:
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                upload_date TEXT NOT NULL,
                class_section TEXT,
                student_name TEXT,
                assignment_title TEXT,
                pdf_author TEXT,
                pdf_creation_date TEXT,
                pdf_title TEXT,
                tags TEXT,
                detected_language TEXT,
                owner TEXT
            )
        """)
        conn.executemany(
            """INSERT INTO documents
               (filename, file_hash, upload_date, class_section, student_name,
                assignment_title, owner)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                ("a.pdf", "ha", "2026-08-01T10:00:00", "CS101", "Alice", "OS", "inst-a"),
                ("b.pdf", "hb", "2026-08-02T10:00:00", "CS101", "Bob", "DB", "inst-a"),
                ("c.pdf", "hc", "2026-08-03T10:00:00", "CS102", "Alice", "OS", "inst-b"),
            ],
        )
    return CorpusRepository()


def test_filters_by_multiple_metadata_fields(tmp_path):
    repo = _seed(tmp_path)

    rows = repo.get_documents_by_metadata(
        class_section="CS101",
        assignment_title="DB",
    )

    assert [row["filename"] for row in rows] == ["b.pdf"]


def test_filters_by_owner_and_student(tmp_path):
    repo = _seed(tmp_path)

    rows = repo.get_documents_by_metadata(student_name="Alice", owner="inst-b")

    assert [row["filename"] for row in rows] == ["c.pdf"]


def test_omitted_filters_return_all_documents(tmp_path):
    repo = _seed(tmp_path)

    rows = repo.get_documents_by_metadata()

    assert [row["filename"] for row in rows] == ["c.pdf", "b.pdf", "a.pdf"]


def test_values_are_parameterized(tmp_path):
    repo = _seed(tmp_path)

    malicious = '" OR 1=1 --'
    rows = repo.get_documents_by_metadata(student_name=malicious)

    assert rows == []


def test_soft_delete_and_restore_document(tmp_path):
    db = tmp_path / "corpus_sd.db"
    corpus_db.configure_db_path(db)
    corpus_db.init_corpus_db()
    repo = CorpusRepository()

    corpus_db.add_document("doc1.pdf", "hash1")
    corpus_db.add_document("doc2.pdf", "hash2")

    # Initial state: 2 documents
    docs = repo.get_all_documents()
    assert len(docs) == 2

    # Soft delete doc1
    assert repo.soft_delete_document("doc1.pdf") is True
    # Attempting to soft-delete again returns False
    assert repo.soft_delete_document("doc1.pdf") is False
    # Attempting to soft-delete non-existent doc returns False
    assert repo.soft_delete_document("nonexistent.pdf") is False

    # Default get_all_documents excludes soft-deleted docs
    active_docs = repo.get_all_documents()
    active_filenames = [d.filename for d in active_docs]
    assert "doc1.pdf" not in active_filenames
    assert "doc2.pdf" in active_filenames

    # With include_deleted=True, soft-deleted doc is returned
    all_docs = repo.get_all_documents(include_deleted=True)
    all_filenames = [d.filename for d in all_docs]
    assert "doc1.pdf" in all_filenames
    assert "doc2.pdf" in all_filenames

    # Restore doc1
    assert repo.restore_document("doc1.pdf") is True
    # Attempting to restore already restored doc returns False
    assert repo.restore_document("doc1.pdf") is False
    # Attempting to restore non-existent doc returns False
    assert repo.restore_document("nonexistent.pdf") is False

    # Verify doc1 is back in default get_all_documents
    restored_docs = repo.get_all_documents()
    restored_filenames = [d.filename for d in restored_docs]
    assert "doc1.pdf" in restored_filenames

