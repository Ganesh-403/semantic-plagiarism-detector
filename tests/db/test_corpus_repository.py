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
