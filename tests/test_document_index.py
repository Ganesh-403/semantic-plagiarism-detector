"""Verify hash indexing and uniqueness on the migrated corpus schema."""

import sqlite3
import pytest
from src.db import corpus_db


def test_document_hash_has_an_explicit_index(mock_db):
    with corpus_db._connect() as connection:
        indexes = connection.execute("PRAGMA index_list(documents)").fetchall()
        matching = [row for row in indexes if row[1] == "idx_documents_file_hash"]
        assert len(matching) == 1
        columns = connection.execute("PRAGMA index_info(idx_documents_file_hash)").fetchall()
        assert [row[2] for row in columns] == ["file_hash"]


def test_duplicate_hash_cannot_create_a_second_document(mock_db):
    corpus_db.add_document("first.txt", "same-hash")
    with corpus_db._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("second.txt", "same-hash", "2026-01-01"),
            )
    assert len(corpus_db.get_all_documents()) == 1


def test_reinitialization_preserves_hash_lookup(mock_db):
    corpus_db.add_document("first.txt", "first-hash")
    corpus_db.close_connections()
    corpus_db.init_corpus_db()
    with corpus_db._connect() as connection:
        assert connection.execute(
            "SELECT filename FROM documents WHERE file_hash = ?", ("first-hash",)
        ).fetchone()[0] == "first.txt"
