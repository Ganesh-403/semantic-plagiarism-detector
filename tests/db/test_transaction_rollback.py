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
tests/db/test_transaction_rollback.py
--------------------------------------
Unit tests for Issue #622: Ensuring database state cleanly rolls back
when an insertion operation fails halfway.
"""

import sqlite3

import numpy as np
import pytest

from src.db.corpus_db import (
    _connect,
    add_document,
    clear_all_data,
    get_all_documents,
    get_chunk_registry,
    init_corpus_db,
)
from src.db.incidents import _get_connection, init_incident_db


@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Ensure a clean database state before and after each test."""
    init_corpus_db()
    clear_all_data()
    yield
    clear_all_data()


def test_raw_sqlite_transaction_rollback_on_halfway_error(sqlite_database_path):
    """
    Verify that an atomic SQLite transaction rolls back completely when an insertion
    fails halfway through a batch/multi-row operation.
    """
    conn = sqlite3.connect(sqlite_database_path)
    conn.execute(
        "CREATE TABLE test_records (id INTEGER PRIMARY KEY, value TEXT UNIQUE NOT NULL)"
    )
    conn.commit()

    # Attempt inserting multiple rows within a single transaction
    try:
        with conn:
            conn.execute("INSERT INTO test_records (id, value) VALUES (1, 'alpha')")
            conn.execute("INSERT INTO test_records (id, value) VALUES (2, 'beta')")
            # Row 3 triggers IntegrityError due to duplicate UNIQUE value 'alpha'
            conn.execute("INSERT INTO test_records (id, value) VALUES (3, 'alpha')")
    except sqlite3.IntegrityError:
        pass

    # Verify that the entire transaction rolled back and zero rows remain
    cursor = conn.execute("SELECT COUNT(*) FROM test_records")
    count = cursor.fetchone()[0]
    conn.close()

    assert (
        count == 0
    ), "Database should contain 0 rows after halfway insertion failure rollback"


def test_corpus_db_rollback_on_chunk_insertion_error():
    """
    Verify that if chunk insertion fails halfway inside a _connect() transaction block,
    all pending insertions are rolled back and no partial chunk data remains in the DB.
    """
    add_document("sample_doc.pdf", "hash_sample_123")
    assert len(get_all_documents()) == 1

    dummy_emb = np.ones(384, dtype=np.float32)

    # Valid chunk + invalid chunk (violating NOT NULL on chunk_text via raw SQL in same transaction)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (0, "sample_doc.pdf", 0, "First valid chunk", dummy_emb.tobytes()),
            )
            # Second insertion fails due to NULL chunk_text constraint
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (1, "sample_doc.pdf", 1, None, dummy_emb.tobytes()),
            )
    except sqlite3.IntegrityError:
        pass

    # Verify no chunks were persisted
    registry = get_chunk_registry()
    assert (
        len(registry) == 0
    ), "No chunks should remain persisted after transaction rollback"


def test_corpus_db_multi_table_atomic_rollback():
    """
    Verify atomicity across multiple tables (documents + chunks): if chunk insertion fails
    after document insertion in the same transaction block, neither document nor chunk data is saved.
    """
    dummy_emb = np.ones(384, dtype=np.float32)

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("atomic_doc.pdf", "hash_atomic_999", "2026-07-28T00:00:00"),
            )
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (10, "atomic_doc.pdf", 0, "Valid chunk", dummy_emb.tobytes()),
            )
            # Simulated failure halfway: duplicate vector_id or constraint failure
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    10,
                    "atomic_doc.pdf",
                    1,
                    "Duplicate vector_id chunk",
                    dummy_emb.tobytes(),
                ),
            )
    except sqlite3.IntegrityError:
        pass

    # Verify atomic rollback: neither document nor chunk exists
    docs = get_all_documents()
    chunks = get_chunk_registry()
    assert (
        len(docs) == 0
    ), "Document insertion must roll back when chunk insertion fails"
    assert len(chunks) == 0, "Chunk insertion must roll back completely"


def test_incidents_batch_insertion_rollback_on_error(sqlite_database_path):
    """
    Verify that when incident batch insertion encounters a database error midway,
    no partial incidents are committed.
    """
    init_incident_db(sqlite_database_path)

    conn = _get_connection(sqlite_database_path)
    try:
        # Check initial state
        initial_count = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents"
        ).fetchone()[0]
        assert initial_count == 0

        # Run a multi-insert statement transaction that fails halfway
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank)
                    VALUES ('INC-001', 'docA.pdf', 'docB.pdf', 0.85, 'High')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank)
                    VALUES ('INC-002', 'docC.pdf', 'docD.pdf', 0.92, 'High')
                    """
                )
                # Fail on 3rd row with duplicate incident_id
                conn.execute(
                    """
                    INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank)
                    VALUES ('INC-001', 'docE.pdf', 'docF.pdf', 0.75, 'Medium')
                    """
                )
        except sqlite3.IntegrityError:
            pass

        final_count = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents"
        ).fetchone()[0]
        assert (
            final_count == 0
        ), "Plagiarism incidents table should have 0 rows after rollback"
    finally:
        conn.close()
