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

import numpy as np
import pytest

from src.db.corpus_db import (
    add_chunks,
    add_document,
    empty_trash,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_deleted_documents,
    permanently_delete_document,
    restore_document,
    soft_delete_document,
)
from src.db.incidents import (
    get_all_incidents,
    get_all_incidents_above_threshold_for_export,
    sync_flagged_incidents,
)


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    yield


def test_soft_delete_and_restore():
    # 1. Add document and chunks
    add_document("doc1.pdf", "hash1")
    add_document("doc2.pdf", "hash2")

    dummy_emb_1 = np.ones(384, dtype=np.float32) * 0.1
    dummy_emb_2 = np.ones(384, dtype=np.float32) * 0.2

    chunks = [
        (0, "doc1.pdf", 0, "Paragraph A", dummy_emb_1),
        (1, "doc2.pdf", 0, "Paragraph B", dummy_emb_2),
    ]
    add_chunks(chunks)

    # Verify initial state
    assert len(get_all_documents(include_deleted=False)) == 2
    assert len(get_deleted_documents()) == 0
    assert len(get_chunk_registry()) == 2
    assert get_all_embeddings().shape == (2, 384)

    # 2. Soft delete doc1
    soft_delete_document("doc1.pdf")

    # Verify soft-deleted state
    assert len(get_all_documents(include_deleted=False)) == 1
    assert get_all_documents(include_deleted=False)[0]["filename"] == "doc2.pdf"

    deleted = get_deleted_documents()
    assert len(deleted) == 1
    assert deleted[0]["filename"] == "doc1.pdf"
    assert deleted[0]["deleted_at"] is not None

    # Verify active chunks compacted and exclude doc1 chunks
    registry = get_chunk_registry()
    assert len(registry) == 1
    assert registry[0].doc_name == "doc2.pdf"
    assert registry[0].chunk_text == "Paragraph B"
    assert get_all_embeddings().shape == (1, 384)

    # 3. Restore doc1
    restore_document("doc1.pdf")

    # Verify restored state
    assert len(get_all_documents(include_deleted=False)) == 2
    assert len(get_deleted_documents()) == 0

    # Chunks are re-indexed and sequential again
    registry = get_chunk_registry()
    assert len(registry) == 2
    assert get_all_embeddings().shape == (2, 384)


def test_permanent_delete_and_empty_trash(mock_db):
    from src.db.incidents import (
        add_false_positive,
        get_all_incidents,
        get_false_positives,
        sync_flagged_incidents,
    )

    add_document("doc1.pdf", "hash1")
    add_document("doc2.pdf", "hash2")

    # Add an incident and a false positive for verification
    flags = [{"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.95}]
    sync_flagged_incidents(flags, db_path=mock_db)
    add_false_positive("doc1.pdf", "doc2.pdf", db_path=mock_db)

    assert len(get_all_incidents(db_path=mock_db)) == 1
    assert len(get_false_positives(db_path=mock_db)) == 1

    # Soft delete doc1
    soft_delete_document("doc1.pdf")

    assert len(get_all_documents(include_deleted=True)) == 2
    assert len(get_deleted_documents()) == 1

    # The incident involves a soft-deleted document, so it is filtered out of active incidents
    assert len(get_all_incidents(db_path=mock_db)) == 0
    # But they are still physically present in the database tables
    import sqlite3

    with sqlite3.connect(mock_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM plagiarism_incidents").fetchone()[0] == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM false_positives").fetchone()[0] == 1

    # Permanently delete doc1
    permanently_delete_document("doc1.pdf")
    assert len(get_all_documents(include_deleted=True)) == 1
    assert len(get_deleted_documents()) == 0

    # Incidents and false positives involving doc1 should be physically removed
    with sqlite3.connect(mock_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM plagiarism_incidents").fetchone()[0] == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM false_positives").fetchone()[0] == 0

    # Test empty_trash
    add_document("doc3.pdf", "hash3")
    sync_flagged_incidents(
        [{"doc_a": "doc2.pdf", "doc_b": "doc3.pdf", "similarity": 0.90}],
        db_path=mock_db,
    )
    add_false_positive("doc2.pdf", "doc3.pdf", db_path=mock_db)

    # Soft delete doc2
    soft_delete_document("doc2.pdf")

    # They should be physically present but filtered from active incidents
    assert len(get_all_incidents(db_path=mock_db)) == 0
    with sqlite3.connect(mock_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM plagiarism_incidents").fetchone()[0] == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM false_positives").fetchone()[0] == 1

    # Empty trash to delete remaining soft-deleted documents (doc2)
    empty_trash()
    assert len(get_all_documents(include_deleted=True)) == 1  # doc3 remains
    assert len(get_deleted_documents()) == 0

    # Incidents and false positives involving doc2 should be physically gone
    with sqlite3.connect(mock_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM plagiarism_incidents").fetchone()[0] == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM false_positives").fetchone()[0] == 0


def test_incidents_filter_soft_deleted(mock_db):
    add_document("doc1.pdf", "hash1")
    add_document("doc2.pdf", "hash2")
    add_document("doc3.pdf", "hash3")

    dummy_emb = np.zeros(384, dtype=np.float32)
    add_chunks(
        [
            (0, "doc1.pdf", 0, "Paragraph 1", dummy_emb),
            (1, "doc2.pdf", 0, "Paragraph 2", dummy_emb),
            (2, "doc3.pdf", 0, "Paragraph 3", dummy_emb),
        ]
    )

    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.90,
            "severity": "High",
        },
        {
            "doc_b": "doc2.pdf",
            "doc_a": "doc3.pdf",
            "similarity": 0.85,
            "severity": "High",
        },
    ]
    # Pass db_path explicitly because the function's default parameter
    # captured the original DEFAULT_DB_PATH at import time, before mock.patch.
    sync_flagged_incidents(flags, db_path=mock_db)

    assert len(get_all_incidents(db_path=mock_db)) == 2
    assert len(get_all_incidents_above_threshold_for_export(0.80, db_path=mock_db)) == 2

    # Soft delete doc2
    soft_delete_document("doc2.pdf")

    # Both incidents involve doc2, so they should be filtered out!
    assert len(get_all_incidents(db_path=mock_db)) == 0
    assert len(get_all_incidents_above_threshold_for_export(0.80, db_path=mock_db)) == 0
