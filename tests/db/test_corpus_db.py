import numpy as np
import pytest

from src.db.corpus_db import (add_chunks, add_document, clear_all_data,
                              delete_document, get_all_documents,
                              get_all_embeddings, get_chunk_registry,
                              get_document_by_hash, get_document_chunks_count,
                              get_documents_by_class,
                              get_unique_class_sections)


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """
    Uses the global mock_db fixture from conftest.py for complete DB isolation
    and automatic teardown per test.
    """
    yield


def test_add_document_metadata():
    # Add first document
    res1 = add_document("test1.pdf", "hash_abc_123")
    assert res1 is True

    # Try adding a duplicate hash/document
    res2 = add_document("test2.pdf", "hash_abc_123")
    assert res2 is False  # Unique hash constraint triggers

    # Try adding a duplicate filename
    res3 = add_document("test1.pdf", "different_hash")
    assert res3 is False  # Unique filename constraint triggers


def test_get_document_by_hash():
    add_document("doc_alpha.txt", "hash_xyz_789")

    match = get_document_by_hash("hash_xyz_789")
    assert match == "doc_alpha.txt"

    no_match = get_document_by_hash("nonexistent_hash")
    assert no_match is None


def test_add_and_retrieve_chunks():
    add_document("doc1.pdf", "hash_1")

    # Format of chunk insertion tuples: (vector_id, filename, chunk_index, chunk_text, embedding)
    dummy_emb_1 = np.ones(384, dtype=np.float32) * 0.5
    dummy_emb_2 = np.ones(384, dtype=np.float32) * 1.5

    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1 text", dummy_emb_1),
        (1, "doc1.pdf", 1, "Paragraph 2 text", dummy_emb_2),
    ]

    add_chunks(chunks)

    # Check count
    assert get_document_chunks_count("doc1.pdf") == 2

    # Check registry loading
    registry = get_chunk_registry()
    assert len(registry) == 2
    assert registry[0].doc_name == "doc1.pdf"
    assert registry[0].chunk_text == "Paragraph 1 text"

    # Check embeddings extraction
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

    # Delete doc1
    delete_document("doc1.pdf")

    # Check document counts
    all_docs = get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0]["filename"] == "doc2.pdf"

    # Check that remaining chunks have compact vector_ids starting at 0
    registry = get_chunk_registry()
    assert len(registry) == 1
    assert registry[0].doc_name == "doc2.pdf"

    embs = get_all_embeddings()
    assert embs.shape == (1, 384)


def test_document_metadata_fields():
    # Insert with metadata fields
    res = add_document(
        "metadata_test.pdf",
        "hash_metadata_123",
        class_section="Class B",
        student_name="Alice Smith",
        assignment_title="Homework 1",
        detected_language="en",
    )
    assert res is True

    # Retrieve and check fields
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
    # Add documents belonging to different classes
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
    add_document("doc_empty.pdf", "hash_empty")  # No metadata class

    # Verify unique class list
    classes = get_unique_class_sections()
    assert "Class A" in classes
    assert "Class B" in classes
    assert len(classes) == 2  # None or empty string shouldn't be included

    # Verify getting documents by class
    class_a_docs = get_documents_by_class("Class A")
    assert "doc_a.pdf" in class_a_docs
    assert "doc_c.pdf" in class_a_docs
    assert len(class_a_docs) == 2

    class_b_docs = get_documents_by_class("Class B")
    assert "doc_b.pdf" in class_b_docs
    assert len(class_b_docs) == 1


def test_clear_all_data_clears_incidents(mock_db):
    from src.db.incidents import get_all_incidents, sync_flagged_incidents
    from pathlib import Path

    db_path = Path(mock_db)

    # 1. Add mock documents
    add_document("doc1.pdf", "hash1")
    add_document("doc2.pdf", "hash2")

    # 2. Add mock incidents
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.85,
            "severity": "High",
        }
    ]
    sync_flagged_incidents(flags, db_path=db_path)

    # Verify they exist
    incidents = get_all_incidents(db_path=db_path)
    assert len(incidents) == 1

    # 3. Clear all data
    clear_all_data()

    # Verify everything is cleared
    assert len(get_all_documents()) == 0
    assert len(get_all_incidents(db_path=db_path)) == 0


def test_get_document_word_counts():
    import numpy as np

    from src.db.corpus_db import (add_chunks, add_document, clear_all_data,
                                  get_document_word_counts)

    clear_all_data()

    # 1. Add mock documents
    add_document("doc1.txt", "hash_doc1")
    add_document("doc2.txt", "hash_doc2")

    # 2. Add chunks with text
    chunks = [
        (1, "doc1.txt", 0, "This is the first chunk.", np.zeros(384)),
        (2, "doc1.txt", 1, "And this is the second chunk of doc1.", np.zeros(384)),
        (3, "doc2.txt", 0, "Doc2 has only one single chunk.", np.zeros(384)),
    ]
    add_chunks(chunks)

    # 3. Retrieve word counts
    word_counts = get_document_word_counts()

    # "This is the first chunk." -> 5 words
    # "And this is the second chunk of doc1." -> 8 words
    # doc1 total = 13 words
    assert word_counts["doc1.txt"] == 13

    # "Doc2 has only one single chunk." -> 6 words
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
    from src.db.corpus_db import optimize_database, configure_db_path, get_corpus_db_path

    original_path = get_corpus_db_path()
    try:
        configure_db_path("Z:\\invalid_dir_xyz_123\\corpus.db")
        res = optimize_database()
        assert res["error"] is not None
        assert res["size_before"] == 0
        assert res["size_after"] == 0
        assert res["reclaimed_bytes"] == 0
    finally:
        configure_db_path(original_path)
