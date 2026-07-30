
import pytest

from src.db.corpus_db import (add_documents_bulk, clear_all_data,
                              get_all_documents, init_corpus_db)


@pytest.fixture(autouse=True)
def setup_teardown():
    init_corpus_db()
    clear_all_data()
    yield
    clear_all_data()

def test_add_documents_bulk_success():
    docs = [
        {
            "filename": "bulk_doc_1.pdf",
            "file_hash": "hash_bulk_1",
            "class_section": "Class A",
            "student_name": "Alice",
            "assignment_title": "HW1",
            "detected_language": "en"
        },
        {
            "filename": "bulk_doc_2.pdf",
            "file_hash": "hash_bulk_2",
            "class_section": "Class B",
            "student_name": "Bob",
            "assignment_title": "HW2",
            "detected_language": "es"
        },
        {
            "filename": "bulk_doc_3.pdf",
            "file_hash": "hash_bulk_3",
            "class_section": "Class A",
            "student_name": "Charlie",
            "assignment_title": "HW1"
        }
    ]
    
    success_count = add_documents_bulk(docs)
    assert success_count == 3
    
    all_docs = get_all_documents()
    assert len(all_docs) == 3
    filenames = [d["filename"] for d in all_docs]
    assert "bulk_doc_1.pdf" in filenames
    assert "bulk_doc_2.pdf" in filenames
    assert "bulk_doc_3.pdf" in filenames

    # Find docs and assert language
    doc1 = next(d for d in all_docs if d["filename"] == "bulk_doc_1.pdf")
    doc2 = next(d for d in all_docs if d["filename"] == "bulk_doc_2.pdf")
    doc3 = next(d for d in all_docs if d["filename"] == "bulk_doc_3.pdf")
    assert doc1["detected_language"] == "en"
    assert doc2["detected_language"] == "es"
    assert doc3["detected_language"] is None

def test_add_documents_bulk_duplicate_ignore():
    # Test that inserting duplicate filenames/hashes does not fail but ignores them
    docs = [
        {"filename": "dup_1.pdf", "file_hash": "hash_dup_1"},
        {"filename": "dup_2.pdf", "file_hash": "hash_dup_2"}
    ]
    assert add_documents_bulk(docs) == 2
    
    # Try inserting same docs plus one new
    docs_with_dups = [
        {"filename": "dup_1.pdf", "file_hash": "hash_dup_1"},
        {"filename": "dup_3.pdf", "file_hash": "hash_dup_3"}
    ]
    # Because of INSERT OR IGNORE, dup_1 is skipped, dup_3 is inserted.
    success_count = add_documents_bulk(docs_with_dups)
    assert success_count == 1
    
    all_docs = get_all_documents()
    assert len(all_docs) == 3

def test_add_documents_bulk_empty():
    assert add_documents_bulk([]) == 0
    assert len(get_all_documents()) == 0

def test_add_documents_bulk_missing_fields():
    # Verify defaults handle missing fields gracefully
    docs = [
        {"file_hash": "hash123"} # filename is missing, triggers IntegrityError on PRIMARY KEY which is swallowed by INSERT OR IGNORE
    ]
    
    success_count = add_documents_bulk(docs)
    assert success_count == 0
    assert len(get_all_documents()) == 0
