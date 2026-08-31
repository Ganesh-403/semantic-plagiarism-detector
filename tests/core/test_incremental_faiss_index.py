"""
tests/core/test_incremental_faiss_index.py
--------------------------------------------
Tests for incremental FAISS index updates (Issue #3913).
Covers add, update, delete, consistency validation, and recovery.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.core.faiss_index import (
    add_vectors_incremental,
    get_index_consistency_status,
    remove_vectors_incremental,
)
from src.core.faiss_index_metadata import FAISSIndexMetadata

try:
    import faiss
except ImportError:
    faiss = None


@pytest.fixture
def temp_metadata_path():
    """Temporary path for metadata file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "metadata.json")


@pytest.fixture
def mock_index():
    """Create a small test FAISS index."""
    if faiss is None:
        pytest.skip("FAISS not installed")
    
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(5, dim).astype("float32")
    index.add(vectors)
    return index


def test_metadata_add_vector(temp_metadata_path):
    """Verify adding vectors to metadata tracking."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    
    mgr.add_vector(0, "doc_a", 0, "Text for chunk 0")
    mgr.add_vector(1, "doc_a", 1, "Text for chunk 1")
    mgr.add_vector(2, "doc_b", 0, "Another document")
    
    assert mgr.metadata.total_vectors == 3
    assert mgr.get_vector_mapping(0)["doc_name"] == "doc_a"
    assert mgr.get_vector_mapping(2)["doc_name"] == "doc_b"


def test_metadata_remove_vector(temp_metadata_path):
    """Verify removing vectors from metadata."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    mgr.add_vector(0, "doc_a", 0, "Text")
    mgr.add_vector(1, "doc_b", 0, "Text")
    
    mgr.remove_vector(0)
    
    assert mgr.metadata.total_vectors == 1
    assert mgr.get_vector_mapping(0) is None
    assert mgr.get_vector_mapping(1) is not None


def test_metadata_get_vectors_for_document(temp_metadata_path):
    """Verify retrieval of all vectors for a document."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    mgr.add_vector(0, "doc_a", 0, "Text 0")
    mgr.add_vector(1, "doc_a", 1, "Text 1")
    mgr.add_vector(2, "doc_b", 0, "Text 0")
    
    doc_a_vecs = mgr.get_vectors_for_document("doc_a")
    assert doc_a_vecs == [0, 1]
    
    doc_b_vecs = mgr.get_vectors_for_document("doc_b")
    assert doc_b_vecs == [2]


def test_metadata_persistence(temp_metadata_path):
    """Verify metadata saves and loads correctly."""
    mgr1 = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    mgr1.add_vector(0, "doc_a", 0, "Text")
    mgr1.save()
    
    mgr2 = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    assert mgr2.metadata.total_vectors == 1
    assert mgr2.get_vector_mapping(0)["doc_name"] == "doc_a"


def test_add_vectors_incremental(mock_index, temp_metadata_path):
    """Verify incremental vector addition."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    
    dim = 384
    new_vectors = [np.random.rand(dim).astype("float32") for _ in range(3)]
    
    initial_size = mock_index.ntotal
    updated_index, vids = add_vectors_incremental(
        mock_index,
        new_vectors,
        "doc_c",
        [0, 1, 2],
        ["Text 0", "Text 1", "Text 2"],
        mgr,
    )
    
    assert updated_index.ntotal == initial_size + 3
    assert len(vids) == 3
    assert vids == [initial_size, initial_size + 1, initial_size + 2]
    assert mgr.metadata.total_vectors == 3


def test_remove_vectors_incremental(mock_index, temp_metadata_path):
    """Verify incremental vector removal."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    
    # Populate metadata for existing index vectors
    for i in range(mock_index.ntotal):
        mgr.add_vector(i, f"doc_{i}", 0, f"Text {i}")
    
    initial_size = mock_index.ntotal
    updated_index = remove_vectors_incremental(
        mock_index,
        [0, 2],  # Remove first and third vectors
        mgr,
    )
    
    assert updated_index.ntotal == initial_size - 2
    assert 0 not in mgr.get_vectors_for_document("doc_0")


def test_consistency_check_matching(mock_index, temp_metadata_path):
    """Verify consistency check passes when index and metadata match."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    
    for i in range(mock_index.ntotal):
        mgr.add_vector(i, f"doc_{i}", 0, f"Text {i}")
    
    status = get_index_consistency_status(mock_index, mgr)
    
    assert status["is_consistent"] is True
    assert status["index_size"] == status["metadata_size"]


def test_consistency_check_mismatch(mock_index, temp_metadata_path):
    """Verify consistency check fails when sizes don't match."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    
    # Only add metadata for some vectors
    for i in range(mock_index.ntotal - 2):
        mgr.add_vector(i, f"doc_{i}", 0, f"Text {i}")
    
    status = get_index_consistency_status(mock_index, mgr)
    
    assert status["is_consistent"] is False
    assert status["index_size"] != status["metadata_size"]
    assert len(status["mismatches"]) > 0


def test_metadata_reset(temp_metadata_path):
    """Verify metadata reset clears all tracking."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    mgr.add_vector(0, "doc_a", 0, "Text")
    mgr.add_vector(1, "doc_b", 0, "Text")
    
    mgr.reset()
    
    assert mgr.metadata.total_vectors == 0
    assert len(mgr.metadata.vector_mappings) == 0


def test_incremental_workflow_add_update_remove(mock_index, temp_metadata_path):
    """End-to-end test: add, update, remove vectors."""
    mgr = FAISSIndexMetadata(metadata_path=temp_metadata_path)
    index = mock_index
    
    dim = 384
    new_vecs = [np.random.rand(dim).astype("float32") for _ in range(2)]
    start_id = index.ntotal
    
    # Add new vectors
    index, new_vids = add_vectors_incremental(
        index, new_vecs, "doc_new", [0, 1], ["Text A", "Text B"], mgr
    )
    assert index.ntotal == start_id + 2
    
    # Update metadata for one vector
    mgr.update_vector(new_vids[0], "Updated text A")
    mapping = mgr.get_vector_mapping(new_vids[0])
    assert mapping["embedding_text"] == "Updated text A"
    
    # Remove one vector
    index = remove_vectors_incremental(index, [new_vids[0]], mgr)
    assert index.ntotal == start_id + 1
    assert mgr.get_vector_mapping(new_vids[1]) is None
    doc_new_vecs = mgr.get_vectors_for_document("doc_new")
    assert len(doc_new_vecs) == 1
    assert mgr.get_vector_mapping(doc_new_vecs[0])["embedding_text"] == "Text B"