import faiss
import numpy as np
import pytest

from src.core.faiss_index import (ChunkRecord, build_index,
                                  find_plagiarised_chunks, load_index,
                                  save_index, search_similar_chunks)


def _unit_vecs(n, dim=384):
    """Return n random L2-normalised float32 vectors."""
    vecs = np.random.rand(n, dim).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def two_doc_data():
    np.random.seed(42)
    emb_a = _unit_vecs(3)
    emb_b = _unit_vecs(3)
    embeddings = {"doc_a": emb_a, "doc_b": emb_b}
    chunked = {
        "doc_a": ["chunk a0", "chunk a1", "chunk a2"],
        "doc_b": ["chunk b0", "chunk b1", "chunk b2"],
    }
    return embeddings, chunked


def test_build_index_flat_returns_correct_total(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    assert index.ntotal == 6
    assert len(registry) == 6


def test_build_index_registry_metadata(two_doc_data):
    embeddings, chunked = two_doc_data
    _, registry = build_index(embeddings, chunked, index_type="flat")
    doc_names = {r.doc_name for r in registry}
    assert doc_names == {"doc_a", "doc_b"}
    assert all(isinstance(r.chunk_text, str) for r in registry)


def test_build_index_empty_returns_flat():
    index, registry = build_index({}, {})
    assert isinstance(index, faiss.IndexFlatIP)
    assert len(registry) == 0


def test_build_index_skips_empty_embedding():
    embeddings = {"doc_a": np.array([]), "doc_b": _unit_vecs(2)}
    chunked = {"doc_a": [], "doc_b": ["c0", "c1"]}
    index, registry = build_index(embeddings, chunked, index_type="flat")
    assert index.ntotal == 2


def test_build_index_ivf(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="ivf", nlist=2)
    assert index.ntotal == 6


def test_search_similar_chunks_returns_results(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    query = embeddings["doc_a"][0]
    results = search_similar_chunks(query, index, registry, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, ChunkRecord) for r, _ in results)
    assert all(isinstance(s, float) for _, s in results)


def test_search_similar_chunks_exclude_doc(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    query = embeddings["doc_a"][0]
    results = search_similar_chunks(
        query, index, registry, top_k=5, exclude_doc="doc_a"
    )
    assert all(r.doc_name != "doc_a" for r, _ in results)


def test_search_similar_chunks_threshold_filters(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    query = embeddings["doc_a"][0]
    results = search_similar_chunks(query, index, registry, top_k=5, threshold=0.9999)
    # Very high threshold — may return 0 or 1 (self-match if not excluded)
    assert all(s >= 0.9999 for _, s in results)


def test_find_plagiarised_chunks_deduplicates(two_doc_data):
    embeddings, chunked = two_doc_data
    # Make doc_a and doc_b identical so every chunk matches
    emb = _unit_vecs(2)
    embeddings = {"doc_a": emb, "doc_b": emb}
    chunked = {"doc_a": ["c0", "c1"], "doc_b": ["c0", "c1"]}
    index, registry = build_index(embeddings, chunked, index_type="flat")
    matches = find_plagiarised_chunks(
        embeddings, chunked, index, registry, threshold=0.5
    )
    # Chunk-pairs should not be duplicated (including symmetric duplicates)
    pair_keys = [
        tuple(
            sorted(
                [
                    (m["source_doc"], m["source_chunk_text"]),
                    (m["match_doc"], m["match_chunk_text"]),
                ]
            )
        )
        for m in matches
    ]
    assert len(pair_keys) == len(set(pair_keys))


def test_find_plagiarised_chunks_sorted_descending(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    matches = find_plagiarised_chunks(
        embeddings, chunked, index, registry, threshold=0.0
    )
    sims = [m["similarity"] for m in matches]
    assert sims == sorted(sims, reverse=True)


def test_save_and_load_index(tmp_path, two_doc_data):
    embeddings, chunked = two_doc_data
    index, _ = build_index(embeddings, chunked, index_type="flat")
    path = str(tmp_path / "test.index")
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.ntotal == index.ntotal


def test_chunk_record_repr():
    r = ChunkRecord("my_doc", 0, "Hello world this is a test chunk.")
    assert "my_doc" in repr(r)
    assert "idx=0" in repr(r)
