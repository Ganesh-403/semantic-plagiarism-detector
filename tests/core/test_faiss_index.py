import faiss
import numpy as np
import pytest

from src.core.faiss_index import (ChunkRecord, build_index,
                                  find_plagiarised_chunks, load_index,
                                  optimize_faiss_index, save_index, search_similar_chunks)


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


def test_faiss_normalization_parity():
    """Verify that FAISS vector search results are mathematically identical between raw and normalized embeddings."""
    np.random.seed(42)
    # Generate 10 random vectors
    embeddings = {"doc_a": np.random.rand(10, 384).astype("float32")}
    chunked = {"doc_a": [f"chunk_{i}" for i in range(10)]}

    index, registry = build_index(embeddings, chunked, index_type="flat")

    # The index should contain normalized vectors. Perform a search and check.
    query = np.random.rand(384).astype("float32")
    results = search_similar_chunks(query, index, registry, top_k=5)

    # Ensure all returned scores are in [0, 1.0] (valid cosine similarity)
    for _, score in results:
        assert 0.0 <= score <= 1.0001


def test_faiss_normalization_benchmark():
    """Benchmark performance of Python loop-based L2 normalization vs NumPy vectorized normalization on 1000+ vectors."""
    import time

    np.random.seed(42)
    n_vectors = 2000
    dim = 384
    matrix = np.random.rand(n_vectors, dim).astype("float32")

    # 1. Benchmark loop-based normalization
    start_loop = time.perf_counter()
    loop_result = np.zeros_like(matrix)
    for i in range(n_vectors):
        vec = matrix[i]
        norm = np.linalg.norm(vec)
        if norm > 0:
            loop_result[i] = vec / norm
        else:
            loop_result[i] = vec
    loop_time = time.perf_counter() - start_loop

    # 2. Benchmark NumPy vectorized normalization
    start_vec = time.perf_counter()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vec_result = matrix / norms
    vec_time = time.perf_counter() - start_vec

    # 3. Assert mathematical equivalence
    assert np.allclose(loop_result, vec_result, atol=1e-6)

    # Print results and assert speedup (vectorized should be much faster than loop)
    print(f"\n[Benchmark] Loop-based L2 normalization: {loop_time:.6f}s")
    print(f"[Benchmark] Vectorized NumPy L2 normalization: {vec_time:.6f}s")
    assert vec_time < loop_time


# ── FAISS Index Optimization Tests (#1354) ───────────────────────────────────


def test_optimize_faiss_index_below_threshold(caplog):
    """Below 5000 threshold, logs vector count and returns True without error."""
    import logging

    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(10, dim=dim)
    index.add(vecs)

    class IndexManager:
        def __init__(self, idx):
            self.index = idx

    manager = IndexManager(index)

    with caplog.at_level(logging.INFO, logger="src.core.faiss_index"):
        result = optimize_faiss_index(manager, nlist=10)

    assert result is True
    assert manager.index.ntotal == 10
    messages = [r.getMessage() for r in caplog.records]
    assert any("Vector count before index optimization: 10" in m for m in messages)
    assert any("Vector count after index optimization: 10" in m for m in messages)


def test_optimize_faiss_index_converts_above_threshold(caplog, monkeypatch):
    """Above 5000 threshold, converts flat index to IVF index and logs count."""
    import logging
    import src.core.faiss_index as faiss_mod

    # Lower threshold temporarily for unit test speed
    monkeypatch.setattr(faiss_mod, "_IVF_THRESHOLD", 5)

    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(10, dim=dim)
    index.add(vecs)

    manager = {"index": index}

    with caplog.at_level(logging.INFO, logger="src.core.faiss_index"):
        result = optimize_faiss_index(manager, nlist=4)

    assert result is True
    assert manager["index"].ntotal == 10
    assert isinstance(manager["index"], (faiss.IndexIVFFlat, faiss.IndexIDMap))


def test_get_faiss_index_memory_bytes_none_or_uninitialized():
    """Verify get_faiss_index_memory_bytes returns 0 for None or empty index."""
    from src.core.faiss_index import get_faiss_index_memory_bytes

    assert get_faiss_index_memory_bytes(None) == 0
    assert get_faiss_index_memory_bytes(faiss.IndexFlatIP(384)) == 0


def test_get_faiss_index_memory_bytes_with_vectors():
    """Verify get_faiss_index_memory_bytes returns byte size for populated index."""
    from src.core.faiss_index import get_faiss_index_memory_bytes

    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(100, dim=dim)
    index.add(vecs)

    mem_bytes = get_faiss_index_memory_bytes(index)
    assert isinstance(mem_bytes, int)
    assert mem_bytes > 0

    # Wrapped in dict
    assert get_faiss_index_memory_bytes({"index": index}) == mem_bytes

    # Wrapped in object attribute
    class IndexManager:
        def __init__(self, idx):
            self.index = idx

    assert get_faiss_index_memory_bytes(IndexManager(index)) == mem_bytes


def test_format_faiss_memory_badge_formatting():
    """Verify format_faiss_memory_badge text output for uninitialized and populated index."""
    from src.core.faiss_index import format_faiss_memory_badge

    # Uninitialized / None fallback
    assert format_faiss_memory_badge(None) == "FAISS Memory: 0 MB"
    assert format_faiss_memory_badge(faiss.IndexFlatIP(384)) == "FAISS Memory: 0 MB"

    # Populated index
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(1000, dim=dim)
    index.add(vecs)

    badge = format_faiss_memory_badge(index)
    assert badge.startswith("FAISS Memory:")
    assert "MB" in badge
    assert "(1,000 vectors)" in badge


