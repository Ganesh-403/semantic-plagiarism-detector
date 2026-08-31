import faiss
import numpy as np
import pytest

from src.core.faiss_index import (
    ChunkRecord,
    FAISSIndex,
    FaissIndexManager,
    add_to_index,
    build_index,
    find_plagiarised_chunks,
    load_index,
    optimize_faiss_index,
    rebuild_index_from_database,
    rebuild_index_from_db,
    remove_document_from_index,
    save_index,
    search_batch_vectors,
    search_index,
    search_similar_chunks,
)

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


def test_search_index_threshold_filtering(two_doc_data):
    """Issue #4036: Test search_index(query_vectors, threshold=0.8) excludes similarity < 0.8."""
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="flat")
    query = embeddings["doc_a"][0]
    threshold = 0.8
    results = search_index(query, index=index, registry=registry, threshold=threshold)
    assert all(r["similarity_score"] >= threshold for r in results)


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


def test_save_index_file_permissions(tmp_path, two_doc_data, monkeypatch):
    """Verify that save_index writes the .faiss index file with restrictive file permissions (0o600)."""
    import os
    import stat

    chmod_calls = []
    orig_chmod = os.chmod

    def mock_chmod(path, mode):
        chmod_calls.append((path, mode))
        try:
            orig_chmod(path, mode)
        except OSError:
            pass

    monkeypatch.setattr(os, "chmod", mock_chmod)

    embeddings, chunked = two_doc_data
    index, _ = build_index(embeddings, chunked, index_type="flat")
    path = str(tmp_path / "test_permissions.faiss")
    save_index(index, path)

    assert os.path.exists(path)
    assert len(chmod_calls) >= 1
    assert chmod_calls[-1][0] == path
    assert chmod_calls[-1][1] == 0o600

    if os.name != "nt":
        file_mode = stat.S_IMODE(os.stat(path).st_mode)
        assert file_mode == 0o600

    loaded = load_index(path)
    assert loaded.ntotal == index.ntotal



def test_chunk_record_repr():
    r = ChunkRecord("my_doc", 0, "Hello world this is a test chunk.")
    assert "my_doc" in repr(r)
    assert "idx=0" in repr(r)


def test_faiss_normalization_parity():
    """Verify that FAISS vector search results are mathematically identical between raw and
    normalized embeddings."""
    np.random.seed(42)
    # Generate 10 random vectors
    embeddings = {"doc_a": np.random.rand(10, 384).astype("float32")}
    chunked = {"doc_a": [f"chunk_{i}" for i in range(10)]}

    index, registry = build_index(embeddings, chunked, index_type="flat")

    # The index should contain normalized vectors. Perform a search and check.
    query = np.random.rand(384).astype("float32")
    query = query / np.linalg.norm(query)
    results = search_similar_chunks(query, index, registry, top_k=5)

    # Ensure all returned scores are in [0, 1.0] (valid cosine similarity)
    for _, score in results:
        assert 0.0 <= score <= 1.0001


def test_faiss_normalization_benchmark():
    """Benchmark performance of Python loop-based vs NumPy vectorized L2 normalization."""
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


def test_search_batch_vectors():
    """Verify that search_batch_vectors successfully queries a batch of vectors and returns correct matrices."""
    # 1. Create populated index
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(10, dim=dim)
    index.add(vecs)

    # 2. Search a batch of 3 vectors
    query_batch = _unit_vecs(3, dim=dim)
    distances, indices = search_batch_vectors(query_batch, index, top_k=5)

    # Check shapes
    assert distances.shape == (3, 5)
    assert indices.shape == (3, 5)

    # Check types and basic properties
    assert distances.dtype == np.float32
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all(indices >= 0)
    assert np.all(indices < 10)

    # 3. Test alternate argument order (index first, then query_matrix)
    distances_alt, indices_alt = search_batch_vectors(index, query_batch, top_k=5)
    assert np.array_equal(distances, distances_alt)
    assert np.array_equal(indices, indices_alt)

    # 4. Test single 1D vector (should be reshaped and searched)
    single_vector = query_batch[0]
    dist_single, ind_single = search_batch_vectors(single_vector, index, top_k=5)
    assert dist_single.shape == (1, 5)
    assert ind_single.shape == (1, 5)

    # 5. Invalid arguments checking
    with pytest.raises(TypeError):
        search_batch_vectors("not-a-numpy-array", index)

    with pytest.raises(ValueError):
        search_batch_vectors(query_batch, "not-a-faiss-index")


# ── remove_document_from_index Tests (#4032) ─────────────────────────────────


def _make_idmap_index(embeddings, chunked):
    """Build an IndexIDMap-wrapped index via add_to_index (mirrors production path)."""
    index = faiss.IndexFlatIP(384)
    index, registry = add_to_index(index, [], embeddings, chunked)
    return index, registry


def test_remove_document_from_index_prunes_registry(two_doc_data, monkeypatch):
    """Normal deletion: after removing doc_a the registry/index counts are consistent."""
    embeddings, chunked = two_doc_data
    index, registry = _make_idmap_index(embeddings, chunked)

    assert isinstance(
        index, faiss.IndexIDMap
    ), "Precondition: index must be IDMap-wrapped"
    assert index.ntotal == 6
    assert len(registry) == 6

    def mock_compact_index(idx, reg):
        new_idx = faiss.IndexFlatIP(384)
        if len(reg) > 0:
            new_idx.add(np.random.rand(len(reg), 384).astype("float32"))
        return new_idx, reg

    import src.core.faiss_index as faiss_mod

    monkeypatch.setattr(faiss_mod, "compact_index", mock_compact_index)

    new_index, new_registry = remove_document_from_index(index, registry, "doc_a")

    # Registry must only contain doc_b records
    assert len(new_registry) == 3
    assert all(r.doc_name == "doc_b" for r in new_registry)
    assert not any(r.doc_name == "doc_a" for r in new_registry)

    # FAISS vector count must match registry length (alignment invariant)
    assert new_index.ntotal == len(new_registry)


def test_remove_document_from_index_nonexistent_doc_noop(two_doc_data, monkeypatch):
    """Removing a doc that is not in the registry is a safe no-op — counts unchanged."""
    embeddings, chunked = two_doc_data
    index, registry = _make_idmap_index(embeddings, chunked)

    ntotal_before = index.ntotal
    reg_len_before = len(registry)

    def mock_compact_index(idx, reg):
        new_idx = faiss.IndexFlatIP(384)
        if len(reg) > 0:
            new_idx.add(np.random.rand(len(reg), 384).astype("float32"))
        return new_idx, reg

    import src.core.faiss_index as faiss_mod

    monkeypatch.setattr(faiss_mod, "compact_index", mock_compact_index)

    new_index, new_registry = remove_document_from_index(
        index, registry, "nonexistent_doc"
    )

    assert new_index.ntotal == ntotal_before
    assert len(new_registry) == reg_len_before


def test_remove_document_from_index_search_excludes_deleted(two_doc_data, monkeypatch):
    """After removal, searching with any query never returns a record from the deleted doc."""
    embeddings, chunked = two_doc_data
    index, registry = _make_idmap_index(embeddings, chunked)

    def mock_compact_index(idx, reg):
        new_idx = faiss.IndexFlatIP(384)
        if len(reg) > 0:
            new_idx.add(np.random.rand(len(reg), 384).astype("float32"))
        return new_idx, reg

    import src.core.faiss_index as faiss_mod

    monkeypatch.setattr(faiss_mod, "compact_index", mock_compact_index)

    new_index, new_registry = remove_document_from_index(index, registry, "doc_a")

    # Use doc_a's own embeddings as queries — the worst case for leaking deleted results.
    for vec in embeddings["doc_a"]:
        results = search_similar_chunks(vec, new_index, new_registry, top_k=10)
        for record, _ in results:
            assert (
                record.doc_name != "doc_a"
            ), f"search returned a ChunkRecord from deleted document 'doc_a': {record!r}"


# ── FAISS k-overflow regression test (#4034) ──────────────────────────────────


def test_search_similar_chunks_k_larger_than_index_size():
    """Regression: search with top_k > ntotal must not return -1-padded results.

    Issue #4034: When top_k exceeds the number of vectors in the index, FAISS
    pads its raw output with -1 sentinel indices.  search_similar_chunks() must
    filter those sentinels and return only the real matches — no more, no fewer.
    """
    np.random.seed(0)
    embeddings = {"doc_a": _unit_vecs(3)}
    chunked = {"doc_a": ["chunk 0", "chunk 1", "chunk 2"]}

    index, registry = build_index(embeddings, chunked, index_type="flat")
    assert index.ntotal == 3  # sanity-check: exactly 3 vectors in the index

    query = _unit_vecs(1)[0]
    results = search_similar_chunks(query, index, registry, top_k=10)

    # Must return exactly 3 matches — the full index — not 10 or fewer
    assert len(results) == 3, f"Expected 3 results (index size), got {len(results)}"
    # Every result must be a valid (ChunkRecord, float) pair — no -1 artifacts
    for record, score in results:
        assert isinstance(record, ChunkRecord)
        assert record.doc_name == "doc_a"
        assert isinstance(score, float)


# ── Dimension mismatch validation tests (#4029) ──────────────────────────────


def test_build_index_dimension_mismatch():
    embeddings = {"doc1": np.random.rand(2, 100).astype("float32")}
    chunked = {"doc1": ["c1", "c2"]}
    with pytest.raises(ValueError, match=r"Embedding dimension mismatch: 100 != 384"):
        build_index(embeddings, chunked)


def test_build_index_from_matrix_dimension_mismatch():
    from src.core.faiss_index import build_index_from_matrix

    matrix = np.random.rand(2, 100).astype("float32")
    with pytest.raises(ValueError, match=r"Embedding dimension mismatch: 100 != 384"):
        build_index_from_matrix(matrix)


def test_add_to_index_dimension_mismatch(two_doc_data):
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked)

    # Try adding vectors with wrong dimension
    bad_embeddings = {"doc2": np.random.rand(2, 100).astype("float32")}
    bad_chunked = {"doc2": ["c1", "c2"]}

    with pytest.raises(
        ValueError, match=rf"Embedding dimension mismatch: 100 != {index.d}"
    ):
        add_to_index(index, registry, bad_embeddings, bad_chunked)
# ── FAISS HNSW Tests (#4030) ──────────────────────────────────────────────────


def test_hnsw_metric_type(two_doc_data):
    """HNSW indexes must be initialized with METRIC_INNER_PRODUCT to ensure cosine similarity correctness."""
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="hnsw")
    base_index = index.index if isinstance(index, faiss.IndexIDMap) else index
    assert base_index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_hnsw_env_var_routing(two_doc_data, monkeypatch):
    """Setting FAISS_INDEX_TYPE='hnsw' should route auto resolution to build an HNSW index."""
    import src.core.config as config

    monkeypatch.setattr(config, "FAISS_INDEX_TYPE", "hnsw")
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="auto")
    base_index = index.index if isinstance(index, faiss.IndexIDMap) else index
    assert isinstance(base_index, faiss.IndexHNSWFlat)


def test_hnsw_ef_construction_search(two_doc_data, monkeypatch):
    """Check that efConstruction and efSearch are wired up from config."""
    import src.core.config as config

    monkeypatch.setattr(config, "FAISS_HNSW_EF_CONSTRUCTION", 99)
    monkeypatch.setattr(config, "FAISS_HNSW_EF_SEARCH", 42)
    embeddings, chunked = two_doc_data
    index, registry = build_index(embeddings, chunked, index_type="hnsw")
    base_index = index.index if isinstance(index, faiss.IndexIDMap) else index
    assert base_index.hnsw.efConstruction == 99
    assert base_index.hnsw.efSearch == 42


def test_remove_document_from_index_hnsw_fallback(two_doc_data, monkeypatch):
    """Removing a doc from HNSW should not crash, and should fall back to compact_index correctly."""
    embeddings, chunked = two_doc_data
    # First, build an IDMap-wrapped HNSW index using add_to_index (simulating real usage)
    index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
    index, registry = add_to_index(index, [], embeddings, chunked)

    # Mock compact_index to just return a dummy flat index with the pruned registry
    # to verify that it was called and the fallback logic worked without trying reconstruct.
    compact_called = False

    def mock_compact_index(idx, reg):
        nonlocal compact_called
        compact_called = True
        new_idx = faiss.IndexFlatIP(384)
        if len(reg) > 0:
            new_idx.add(np.random.rand(len(reg), 384).astype("float32"))
        return new_idx, reg

    import src.core.faiss_index as faiss_mod

    monkeypatch.setattr(faiss_mod, "compact_index", mock_compact_index)

    # Delete doc_a
    new_index, new_registry = remove_document_from_index(index, registry, "doc_a")

    assert compact_called is True
    assert len(new_registry) == 3
    assert all(r.doc_name == "doc_b" for r in new_registry)


# ── rebuild_index_from_db tests (Issue #4031) ─────────────────────────────────


def test_rebuild_index_from_db_populates_index(tmp_path):
    """Verify rebuild_index_from_db reads embeddings from SQLite chunks and builds FAISS index."""
    import sqlite3

    db_path = tmp_path / "test_corpus.db"
    index_path = tmp_path / "test_corpus.index"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    vecs = _unit_vecs(5, dim=384)
    for idx, vec in enumerate(vecs):
        conn.execute(
            "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            (idx, f"doc_{idx}.txt", 0, f"text {idx}", vec.tobytes()),
        )
    conn.commit()
    conn.close()

    total_added = rebuild_index_from_db(db_path=db_path, index_path=index_path)
    assert total_added == 5
    assert index_path.exists()

    loaded = load_index(str(index_path))
    assert loaded.ntotal == 5


def test_rebuild_index_from_db_empty_db(tmp_path):
    """Verify rebuild_index_from_db returns 0 on an empty chunks table."""
    import sqlite3

    db_path = tmp_path / "empty.db"
    index_path = tmp_path / "empty.index"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    total_added = rebuild_index_from_db(db_path=db_path, index_path=index_path)
    assert total_added == 0
    assert index_path.exists()

    loaded = load_index(str(index_path))
    assert loaded.ntotal == 0


def test_rebuild_index_from_db_nonexistent_db(tmp_path):
    """Verify rebuild_index_from_db handles nonexistent database file gracefully."""
    db_path = tmp_path / "nonexistent.db"
    index_path = tmp_path / "fallback.index"

    total_added = rebuild_index_from_db(db_path=db_path, index_path=index_path)
    assert total_added == 0
    assert index_path.exists()

    loaded = load_index(str(index_path))
    assert loaded.ntotal == 0


def test_rebuild_index_from_db_default_paths(tmp_path, monkeypatch):
    """Verify rebuild_index_from_db uses default paths when db_path and index_path are None."""
    import sqlite3
    import src.core.app_config as app_config

    db_path = tmp_path / "default_data" / "corpus.db"
    index_path = tmp_path / "default_data" / "corpus.index"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    vecs = _unit_vecs(3, dim=384)
    for idx, vec in enumerate(vecs):
        conn.execute(
            "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            (idx, f"doc_{idx}.txt", 0, f"text {idx}", vec.tobytes()),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_config, "CORPUS_DB_PATH", db_path)
    monkeypatch.setattr(app_config, "FAISS_INDEX_PATH", index_path)

    total_added = rebuild_index_from_db()
    assert total_added == 3
    assert index_path.exists()

    loaded = load_index(str(index_path))
    assert loaded.ntotal == 3


def test_rebuild_index_from_database_alias():
    """Verify rebuild_index_from_database alias points to rebuild_index_from_db."""
    assert rebuild_index_from_database is rebuild_index_from_db


# ── rebuild_index_from_db after database reset tests (Issue #4065) ────────────


def test_rebuild_index_from_db_after_database_reset(tmp_path):
    """Verify rebuild_index_from_db produces an empty index after all DB records are cleared (Issue #4065)."""
    import sqlite3

    db_path = tmp_path / "reset_corpus.db"
    index_path = tmp_path / "reset_corpus.index"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    vecs = _unit_vecs(4, dim=384)
    for idx, vec in enumerate(vecs):
        conn.execute(
            "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            (idx, f"doc_{idx}.txt", 0, f"text {idx}", vec.tobytes()),
        )
    conn.commit()
    conn.close()

    # Pre-reset state: populated on-disk index backed by a populated database.
    assert rebuild_index_from_db(db_path=db_path, index_path=index_path) == 4
    assert load_index(str(index_path)).ntotal == 4

    # Simulate a full database reset by clearing every chunk record.
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM chunks")
    conn.commit()
    remaining_rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert remaining_rows == 0

    # Rebuild must not crash and must overwrite the stale populated index.
    total_added = rebuild_index_from_db(db_path=db_path, index_path=index_path)
    assert total_added == 0

    assert index_path.exists()
    loaded = load_index(str(index_path))
    assert loaded.ntotal == 0
    assert loaded.d == 384


def test_rebuild_index_from_db_after_reset_rebuilt_index_queryable(tmp_path):
    """Verify the index rebuilt from a cleared database remains queryable without crashing (Issue #4065)."""
    import sqlite3

    db_path = tmp_path / "reset_query.db"
    index_path = tmp_path / "reset_query.index"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            vector_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    vecs = _unit_vecs(3, dim=384)
    for idx, vec in enumerate(vecs):
        conn.execute(
            "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            (idx, f"doc_{idx}.txt", 0, f"text {idx}", vec.tobytes()),
        )
    conn.commit()
    conn.close()

    assert rebuild_index_from_db(db_path=db_path, index_path=index_path) == 3

    # Clear all database records (database reset).
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()

    assert rebuild_index_from_db(db_path=db_path, index_path=index_path) == 0

    # The rebuilt empty index must still answer searches without raising.
    loaded = load_index(str(index_path))
    query = np.random.rand(1, 384).astype("float32")
    distances, indices = loaded.search(query, k=2)
    assert indices.shape == (1, 2)
    assert (indices == -1).all()
    # FAISS reports the "no result" distance sentinel for empty indexes.
    no_result_distance = -np.finfo(np.float32).max
    assert (distances == no_result_distance).all()


# ── FaissIndexManager wrapper tests (Issue #4033) ─────────────────────────────


def test_faiss_index_manager_total_vectors_uninitialized():
    """Verify total_vectors returns 0 when index is None / uninitialized."""
    manager = FaissIndexManager(index=None)
    assert manager.total_vectors == 0
    assert manager.ntotal == 0


def test_faiss_index_manager_total_vectors_empty_init():
    """Verify total_vectors returns 0 when newly initialized with 0 vectors."""
    manager = FaissIndexManager(dimension=384)
    assert manager.total_vectors == 0
    assert manager.ntotal == 0


def test_faiss_index_manager_total_vectors_with_existing_index():
    """Verify total_vectors returns index.ntotal for populated FAISS index."""
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = _unit_vecs(7, dim=dim)
    index.add(vecs)

    manager = FaissIndexManager(index=index)
    assert manager.total_vectors == 7
    assert manager.ntotal == 7


def test_faiss_index_manager_total_vectors_after_adding_vectors():
    """Verify total_vectors updates correctly when vectors are added via manager."""
    manager = FaissIndexManager(dimension=384)
    assert manager.total_vectors == 0

    vecs1 = _unit_vecs(5, dim=384)
    manager.add(vecs1)
    assert manager.total_vectors == 5
    assert manager.ntotal == 5

    vecs2 = _unit_vecs(3, dim=384)
    manager.add(vecs2)
    assert manager.total_vectors == 8
    assert manager.ntotal == 8


def test_faiss_index_manager_search():
    """Verify nearest neighbor search using FaissIndexManager."""
    manager = FaissIndexManager(dimension=384)
    vecs = _unit_vecs(5, dim=384)
    manager.add(vecs)

    distances, indices = manager.search(vecs[0], top_k=3)
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert indices[0][0] == 0


def test_faiss_index_manager_search_empty():
    """Verify search on empty/uninitialized index returns sentinel -1 indices."""
    manager = FaissIndexManager(index=None)
    query = np.random.rand(384).astype("float32")
    distances, indices = manager.search(query, top_k=3)
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (indices == -1).all()


def test_faiss_index_alias():
    """Verify FAISSIndex alias matches FaissIndexManager."""
    assert FAISSIndex is FaissIndexManager
