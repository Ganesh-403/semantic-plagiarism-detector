"""
faiss_index.py
--------------
Builds and queries a FAISS index over all document chunk embeddings.

Why FAISS?
----------
Pairwise cosine similarity is O(N²) — fine for 10 documents, painful for 1000+.
FAISS offers multiple index types for different scale requirements:

Index types available:
  - IndexFlatIP  : Exact inner product (brute-force). O(N) per query.
                   Best for < 10k vectors. No approximation error.
  - IndexIVFFlat : Inverted-file index with Voronoi cells. O(N/nlist × nprobe)
                   per query — significantly faster at scale. Requires training.
                   Best for 10k–10M vectors.

Since embeddings are L2-normalised in embedding_model.py,
inner product == cosine similarity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.metrics import faiss_vectors_gauge
from src.core.text_chunking import ChunkString

# FAISS has no official type stubs; suppress Pylance false positives
import faiss  # type: ignore
import numpy as np
logger = logging.getLogger(__name__)

# ── Threshold for automatic index selection ────────────────────────────────────
_IVF_THRESHOLD = 5_000  # Switch from flat to IVF when vectors exceed this


class ChunkRecord:
    """Stores metadata for a single chunk stored in the FAISS index."""

    __slots__ = ("doc_name", "chunk_index", "chunk_text", "metadata")

    def __init__(
        self,
        doc_name: str,
        chunk_index: int,
        chunk_text: str,
        metadata: Optional[dict] = None,
    ):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text
        self.metadata = metadata or getattr(chunk_text, "metadata", {})

    def __repr__(self):
        preview = self.chunk_text[:60].replace("\n", " ")
        return f"ChunkRecord({self.doc_name!r}, idx={self.chunk_index}, '{preview}…')"


FaissChunkRecord = ChunkRecord



def build_index(
    embeddings: dict[str, np.ndarray],
    chunked_docs: dict[str, list[str]],
    index_type: str = "auto",
    nlist: Optional[int] = None,
    nprobe: int = 10,
) -> tuple[faiss.Index, list[ChunkRecord]]:
    """
    Build a FAISS index over all chunk embeddings.

    Args:
        embeddings:   Dict mapping doc name → embedding array (chunks × 384).
        chunked_docs: Dict mapping doc name → list of chunk strings.
        index_type:   Index selection strategy:
                        'flat' — IndexFlatIP (exact, O(N) per query)
                        'ivf'  — IndexIVFFlat (approximate, faster at scale)
                        'auto' — flat if < 5k vectors, IVF if >= 5k (default)
        nlist:        Number of Voronoi cells for IVF (auto-sized if None).
        nprobe:       Number of cells to visit at query time for IVF (default 10).

    Returns:
        (index, registry) — the FAISS index and a list mapping each vector
        position to its source ChunkRecord.
    """
    dim = 384
    all_vectors: list[np.ndarray] = []
    registry: list[ChunkRecord] = []

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue
        for i, (vec, chunk) in enumerate(zip(emb, chunks)):
            all_vectors.append(vec.astype("float32"))
            if isinstance(chunk, ChunkString):
                registry.append(ChunkRecord(doc_name, i, chunk.text, metadata=chunk.metadata))
            else:
                registry.append(ChunkRecord(doc_name, i, chunk))

    if not all_vectors:
        faiss_vectors_gauge.set(0)
        return faiss.IndexFlatIP(dim), registry
    matrix = np.vstack(all_vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    n_vectors = matrix.shape[0]

    # ── Resolve index type ────────────────────────────────────────────────────
    if index_type == "auto":
        index_type = "ivf" if n_vectors >= _IVF_THRESHOLD else "flat"

    if index_type == "ivf":
        # IVF requires nlist <= n_vectors; auto-size using sqrt heuristic
        if nlist is None:
            nlist = max(4, int(np.sqrt(n_vectors)))
        nlist = min(nlist, n_vectors)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)  # type: ignore[arg-type]
        index.add(matrix)  # type: ignore[arg-type]
        index.nprobe = nprobe
        logger.info(
            f"[faiss_index] Built IndexIVFFlat  "
            f"({n_vectors} vectors, nlist={nlist}, nprobe={nprobe})"
        )
    else:
        # Flat index — exact search, best for small-to-medium collections
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)  # type: ignore[arg-type]
        logger.info(
            f"[faiss_index] Built IndexFlatIP  ({n_vectors} vectors, exact search)"
        )

    faiss_vectors_gauge.set(index.ntotal)
    return index, registry


def search_similar_chunks(    query_embedding: np.ndarray,
    index: faiss.Index,
    registry: list[ChunkRecord],
    top_k: int = 10,
    exclude_doc: Optional[str] = None,
    threshold: float = 0.0,
) -> list[tuple[ChunkRecord, float]]:
    """
    Search the FAISS index for the most similar chunks to a query vector.

    Args:
        query_embedding: 1-D embedding vector (384,).
        index:           FAISS index built by build_index().
        registry:        ChunkRecord list built by build_index().
        top_k:           Number of results to return.
        exclude_doc:     Skip results from this document (for cross-doc search).
        threshold:       Minimum similarity score to include.

    Returns:
        List of (ChunkRecord, similarity_score) tuples, descending by score.
    """
    vec = query_embedding.astype("float32").reshape(1, -1)
    fetch_k = min(top_k * 3, index.ntotal) if exclude_doc else top_k
    fetch_k = max(fetch_k, 1)

    scores, indices = index.search(vec, fetch_k)  # type: ignore[call-arg]

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if idx >= len(registry):
            logger.warning(
                f"[faiss_index] search returned out-of-range index {idx} "
                f"(registry size: {len(registry)}). "
                "Call compact_index() after removals to restore alignment."
            )
            continue
        record = registry[idx]
        if exclude_doc and record.doc_name == exclude_doc:
            continue
        if score < threshold:
            continue
        results.append((record, float(score)))
        if len(results) >= top_k:
            break

    return results


def search_batch_vectors(
    query_matrix: np.ndarray,
    index: faiss.Index,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for a batch of query vectors.

    Supports flexible parameter ordering for convenience: (query_matrix, index)
    or (index, query_matrix).

    Args:
        query_matrix: 2D numpy array of shape (N, dim) containing query vectors.
        index:        FAISS index built by build_index().
        top_k:        Number of nearest neighbors to retrieve. Defaults to 5.

    Returns:
        (distances, indices) - Distance matrix and index ID matrix of shape (N, top_k).
    """
    if isinstance(query_matrix, faiss.Index) or (
        not isinstance(query_matrix, np.ndarray) and hasattr(query_matrix, "search")
    ):
        index, query_matrix = query_matrix, index

    if not isinstance(query_matrix, np.ndarray):
        raise TypeError("query_matrix must be a numpy.ndarray")
    if index is None or not hasattr(index, "search"):
        raise ValueError("index must be a valid FAISS index")

    queries = query_matrix.astype("float32")
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    distances, indices = index.search(queries, top_k)  # type: ignore[call-arg]
    return distances, indices


def find_plagiarised_chunks(
    embeddings: dict[str, np.ndarray],
    chunked_docs: dict[str, list[str]],
    index: faiss.Index,
    registry: list[ChunkRecord],
    threshold: float = 0.75,
    top_k: int = 5,
) -> list[dict]:
    """
    Search every chunk against the FAISS index to find cross-document matches.

    For each chunk, queries the index for nearest neighbours in other documents.
    Deduplicates symmetric pairs so (A,B) and (B,A) appear only once.

    Returns:
        List of match dicts sorted by similarity descending, each containing:
        source_doc, source_chunk_text, match_doc, match_chunk_text, similarity.
    """
    matches = []
    seen_pairs = set()

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue

        for chunk_idx, vec in enumerate(emb):
            results = search_similar_chunks(
                vec,
                index,
                registry,
                top_k=top_k,
                exclude_doc=doc_name,
                threshold=threshold,
            )
            for record, score in results:
                pair_key = tuple(
                    sorted(
                        [(doc_name, chunk_idx), (record.doc_name, record.chunk_index)]
                    )
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                matches.append(
                    {
                        "source_doc": doc_name,
                        "source_chunk_text": (
                            chunks[chunk_idx].text if chunk_idx < len(chunks) else ""
                        ),
                        "match_doc": record.doc_name,
                        "match_chunk_text": record.chunk_text,
                        "similarity": round(score, 4),
                    }
                )

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def add_to_index(
    index: faiss.Index,
    registry: list[ChunkRecord],
    embeddings: dict[str, np.ndarray],
    chunked_docs: dict[str, list[str]],
) -> tuple[faiss.Index, list[ChunkRecord]]:
    """
    Incrementally add new chunk vectors to an existing FAISS index without a full rebuild.

    Wraps the index with ``IndexIDMap`` on first call if it isn't already, then uses
    ``add_with_ids()`` to append vectors with sequential IDs starting from the current
    registry length.

    Args:
        index:      Existing FAISS index (bare or already IDMap-wrapped).
        registry:   Existing chunk registry list.
        embeddings: Dict mapping doc name -> embedding matrix (chunks x dim).
        chunked_docs: Dict mapping doc name -> list of chunk strings.

    Returns:
        (updated_index, updated_registry) — the index may be wrapped in ``IndexIDMap``
        if it was bare on entry.
    """
    new_vectors: list[np.ndarray] = []
    new_registry: list[ChunkRecord] = []

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue
        for i, (vec, chunk) in enumerate(zip(emb, chunks)):
            new_vectors.append(vec.astype("float32"))
            if isinstance(chunk, ChunkString):
                new_registry.append(ChunkRecord(doc_name, i, chunk.text, metadata=chunk.metadata))
            else:
                new_registry.append(ChunkRecord(doc_name, i, chunk))

    if not new_vectors:
        return index, registry

    matrix = np.vstack(new_vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    offset = len(registry)
    ids = np.arange(offset, offset + len(new_vectors), dtype=np.int64)

    # Wrap bare index with IndexIDMap on first incremental add
    if not isinstance(index, faiss.IndexIDMap):
        index = faiss.IndexIDMap(index)

    index.add_with_ids(matrix, ids)
    logger.info(
        f"[faiss_index] Incrementally added {len(new_vectors)} vectors "
        f"(total: {index.ntotal})"
    )
    faiss_vectors_gauge.set(index.ntotal)
    return index, registry + new_registry

def remove_vectors_by_doc(
    index: faiss.Index,
    registry: list[ChunkRecord],
    doc_name: str,
) -> tuple[faiss.Index, list[ChunkRecord]]:
    """
    Remove all vectors belonging to a given document from the index.

    .. note::

       After removal the ID-to-registry alignment is broken.  Call
       :func:`compact_index` to restore alignment, or call
       :func:`load_or_rebuild_index` for a full rebuild.

    Args:
        index:    FAISS index (should be ``IndexIDMap``).
        registry: Current chunk registry.
        doc_name: Document whose vectors should be removed.

    Returns:
        (index, updated_registry) — the registry is filtered but ID indices may
        no longer align with its positions.
    """
    if not isinstance(index, faiss.IndexIDMap):
        logger.warning(
            "[faiss_index] Cannot remove vectors by doc — index is not IDMap-wrapped. "
            "Call load_or_rebuild_index() instead."
        )
        return index, registry

    ids_to_remove = [
        np.int64(i) for i, rec in enumerate(registry) if rec.doc_name == doc_name
    ]
    if not ids_to_remove:
        return index, registry

    selector = faiss.IDSelectorArray(np.array(ids_to_remove, dtype=np.int64))
    index.remove_ids(selector)
    updated_registry = [rec for rec in registry if rec.doc_name != doc_name]

    logger.info(
        f"[faiss_index] Removed {len(ids_to_remove)} vectors for doc '{doc_name}' "
        f"(remaining: {index.ntotal})"
    )
    return index, updated_registry


def compact_index(
    index: faiss.Index,
    registry: list[ChunkRecord],
) -> tuple[faiss.Index, list[ChunkRecord]]:
    """
    Rebuild the index with sequential IDs matching the current registry order.

    Use after :func:`remove_vectors_by_doc` to restore ID-to-registry alignment
    without loading embeddings from the database.

    The underlying embedding matrix is reconstructed by re-wrapping the remaining
    vectors from the index itself (for flat indexes) or by loading from the
    database (for IVF indexes).
    """
    from src.db.corpus_db import get_all_embeddings

    matrix = get_all_embeddings()
    n_matrix = matrix.shape[0] if (matrix is not None and matrix.size > 0) else 0
    if n_matrix != len(registry):
        raise ValueError(
            f"Embedding count ({n_matrix}) does not match registry size ({len(registry)})"
        )

    if n_matrix == 0:
        dim = 384
        return faiss.IndexFlatIP(dim), registry

    nprobe = getattr(index, "nprobe", 10)
    new_index = build_index_from_matrix(matrix, nprobe=nprobe)
    return new_index, registry


def save_index(index: faiss.Index, path: str) -> None:
    """Persist a FAISS index to disk."""
    faiss.write_index(index, path)
    logger.info(f"[faiss_index] Index saved → {path}  ({index.ntotal} vectors)")


def load_index(path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    index = faiss.read_index(path)
    logger.info(f"[faiss_index] Index loaded ← {path}  ({index.ntotal} vectors)")
    return index


def build_index_from_matrix(
    matrix: np.ndarray,
    index_type: str = "auto",
    nlist: Optional[int] = None,
    nprobe: int = 10,
    *,
    use_id_map: bool = True,
) -> faiss.Index:
    """Build a FAISS index from a pre-computed 2D numpy matrix of embeddings.

    When *use_id_map* is True (default) the index is wrapped with ``IndexIDMap``
    so that ``add_to_index()`` and ``remove_vectors_by_doc()`` can be used later
    without a full rebuild.
    """
    dim = 384
    if matrix.size == 0 or matrix.shape[0] == 0:
        return faiss.IndexFlatIP(dim)

    n_vectors = matrix.shape[0]
    mat = matrix.astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    # Resolve index type
    if index_type == "auto":
        index_type = "ivf" if n_vectors >= _IVF_THRESHOLD else "flat"

    if index_type == "ivf":
        if nlist is None:
            nlist = max(4, int(np.sqrt(n_vectors)))
        nlist = min(nlist, n_vectors)

        quantizer = faiss.IndexFlatIP(dim)
        base = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        base.train(mat)

        if use_id_map:
            ids = np.arange(n_vectors, dtype=np.int64)
            index = faiss.IndexIDMap(base)
            index.add_with_ids(mat, ids)
            base.nprobe = nprobe
        else:
            base.add(mat)
            base.nprobe = nprobe
            index = base
    else:
        base = faiss.IndexFlatIP(dim)
        if use_id_map:
            ids = np.arange(n_vectors, dtype=np.int64)
            index = faiss.IndexIDMap(base)
            index.add_with_ids(mat, ids)
        else:
            base.add(mat)
            index = base

    faiss_vectors_gauge.set(index.ntotal)
    return index


def validate_index(    index: Optional[faiss.Index], expected_count: int, expected_dimension: int = 384
) -> bool:
    """Check whether a loaded index matches the expected vector count and dimension."""
    if index is None:
        return False
    try:
        return bool(index.ntotal == expected_count and index.d == expected_dimension)
    except Exception:
        return False


def load_or_rebuild_index(filepath: str) -> tuple[faiss.Index, list[ChunkRecord], bool]:
    """
    Load a FAISS index from disk if valid, otherwise rebuild it from corpus.db.
    Returns (index, registry, recovered_flag).
    """
    import os

    from src.db.corpus_db import get_all_embeddings, get_chunk_registry

    matrix = get_all_embeddings()
    registry = get_chunk_registry()

    n_matrix = matrix.shape[0] if (matrix is not None and matrix.size > 0) else 0
    n_registry = len(registry)

    if n_matrix != n_registry:
        from src.errors import FAISS_EMB_REGISTRY_MISMATCH

        raise ValueError(FAISS_EMB_REGISTRY_MISMATCH.format(emb_count=n_matrix, reg_count=n_registry))

    if os.path.exists(filepath):
        try:
            index = load_index(filepath)
            if validate_index(index, n_matrix, 384):
                return index, registry, False
        except Exception:
            pass

    index = build_index_from_matrix(matrix)
    save_index(index, filepath)
    return index, registry, True


# ── FAISS Index Optimization Helper (Issue #1354) ───────────────────────────


def optimize_faiss_index(index_manager: Any, nlist: int = 100) -> bool:
    """Optimize FAISS index structures by converting flat index to trained IVF quantizer.

    When total vectors exceed 5,000, converts IndexFlatIP to IndexIVFFlat for scale.
    Logs vector count before and after index optimization.

    Args:
        index_manager: FAISS index manager instance, dict containing 'index', or raw FAISS Index.
        nlist: Number of Voronoi cells for IVF quantization (default: 100).

    Returns:
        bool: True if optimization was performed or index is valid; False otherwise.
    """
    index = None
    manager_type = "direct"

    if hasattr(index_manager, "index"):
        index = getattr(index_manager, "index")
        manager_type = "attr"
    elif isinstance(index_manager, dict) and "index" in index_manager:
        index = index_manager["index"]
        manager_type = "dict"
    elif hasattr(index_manager, "ntotal"):
        index = index_manager
        manager_type = "direct"

    if index is None:
        logger.warning("[faiss_index] Unable to resolve index object for optimization.")
        return False

    count_before = getattr(index, "ntotal", 0)
    logger.info(f"[faiss_index] Vector count before index optimization: {count_before}")

    if count_before > _IVF_THRESHOLD:
        dim = getattr(index, "d", 384)
        base_index = index.index if isinstance(index, faiss.IndexIDMap) else index

        vectors = np.zeros((count_before, dim), dtype="float32")
        for i in range(count_before):
            vectors[i] = base_index.reconstruct(i)

        nlist_actual = max(4, min(nlist, count_before))
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(
            quantizer, dim, nlist_actual, faiss.METRIC_INNER_PRODUCT
        )
        ivf_index.train(vectors)

        if isinstance(index, faiss.IndexIDMap):
            new_index = faiss.IndexIDMap(ivf_index)
            ids = np.arange(count_before, dtype=np.int64)
            new_index.add_with_ids(vectors, ids)
        else:
            ivf_index.add(vectors)
            new_index = ivf_index

        if manager_type == "attr":
            setattr(index_manager, "index", new_index)
        elif manager_type == "dict":
            index_manager["index"] = new_index

        count_after = new_index.ntotal
        logger.info(
            f"[faiss_index] Vector count after index optimization: {count_after}"
        )
        return True
    else:
        logger.info(
            f"[faiss_index] Vector count after index optimization: {count_before}"
        )
        return True


# ── FAISS Memory Footprint Helper (Issue #1563) ───────────────────────────────


def get_faiss_index_memory_bytes(index: Optional[Any] = None) -> int:
    """Calculate the RAM memory footprint of a FAISS vector index in bytes.

    Args:
        index: FAISS index instance, IndexIDMap wrapper, dict containing 'index',
               or object with an 'index' attribute.

    Returns:
        int: Memory footprint in bytes, or 0 if index is None, empty, or uninitialized.
    """
    if index is None:
        return 0

    # Unwrap object or dictionary wrapping FAISS index
    if hasattr(index, "index") and not isinstance(index, faiss.Index):
        index = getattr(index, "index")
    elif isinstance(index, dict) and "index" in index:
        index = index["index"]

    if index is None:
        return 0

    try:
        ntotal = getattr(index, "ntotal", 0)
        if not ntotal or ntotal <= 0:
            return 0

        # Exact serialized byte footprint from FAISS serializer if available
        buf = faiss.serialize_index(index)
        return int(getattr(buf, "nbytes", len(buf)))
    except Exception:
        # Fallback estimation if serialize_index fails or for mocks
        try:
            ntotal = getattr(index, "ntotal", 0)
            dim = getattr(index, "d", 384)
            if ntotal > 0:
                bytes_per_vec = dim * 4 + (
                    8 if isinstance(index, faiss.IndexIDMap) else 0
                )
                return int(ntotal * bytes_per_vec)
        except Exception:
            pass
        return 0


def format_faiss_memory_badge(index: Optional[Any] = None) -> str:
    """Format the FAISS vector index memory footprint badge text for display.

    Example output:
        "FAISS Memory: 12.4 MB (10,000 vectors)"
        "FAISS Memory: 0 MB" (if uninitialized/empty)

    Args:
        index: FAISS index instance or wrapper.

    Returns:
        str: Formatted memory badge string.
    """
    bytes_val = get_faiss_index_memory_bytes(index)
    if bytes_val <= 0:
        return "FAISS Memory: 0 MB"

    unwrapped = index
    if hasattr(index, "index") and not isinstance(index, faiss.Index):
        unwrapped = getattr(index, "index")
    elif isinstance(index, dict) and "index" in index:
        unwrapped = index["index"]

    vector_count = getattr(unwrapped, "ntotal", 0) if unwrapped is not None else 0
    mb_val = bytes_val / (1024 * 1024)

    if vector_count > 0:
        return f"FAISS Memory: {mb_val:.1f} MB ({vector_count:,} vectors)"
    return f"FAISS Memory: {mb_val:.1f} MB"
