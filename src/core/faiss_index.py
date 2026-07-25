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
from typing import Dict, List, Optional, Tuple

# FAISS has no official type stubs; suppress Pylance false positives
import faiss  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)

# ── Threshold for automatic index selection ────────────────────────────────────
_IVF_THRESHOLD = 5_000  # Switch from flat to IVF when vectors exceed this


class ChunkRecord:
    """Stores metadata for a single chunk stored in the FAISS index."""

    __slots__ = ("doc_name", "chunk_index", "chunk_text")

    def __init__(self, doc_name: str, chunk_index: int, chunk_text: str):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text

    def __repr__(self):
        preview = self.chunk_text[:60].replace("\n", " ")
        return f"ChunkRecord({self.doc_name!r}, idx={self.chunk_index}, '{preview}…')"


def build_index(
    embeddings: Dict[str, np.ndarray],
    chunked_docs: Dict[str, List[str]],
    index_type: str = "auto",
    nlist: Optional[int] = None,
    nprobe: int = 10,
) -> Tuple[faiss.Index, List[ChunkRecord]]:
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
    all_vectors: List[np.ndarray] = []
    registry: List[ChunkRecord] = []

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue
        for i, (vec, text) in enumerate(zip(emb, chunks)):
            all_vectors.append(vec.astype("float32"))
            registry.append(ChunkRecord(doc_name, i, text))

    if not all_vectors:
        return faiss.IndexFlatIP(dim), registry

    matrix = np.vstack(all_vectors)
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

    return index, registry


def search_similar_chunks(
    query_embedding: np.ndarray,
    index: faiss.Index,
    registry: List[ChunkRecord],
    top_k: int = 10,
    exclude_doc: Optional[str] = None,
    threshold: float = 0.0,
) -> List[Tuple[ChunkRecord, float]]:
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
        if idx < 0 or idx >= len(registry):
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


def find_plagiarised_chunks(
    embeddings: Dict[str, np.ndarray],
    chunked_docs: Dict[str, List[str]],
    index: faiss.Index,
    registry: List[ChunkRecord],
    threshold: float = 0.75,
    top_k: int = 5,
) -> List[Dict]:
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
                            chunks[chunk_idx] if chunk_idx < len(chunks) else ""
                        ),
                        "match_doc": record.doc_name,
                        "match_chunk_text": record.chunk_text,
                        "similarity": round(score, 4),
                    }
                )

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


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
) -> faiss.Index:
    """Build a FAISS index from a pre-computed 2D numpy matrix of embeddings."""
    dim = 384
    if matrix.size == 0 or matrix.shape[0] == 0:
        return faiss.IndexFlatIP(dim)

    n_vectors = matrix.shape[0]

    # Resolve index type
    if index_type == "auto":
        index_type = "ivf" if n_vectors >= _IVF_THRESHOLD else "flat"

    if index_type == "ivf":
        if nlist is None:
            nlist = max(4, int(np.sqrt(n_vectors)))
        nlist = min(nlist, n_vectors)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix.astype("float32"))
        index.add(matrix.astype("float32"))
        index.nprobe = nprobe
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(matrix.astype("float32"))

    return index


def validate_index(
    index: Optional[faiss.Index], expected_count: int, expected_dimension: int = 384
) -> bool:
    """Check whether a loaded index matches the expected vector count and dimension."""
    if index is None:
        return False
    try:
        return bool(index.ntotal == expected_count and index.d == expected_dimension)
    except Exception:
        return False


def load_or_rebuild_index(filepath: str) -> Tuple[faiss.Index, List[ChunkRecord], bool]:
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
        from src.errors import faiss_emb_registry_mismatch

        raise ValueError(
            faiss_emb_registry_mismatch(emb_count=n_matrix, reg_count=n_registry)
        )

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
