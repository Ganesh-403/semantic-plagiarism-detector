"""
src/core/similarity.py
----------------------
Computes semantic similarity between documents at two levels:
  1. Document-level  – single score per pair (mean-pooled embeddings)
  2. Chunk-level     – max-similarity per chunk pair (detects local plagiarism)

Uses cosine similarity. Since embeddings are L2-normalised in embedding_model.py,
cosine similarity reduces to the dot product, making this very fast.

Recent Additions (Issue #1956):
- Added find_cross_lingual_matches() for back-translated chunk matching.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

import faiss  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import (
    DEFAULT_THRESHOLDS,
    PLAGIARISM_THRESHOLD,
    is_plagiarism,
    severity_from_score,
)
from src.core.cross_lingual import detect_chunk_language

# ── Distance / similarity conversion ──────────────────────────────────────────


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert a cosine distance to a standardized cosine similarity score.

    Formula:
        similarity = max(0.0, min(1.0, 1.0 - distance))

    Args:
        distance: Cosine distance value (typically in [0.0, 2.0]).

    Returns:
        A float similarity score strictly bounded in [0.0, 1.0].
    """
    if isinstance(distance, np.ndarray):
        return np.clip(1.0 - distance, 0.0, 1.0)
    return float(max(0.0, min(1.0, 1.0 - distance)))


# ── Validation helpers ─────────────────────────────────────────────────────────


def _validated_batch_size(batch_size: Optional[int]) -> Optional[int]:
    """Return a safe integer batch size or None for unbatched execution."""
    from src.errors import SIM_BATCH_SIZE_INVALID

    if batch_size is None:
        return None
    if isinstance(batch_size, bool):
        raise ValueError(SIM_BATCH_SIZE_INVALID)
    if isinstance(batch_size, (float, np.floating)):
        if not float(batch_size).is_integer():
            raise ValueError(SIM_BATCH_SIZE_INVALID)
        batch_size = int(batch_size)
    try:
        size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(SIM_BATCH_SIZE_INVALID) from exc
    return size if size > 0 else None


def _apply_min_percentile_filter(
    matrix: pd.DataFrame | np.ndarray,
    min_percentile: Optional[float],
) -> pd.DataFrame | np.ndarray:
    """Zero out similarity scores below the given percentile threshold.

    The percentile is computed over the off-diagonal scores only, so a
    document's similarity with itself (always 1.0) doesn't skew the cutoff.
    """
    if min_percentile is None:
        return matrix

    if not (0.0 <= min_percentile <= 100.0):
        raise ValueError(
            f"min_percentile must be between 0 and 100, got {min_percentile}."
        )

    values = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    if values.size == 0:
        return matrix

    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]
    else:
        off_diagonal = values.flatten()

    if off_diagonal.size == 0:
        return matrix

    threshold = np.percentile(off_diagonal, min_percentile)
    filtered = np.where(values >= threshold, values, 0.0)

    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(filtered, index=matrix.index, columns=matrix.columns)
    return filtered


# ── Distance-based similarity ──────────────────────────────────────────────────


def manhattan_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """Return normalized Manhattan similarity for equally shaped arrays.

    The Manhattan (L1) distance is converted to similarity using
    ``1 / (1 + distance)``. Identical inputs therefore return ``1.0``;
    larger distances approach ``0.0`` without ever producing a value outside
    the inclusive ``[0.0, 1.0]`` range.

    Args:
        vec_a: First numeric vector or array.
        vec_b: Second numeric vector or array.

    Returns:
        A finite Python ``float`` between ``0.0`` and ``1.0``.

    Raises:
        TypeError: If either input cannot be converted to a numeric array.
        ValueError: If shapes differ, either input is empty, or either input
            contains NaN or infinity.
    """
    try:
        array_a = np.asarray(vec_a, dtype=np.float64)
        array_b = np.asarray(vec_b, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Manhattan similarity requires numeric array inputs.") from exc

    if array_a.shape != array_b.shape:
        raise ValueError("Manhattan similarity requires arrays with matching shapes.")

    if array_a.size == 0:
        raise ValueError("Manhattan similarity requires non-empty arrays.")

    if not np.all(np.isfinite(array_a)) or not np.all(np.isfinite(array_b)):
        raise ValueError("Manhattan similarity requires finite numeric values.")

    distance = float(np.sum(np.abs(array_a - array_b), dtype=np.float64))
    similarity = 1.0 / (1.0 + distance)

    # Protect the public contract from tiny floating-point excursions.
    return float(np.clip(similarity, 0.0, 1.0))


# ── Document-level similarity ──────────────────────────────────────────────────


def document_similarity_matrix(
    doc_embeddings: Dict[str, np.ndarray] | np.ndarray | List[np.ndarray],
    batch_size: Optional[int] = None,
    min_threshold: float = 0.0,
    min_percentile: Optional[float] = None,
    top_k: Optional[int] = None,
    candidate_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> pd.DataFrame | np.ndarray:
    """
    Build an N×N cosine similarity matrix between all document pairs.

    Optionally pre-filters comparisons using an ephemeral FAISS index (when top_k is provided)
    or an explicit candidate_pairs set, drastically reducing direct cosine similarity computations.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array, or direct array/list of embeddings.
        batch_size: Optional number of documents to compare per batch.
        min_threshold: Minimum similarity score to keep; values below this will be 0.0.
        min_percentile: Optional percentile threshold for filtering.
        top_k: Optional top-K nearest neighbors to pre-filter document pairs using FAISS.
        candidate_pairs: Optional pre-computed candidate pair set (doc_a, doc_b).

    Returns:
        Symmetric pandas DataFrame or numpy ndarray with similarity values.
    """
    if isinstance(doc_embeddings, (np.ndarray, list)):
        stacked = np.array(doc_embeddings)
        if stacked.ndim == 1 or stacked.size == 0:
            return np.array([[]])
        if top_k is not None and top_k > 0 and stacked.shape[0] > top_k:
            n_items = stacked.shape[0]
            matrix = np.zeros((n_items, n_items))
            np.fill_diagonal(matrix, 1.0)
            dim = stacked.shape[1]
            index = faiss.IndexFlatIP(dim)
            matrix_f32 = stacked.astype("float32")
            index.add(matrix_f32)
            _, indices = index.search(matrix_f32, min(top_k + 1, n_items))
            for i in range(n_items):
                for j in indices[i]:
                    if j < n_items and j != i:
                        sim_val = float(np.dot(stacked[i], stacked[j]))
                        sim_val = float(np.clip(sim_val, 0.0, 1.0))
                        if sim_val >= min_threshold:
                            matrix[i, j] = sim_val
                            matrix[j, i] = sim_val
            return _apply_min_percentile_filter(matrix, min_percentile)

        sim = np.clip(cosine_similarity(stacked), 0.0, 1.0)
        sim = np.where(sim < min_threshold, 0.0, sim)
        return _apply_min_percentile_filter(sim, min_percentile)

    doc_names = list(doc_embeddings.keys())
    n = len(doc_names)

    # Build document-level vectors (mean pool over chunks)
    doc_vectors = []
    for name in doc_names:
        emb = doc_embeddings[name]
        if isinstance(emb, np.ndarray):
            if emb.ndim == 2 and emb.shape[0] > 0:
                vec = np.mean(emb, axis=0)
            elif emb.ndim == 1 and emb.shape[0] > 0:
                vec = emb
            else:
                vec = np.zeros(384)
        else:
            vec = np.zeros(384)
        doc_vectors.append(vec)

    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)

    if doc_vectors:
        stacked = np.vstack(doc_vectors)

        # Pre-filtering with FAISS top_k or candidate_pairs
        active_candidates = candidate_pairs
        if active_candidates is None and top_k is not None and top_k > 0 and n > top_k:
            active_candidates = find_candidate_pairs(
                doc_names, doc_vectors, top_k=top_k
            )

        if active_candidates is not None:
            name_to_idx = {name: i for i, name in enumerate(doc_names)}
            for a, b in active_candidates:
                if a in name_to_idx and b in name_to_idx:
                    i, j = name_to_idx[a], name_to_idx[b]
                    vec_i = stacked[i].reshape(1, -1)
                    vec_j = stacked[j].reshape(1, -1)
                    sim_val = float(cosine_similarity(vec_i, vec_j)[0, 0])
                    sim_val = float(np.clip(sim_val, 0.0, 1.0))
                    if sim_val >= min_threshold:
                        matrix[i, j] = sim_val
                        matrix[j, i] = sim_val
        else:
            safe_batch_size = _validated_batch_size(batch_size)
            if safe_batch_size is None:
                sim = cosine_similarity(stacked)
                sim = np.clip(sim, 0.0, 1.0)
                matrix = np.where(sim < min_threshold, 0.0, sim)
            else:
                for start in range(0, n, safe_batch_size):
                    end = min(start + safe_batch_size, n)
                    sim = cosine_similarity(stacked[start:end], stacked)
                    sim = np.clip(sim, 0.0, 1.0)
                    matrix[start:end] = np.where(sim < min_threshold, 0.0, sim)

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return _apply_min_percentile_filter(df, min_percentile)


def compute_similarity_matrix(
    embeddings: Dict[str, np.ndarray] | np.ndarray | List[np.ndarray],
    batch_size: Optional[int] = None,
    min_threshold: float = 0.0,
    min_percentile: Optional[float] = None,
    top_k: Optional[int] = None,
    candidate_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> pd.DataFrame | np.ndarray:
    """
    Direct alias/wrapper for document_similarity_matrix to maintain backwards compatibility
    with app/streamlit_app.py and external modules.
    """
    return document_similarity_matrix(
        embeddings,
        batch_size=batch_size,
        min_threshold=min_threshold,
        min_percentile=min_percentile,
        top_k=top_k,
        candidate_pairs=candidate_pairs,
    )


# ── Hybrid similarity (lexical + semantic) ─────────────────────────────────────


def hybrid_similarity_matrix(
    semantic_df: pd.DataFrame, lexical_df: pd.DataFrame, w: float = 0.7
) -> pd.DataFrame:
    """
    Combine semantic and lexical similarity matrices using a weighted formula.
    """
    if not (0.0 <= w <= 1.0):
        from src.errors import SIM_WEIGHT_OUT_OF_RANGE

        raise ValueError(SIM_WEIGHT_OUT_OF_RANGE.format(w=w))

    if semantic_df.shape != lexical_df.shape:
        from src.errors import SIM_SHAPE_MISMATCH

        raise ValueError(SIM_SHAPE_MISMATCH)
    if not semantic_df.index.equals(lexical_df.index) or not semantic_df.columns.equals(
        lexical_df.columns
    ):
        from src.errors import SIM_INDEX_MISMATCH

        raise ValueError(SIM_INDEX_MISMATCH)

    hybrid_df = w * semantic_df + (1 - w) * lexical_df
    return hybrid_df


def _compute_bm25_similarity(
    doc_a: str,
    doc_b: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calculate BM25 relevance score between document pairs, normalized in [0.0, 1.0]."""
    if (
        not doc_a
        or not doc_b
        or not isinstance(doc_a, str)
        or not isinstance(doc_b, str)
    ):
        return 0.0

    import math
    import re
    from collections import Counter

    tokens_a = [t.lower() for t in re.findall(r"\w+", doc_a)]
    tokens_b = [t.lower() for t in re.findall(r"\w+", doc_b)]

    if not tokens_a or not tokens_b:
        return 0.0

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    common_terms = set(freq_a.keys()) & set(freq_b.keys())

    if not common_terms:
        return 0.0

    len_a = len(tokens_a)
    len_b = len(tokens_b)
    avg_len = (len_a + len_b) / 2.0

    N = 2
    all_terms = set(freq_a.keys()) | set(freq_b.keys())
    idf = {}
    for t in all_terms:
        df_t = 2 if (t in freq_a and t in freq_b) else 1
        idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

    score_a = sum(
        idf[t]
        * (freq_b[t] * (k1 + 1.0))
        / (freq_b[t] + k1 * (1.0 - b + b * (len_b / avg_len)))
        for t in common_terms
    )
    score_max_a = sum(
        idf[t]
        * (freq_a[t] * (k1 + 1.0))
        / (freq_a[t] + k1 * (1.0 - b + b * (len_a / avg_len)))
        for t in freq_a.keys()
    )

    if score_max_a == 0:
        return 0.0

    bm25 = score_a / score_max_a
    return float(np.clip(bm25, 0.0, 1.0))


def compute_hybrid_similarity(
    vector_sim: float,
    doc_a: str,
    doc_b: str,
    alpha: float = 0.7,
) -> float:
    """Compute hybrid similarity combining dense vector cosine similarity and BM25 lexical relevance.

    Formula:
        hybrid_score = alpha * vector_sim + (1 - alpha) * bm25_score

    Args:
        vector_sim: Dense vector cosine similarity score.
        doc_a: First document text string.
        doc_b: Second document text string.
        alpha: Weight factor for dense vector similarity (default: 0.7).

    Returns:
        float: Hybrid similarity score strictly bounded in [0.0, 1.0].
    """
    if not (0.0 <= alpha <= 1.0):
        from src.errors import SIM_WEIGHT_OUT_OF_RANGE

        raise ValueError(SIM_WEIGHT_OUT_OF_RANGE.format(w=alpha))

    bm25_score = _compute_bm25_similarity(doc_a, doc_b)
    hybrid_score = alpha * vector_sim + (1.0 - alpha) * bm25_score
    return float(np.clip(hybrid_score, 0.0, 1.0))


# ── Chunk-level similarity (local plagiarism detection) ────────────────────────


def chunk_max_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    batch_size: Optional[int] = None,
) -> float:
    """
    Compute the maximum pairwise cosine similarity between chunks of two documents.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0

    safe_batch_size = _validated_batch_size(batch_size)
    if safe_batch_size is None:
        sim_matrix = cosine_similarity(emb_a, emb_b)
        return float(np.max(sim_matrix))

    max_score = 0.0
    for start_a in range(0, emb_a.shape[0], safe_batch_size):
        end_a = min(start_a + safe_batch_size, emb_a.shape[0])
        for start_b in range(0, emb_b.shape[0], safe_batch_size):
            end_b = min(start_b + safe_batch_size, emb_b.shape[0])
            sim_matrix = cosine_similarity(emb_a[start_a:end_a], emb_b[start_b:end_b])
            max_score = max(max_score, float(np.max(sim_matrix)))
            if max_score >= 1.0:
                return max_score
    return max_score


def chunk_similarity_matrix(
    doc_embeddings: Dict[str, np.ndarray],
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build an N×N matrix where each cell is the MAX chunk-pair similarity.
    """
    doc_names = list(doc_embeddings.keys())
    n = len(doc_names)
    matrix = np.zeros((n, n))

    for i, name_a in enumerate(doc_names):
        for j, name_b in enumerate(doc_names):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                score = chunk_max_similarity(
                    doc_embeddings[name_a],
                    doc_embeddings[name_b],
                    batch_size=batch_size,
                )
                matrix[i][j] = score
                matrix[j][i] = score

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return df


# ── ANN Pre-filtering ──────────────────────────────────────────────────────────


def find_candidate_pairs(
    doc_names: List[str],
    doc_vectors: List[np.ndarray],
    *,
    top_k: int = 10,
) -> Set[Tuple[str, str]]:
    """
    Use a temporary FAISS flat index to find the top-K nearest-neighbour
    document pairs for each document.

    This replaces the O(n²) brute-force pair enumeration with an O(n log n)
    approximate search.  When the number of documents is smaller than *top_k*,
    all pairs are returned (no filtering is needed).

    Args:
        doc_names:   Sorted list of document names.
        doc_vectors: List of document-level embedding vectors (same order).
        top_k:       Number of nearest neighbours to retrieve per document.

    Returns:
        A set of ``(doc_a, doc_b)`` tuples (sorted alphabetically) that
        should be checked for plagiarism.
    """
    n = len(doc_names)
    if n <= top_k:
        return {(doc_names[i], doc_names[j]) for i in range(n) for j in range(i + 1, n)}

    matrix = np.vstack([v.astype("float32") for v in doc_vectors])
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    candidates: Set[Tuple[str, str]] = set()
    for i in range(n):
        query = matrix[i].reshape(1, -1)
        _, indices = index.search(query, top_k + 1)
        for j in indices[0]:
            if j == i or j >= n:
                continue
            pair: Tuple[str, str] = tuple(sorted([doc_names[i], doc_names[j]]))
            candidates.add(pair)

    return candidates


# ── Plagiarism flagging ────────────────────────────────────────────────--------


def flag_plagiarism(
    similarity_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD,
    chunked_docs: dict = None,
    embeddings: dict = None,
    *,
    candidate_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Dict]:
    """Identify document pairs whose similarity reaches the threshold.

    Flagging uses the configurable plagiarism threshold. Severity uses the
    central fixed boundaries: Medium at 0.75 and High at 0.90.

    When *candidate_pairs* is provided (e.g. from :func:`find_candidate_pairs`),
    only those pairs are checked instead of the full upper triangle, which
    significantly reduces computation for large document sets.
    """
    flags = []
    doc_names = similarity_df.columns.tolist()
    name_to_idx = {name: i for i, name in enumerate(doc_names)}

    if candidate_pairs is not None:
        pairs_to_check = [
            (name_to_idx[a], name_to_idx[b])
            for a, b in candidate_pairs
            if a in name_to_idx and b in name_to_idx
        ]
    else:
        pairs_to_check = [
            (i, j) for i in range(len(doc_names)) for j in range(i + 1, len(doc_names))
        ]

    for i, j in pairs_to_check:
        score = float(similarity_df.iloc[i, j])

        if is_plagiarism(score, threshold):
            doc_a = doc_names[i]
            doc_b = doc_names[j]
            matched_length = 0

            if chunked_docs is not None and embeddings is not None:
                sim_matrix = cosine_similarity(embeddings[doc_a], embeddings[doc_b])
                idx_a, idx_b = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
                chunk_text = chunked_docs[doc_a][idx_a]
                matched_length = len(chunk_text.split())

            flags.append(
                {
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "similarity": round(score, 4),
                    "threshold_at_time_of_flag": float(threshold),
                    "matched_length": matched_length,
                    "severity": severity_from_score(
                        score,
                        DEFAULT_THRESHOLDS,
                    ),
                }
            )

    flags.sort(key=lambda item: item["similarity"], reverse=True)

    return flags


def find_most_similar_chunks(
    chunks_a: List[str],
    chunks_b: List[str],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    top_k: int = 3,
    threshold: float = PLAGIARISM_THRESHOLD,
) -> List[Tuple[str, str, float]]:
    """
    Find the top-K most similar chunk pairs between two documents.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return []

    sim_matrix = cosine_similarity(emb_a, emb_b)

    pairs = []
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1]

    for idx in flat_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        score = float(sim_matrix[i, j])

        if score < threshold:
            break

        if len(pairs) >= top_k:
            break

        pairs.append((chunks_a[i], chunks_b[j], score))

    return pairs


# ── Cross-Lingual Chunk Matching (Issue #1956) ────────────────────────────────


def find_cross_lingual_matches(
    chunks_a: List[str],
    chunks_b: List[str],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    cross_lingual_mode: bool = False,
    top_k: int = 3,
    threshold: float = 0.59,
) -> List[tuple]:
    """Find similar chunks with optional cross-lingual back-translation.

    If cross_lingual_mode is True, chunks from the non-English document
    are back-translated to English before computing similarity.

    Args:
        chunks_a: List of text chunks from document A.
        chunks_b: List of text chunks from document B.
        emb_a: Embedding matrix for document A chunks.
        emb_b: Embedding matrix for document B chunks.
        cross_lingual_mode: Enable cross-lingual back-translation pipeline.
        top_k: Maximum number of matching chunk pairs to return.
        threshold: Minimum similarity score to include a match.

    Returns:
        List of (chunk_a, chunk_b, score) tuples sorted by descending similarity.
    """
    if not cross_lingual_mode:
        # Fallback to standard semantic matching
        return find_most_similar_chunks(
            chunks_a, chunks_b, emb_a, emb_b, top_k, threshold
        )

    # Determine which document needs translation
    lang_a = detect_chunk_language(" ".join(chunks_a[:3])) if chunks_a else "en"  # noqa: F841
    lang_b = detect_chunk_language(" ".join(chunks_b[:3])) if chunks_b else "en"  # noqa: F841

    # For this implementation, we assume emb_a and emb_b are already computed
    # on the back-translated text by the calling pipeline.
    # Here we just perform the standard cosine similarity matching.
    if emb_a.size == 0 or emb_b.size == 0:
        return []

    sim_matrix = cosine_similarity(emb_a, emb_b)
    matches = []

    # Get top K matches above threshold
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1]

    for idx in flat_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        score = float(sim_matrix[i, j])

        if score < threshold:
            break

        if len(matches) >= top_k:
            break

        matches.append((chunks_a[i], chunks_b[j], score))

    return matches


# ── Per-Paragraph Similarity Breakdown ────────────────────────────────────────


def calculate_paragraph_similarity_breakdown(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """
    Compute a per-paragraph similarity breakdown between two documents.

    For each paragraph (chunk) in Document A, finds the single best-matching
    paragraph in Document B using cosine similarity and returns a structured list
    of ``(paragraph_a_idx, paragraph_b_idx, score)`` tuples, sorted by
    descending similarity score.

    Args:
        emb_a: Paragraph embedding matrix for Document A. Shape ``(n_paragraphs, dim)``
               or ``(dim,)`` for a single paragraph.
        emb_b: Paragraph embedding matrix for Document B. Shape ``(m_paragraphs, dim)``
               or ``(dim,)`` for a single paragraph.

    Returns:
        A list of ``(paragraph_a_idx, paragraph_b_idx, score)`` tuples, sorted
        by descending score. Returns an empty list when either input is empty.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return []

    # Ensure 2-D matrices so cosine_similarity works uniformly.
    matrix_a = emb_a.reshape(1, -1) if emb_a.ndim == 1 else emb_a
    matrix_b = emb_b.reshape(1, -1) if emb_b.ndim == 1 else emb_b

    # shape: (n_paragraphs_a, n_paragraphs_b)
    sim_matrix = cosine_similarity(matrix_a, matrix_b)

    breakdown: List[Tuple[int, int, float]] = []
    for idx_a in range(sim_matrix.shape[0]):
        idx_b = int(np.argmax(sim_matrix[idx_a]))
        score = float(np.clip(sim_matrix[idx_a, idx_b], 0.0, 1.0))
        breakdown.append((idx_a, idx_b, score))

    breakdown.sort(key=lambda t: t[2], reverse=True)
    return breakdown


def find_exact_matches(
    text_a: str,
    text_b: str,
    case_sensitive: bool = False,
) -> List[str]:
    """
    Find exact matching sentences/segments from text_a that exist in text_b.

    Args:
        text_a: Source text containing potential matches.
        text_b: Reference text to search within.
        case_sensitive: If True, performs strict case-sensitive matching.

    Returns:
        List of matching segments.
    """
    if not text_a or not text_b:
        return []

    # Lowercase both text buffers before comparison when case_sensitive=False
    if not case_sensitive:
        norm_a = text_a.lower()
        norm_b = text_b.lower()
    else:
        norm_a = text_a
        norm_b = text_b

    import re

    segments = [s.strip() for s in re.split(r"[\n\.]", text_a) if s.strip()]
    segments_norm = [s.strip() for s in re.split(r"[\n\.]", norm_a) if s.strip()]

    matches = []
    for orig, norm in zip(segments, segments_norm):
        if norm in norm_b:
            if orig not in matches:
                matches.append(orig)

    return matches


# ── Cross-Encoder Rescoring Stage (#1355) ──────────────────────────────────────

_CROSS_ENCODER_MODELS: Dict[str, Any] = {}
_CROSS_ENCODER_FAILED_MODELS: Set[str] = set()


def clear_cross_encoder_cache() -> None:
    """Clear the cached CrossEncoder model instances and failure history."""
    global _CROSS_ENCODER_MODELS, _CROSS_ENCODER_FAILED_MODELS
    _CROSS_ENCODER_MODELS.clear()
    _CROSS_ENCODER_FAILED_MODELS.clear()


def get_cross_encoder_info(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    """
    Return diagnostic status information for the specified CrossEncoder model.

    Returns:
        Dict containing model_name, is_loaded, and is_failed status flags.
    """
    global _CROSS_ENCODER_MODELS, _CROSS_ENCODER_FAILED_MODELS
    return {
        "model_name": model_name,
        "is_loaded": model_name in _CROSS_ENCODER_MODELS
        and _CROSS_ENCODER_MODELS[model_name] is not None,
        "is_failed": model_name in _CROSS_ENCODER_FAILED_MODELS,
    }


def _get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Optional[Any]:
    """
    Safely load and cache a SentenceTransformers CrossEncoder model.

    If loading fails (e.g. model not found, offline network, or missing dependency),
    logs a warning and returns None to trigger bi-encoder fallback.

    Args:
        model_name: HuggingFace model path or identifier.

    Returns:
        CrossEncoder model instance or None if load failed.
    """
    global _CROSS_ENCODER_MODELS, _CROSS_ENCODER_FAILED_MODELS

    if model_name in _CROSS_ENCODER_FAILED_MODELS:
        return None

    if model_name in _CROSS_ENCODER_MODELS:
        return _CROSS_ENCODER_MODELS[model_name]

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        _CROSS_ENCODER_MODELS[model_name] = model
        return model
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[similarity] Cross-Encoder model '%s' load failed: %s. "
            "Falling back to initial bi-encoder vector similarity scores.",
            model_name,
            exc,
        )
        _CROSS_ENCODER_FAILED_MODELS.add(model_name)
        return None


def rerank_candidates_with_cross_encoder(
    pairs: list[tuple],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
    batch_size: int = 32,
    apply_sigmoid: bool = True,
) -> list[tuple]:
    """
    Re-scores and re-ranks top candidate text pairs using a joint Cross-Encoder model.

    Bi-encoder vector embeddings (SentenceTransformers) are fast for top-K candidate
    retrieval, but approximate. This Cross-Encoder stage performs full attention
    between both text inputs to refine candidate similarity scores with high precision.

    Args:
        pairs: List of candidate tuples. Each item should be a tuple containing at least
               two string elements: (text_a, text_b, [optional_initial_score, ...]).
        model_name: HuggingFace model name for SentenceTransformers CrossEncoder.
                    Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        top_k: Optional maximum number of candidate pairs to re-score and return.
        batch_size: Batch size passed to CrossEncoder evaluation.
        apply_sigmoid: Whether to apply a logistic sigmoid function to normalize
                       raw logits to [0.0, 1.0].

    Returns:
        List of re-scored candidate tuples sorted in descending order of similarity score.
        If model loading or evaluation fails, falls back gracefully to the original input pairs.
    """
    if not pairs:
        return []

    # Limit candidate pairs to top_k if specified
    candidates = pairs[:top_k] if top_k is not None and top_k > 0 else pairs

    model = _get_cross_encoder(model_name=model_name)
    if model is None:
        # Fall back gracefully to original pairs with bi-encoder scores preserved
        return candidates

    try:
        # Extract text pairs for joint encoding
        sentence_pairs = []
        for p in candidates:
            if not isinstance(p, (tuple, list)) or len(p) < 2:
                continue
            text_a = str(p[0])
            text_b = str(p[1])
            sentence_pairs.append((text_a, text_b))

        if not sentence_pairs:
            return candidates

        raw_scores = model.predict(sentence_pairs, batch_size=batch_size)

        # Normalize raw scores using sigmoid if enabled
        rescored_pairs = []
        for idx, orig_tuple in enumerate(candidates):
            if idx >= len(raw_scores):
                rescored_pairs.append(orig_tuple)
                continue

            raw_s = float(raw_scores[idx])
            if apply_sigmoid:
                # Logistic sigmoid: 1 / (1 + exp(-x))
                score = float(1.0 / (1.0 + np.exp(-raw_s)))
            else:
                score = float(np.clip(raw_s, 0.0, 1.0))

            score = round(float(np.clip(score, 0.0, 1.0)), 4)

            # Build updated tuple preserving extra metadata if present
            if len(orig_tuple) > 2:
                updated_tuple = (orig_tuple[0], orig_tuple[1], score) + orig_tuple[3:]
            else:
                updated_tuple = (orig_tuple[0], orig_tuple[1], score)

            rescored_pairs.append(updated_tuple)

        # Sort candidate pairs descending by cross-encoder score
        rescored_pairs.sort(
            key=lambda item: (
                item[2] if len(item) > 2 and isinstance(item[2], (int, float)) else 0.0
            ),
            reverse=True,
        )
        return rescored_pairs

    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[similarity] Cross-Encoder rescoring execution failed: %s. "
            "Falling back to bi-encoder vector scores.",
            exc,
        )
        return candidates


# ─── Plagiarism Cluster Detection (Issue #1675) ──────────────────────────────


def detect_plagiarism_clusters(
    similarity_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD,
) -> dict:
    """Detect groups (clusters) of highly related documents using connected components.

    Instead of only showing document pairs, this function builds a similarity graph
    where edges exist between documents exceeding the threshold, then identifies
    connected components to find groups of students who may be colluding or
    sharing source material.

    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold: Minimum similarity score to create an edge in the graph.

    Returns:
        Dictionary containing:
        - 'clusters': Dict mapping cluster_id (int) to list of document names.
        - 'cluster_map': Dict mapping document name to its cluster_id.
        - 'suspicious_groups': List of clusters with 3+ documents (potential collusion rings).
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning(
            "networkx is not installed. Install it with: pip install networkx>=3.0"
        )
        return {
            "clusters": {},
            "cluster_map": {},
            "suspicious_groups": [],
            "total_clusters": 0,
            "error": "networkx not installed",
            "message": "Please install networkx: pip install networkx>=3.0",
        }

    doc_names = list(similarity_df.columns)
    G = nx.Graph()

    # Add all documents as nodes
    G.add_nodes_from(doc_names)

    # Add edges for pairs exceeding threshold
    n = len(doc_names)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_df.iloc[i, j])
            if score >= threshold:
                G.add_edge(doc_names[i], doc_names[j], weight=score)

    # Detect connected components (clusters)
    clusters = {}
    cluster_map = {}

    for cluster_id, component in enumerate(nx.connected_components(G)):
        cluster_list = sorted(list(component))
        clusters[cluster_id] = cluster_list
        for doc in cluster_list:
            cluster_map[doc] = cluster_id

    # Identify suspicious groups (3+ documents in a cluster)
    suspicious_groups = [
        {
            "cluster_id": cid,
            "documents": docs,
            "size": len(docs),
        }
        for cid, docs in clusters.items()
        if len(docs) >= 3
    ]

    # Sort suspicious groups by size descending
    suspicious_groups.sort(key=lambda x: x["size"], reverse=True)

    logger.info(
        "Detected %d plagiarism clusters, %d suspicious groups (3+ docs).",
        len(clusters),
        len(suspicious_groups),
    )

    return {
        "clusters": clusters,
        "cluster_map": cluster_map,
        "suspicious_groups": suspicious_groups,
        "total_clusters": len(clusters),
    }


# ============================================================================
# HYBRID SIMILARITY INTEGRATION - Issue #2676
# ============================================================================

from src.core.hybrid_scorer import HybridConfig, HybridScorer


def compute_hybrid_similarity_df(
    semantic_df: pd.DataFrame,
    texts: Dict[str, str],
    alpha: float = 0.7,
    lexical_method: str = "tfidf",
) -> pd.DataFrame:
    """
    Compute hybrid similarity DataFrame combining semantic and lexical scores.

    Args:
        semantic_df: Semantic similarity DataFrame
        texts: Document texts
        alpha: Semantic weight (0.7 = 70% semantic, 30% lexical)
        lexical_method: Lexical method to use

    Returns:
        Hybrid similarity DataFrame
    """
    scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=lexical_method))
    return scorer.compute_hybrid_matrix(texts, semantic_df, alpha, lexical_method)


def flag_plagiarism_hybrid(
    hybrid_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD,
) -> List[Dict]:
    """
    Flag plagiarism using hybrid similarity scores.

    Args:
        hybrid_df: Hybrid similarity DataFrame
        threshold: Flagging threshold

    Returns:
        List of flagged pairs
    """
    scorer = HybridScorer()
    return scorer.flag_plagiarism_hybrid(hybrid_df, threshold)
