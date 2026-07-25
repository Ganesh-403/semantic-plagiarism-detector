"""
src/core/similarity.py
----------------------
Computes semantic similarity between documents at two levels:
  1. Document-level  – single score per pair (mean-pooled embeddings)
  2. Chunk-level     – max-similarity per chunk pair (detects local plagiarism)

Uses cosine similarity. Since embeddings are L2-normalised in embedding_model.py,
cosine similarity reduces to the dot product, making this very fast.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import (
    DEFAULT_THRESHOLDS,
    PLAGIARISM_THRESHOLD,
    is_plagiarism,
    severity_from_score,
)

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


# ── Document-level similarity ──────────────────────────────────────────────────


def document_similarity_matrix(
    doc_embeddings: Union[Dict[str, np.ndarray], np.ndarray, List[np.ndarray]],
    batch_size: Optional[int] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Build an N×N cosine similarity matrix between all document pairs.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array, or direct array/list of embeddings.
        batch_size: Optional number of documents to compare per batch.

    Returns:
        Symmetric pandas DataFrame or numpy ndarray with similarity values.
    """
    if isinstance(doc_embeddings, (np.ndarray, list)):
        stacked = np.array(doc_embeddings)
        if stacked.ndim == 1 or stacked.size == 0:
            return np.array([[]])
        sim = cosine_similarity(stacked)
        return np.clip(sim, 0.0, 1.0)

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
    if doc_vectors:
        stacked = np.vstack(doc_vectors)
        safe_batch_size = _validated_batch_size(batch_size)
        if safe_batch_size is None:
            sim = cosine_similarity(stacked)
            matrix = np.clip(sim, 0.0, 1.0)
        else:
            for start in range(0, n, safe_batch_size):
                end = min(start + safe_batch_size, n)
                sim = cosine_similarity(stacked[start:end], stacked)
                matrix[start:end] = np.clip(sim, 0.0, 1.0)

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return df


def compute_similarity_matrix(
    embeddings: Union[Dict[str, np.ndarray], np.ndarray, List[np.ndarray]],
    batch_size: Optional[int] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Direct alias/wrapper for document_similarity_matrix to maintain backwards compatibility
    with app/streamlit_app.py and external modules.
    """
    return document_similarity_matrix(embeddings, batch_size=batch_size)


# ── Hybrid similarity (lexical + semantic) ─────────────────────────────────────


def hybrid_similarity_matrix(
    semantic_df: pd.DataFrame, lexical_df: pd.DataFrame, w: float = 0.7
) -> pd.DataFrame:
    """
    Combine semantic and lexical similarity matrices using a weighted formula.
    """
    if not (0.0 <= w <= 1.0):
        from src.errors import sim_weight_out_of_range

        raise ValueError(sim_weight_out_of_range(w=w))

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


# ── Plagiarism flagging ────────────────────────────────────────────────────────


def flag_plagiarism(
    similarity_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD,
    chunked_docs: dict = None,
    embeddings: dict = None,
) -> List[Dict]:
    """Identify document pairs whose similarity reaches the threshold.

    Flagging uses the configurable plagiarism threshold. Severity uses the
    central fixed boundaries: Medium at 0.75 and High at 0.90.

    """
    flags = []
    doc_names = similarity_df.columns.tolist()

    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            score = float(similarity_df.iloc[i, j])

            if is_plagiarism(score, threshold):

                doc_a = doc_names[i]
                doc_b = doc_names[j]
                matched_length = 0

                # Find the exact matching chunk and its word count
                if chunked_docs is not None and embeddings is not None:
                    sim_matrix = cosine_similarity(embeddings[doc_a], embeddings[doc_b])
                    idx_a, idx_b = np.unravel_index(
                        np.argmax(sim_matrix), sim_matrix.shape
                    )
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

    flags.sort(key=lambda x: x["similarity"], reverse=True)

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
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            score = sim_matrix[i, j]
            if score >= threshold:
                pairs.append((chunks_a[i], chunks_b[j], float(score)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
