# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/core/semantic_alignment.py
------------------------------
Semantic-aware sequence alignment for detecting paraphrased plagiarism.

Standard lexical diff tools (like Python's difflib) fail when students
use synonym swapping, sentence restructuring, or structural reordering.
This module implements a Dynamic Programming (DP) alignment algorithm
(similar to Needleman-Wunsch) that operates on sentence embedding vectors
instead of exact character matches.

Recent Additions (Issue #2001):
- Added memory allocation guard to prevent OOM errors on massive documents.
- The current implementation allocates a full (N+1) × (M+1) DP matrix.
  While banding restricts computation to a diagonal window, the underlying
  NumPy array still reserves the full N×M memory footprint.
- Raises ValueError if N > 1000 or M > 1000 to prevent regressions.
"""

import logging
from typing import Any, Dict, List, Optional  # noqa: F401

import numpy as np

logger = logging.getLogger(__name__)

# Default alignment parameters
DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_GAP_PENALTY = -0.25
DEFAULT_BAND_WIDTH = 20  # Sakoe-Chiba band width for O(N*W) complexity

# Memory allocation limits (Issue #2001)
# A 1000x1000 float32 matrix is ~4 MB. 5000x5000 is ~100 MB.
# We cap at 1000 to prevent accidental OOM on large document comparisons
# until a true sparse/banded memory allocation is implemented.
MAX_SEQUENCE_LENGTH = 1000


def _cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine similarity matrix between two embedding arrays.

    Args:
        emb_a: Array of shape (N, D).
        emb_b: Array of shape (M, D).

    Returns:
        Array of shape (N, M) containing cosine similarities.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return np.array([[]])

    # Normalize embeddings to unit length for dot product = cosine similarity
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)

    # Prevent division by zero
    norm_a = np.where(norm_a == 0, 1.0, norm_a)
    norm_b = np.where(norm_b == 0, 1.0, norm_b)

    emb_a_norm = emb_a / norm_a
    emb_b_norm = emb_b / norm_b

    return np.dot(emb_a_norm, emb_b_norm.T)


def align_semantic_sequences(
    chunks_a: list[str],
    chunks_b: list[str],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
    band_width: int = DEFAULT_BAND_WIDTH,
) -> list[dict[str, Any]]:
    """Align two sequences of text chunks based on semantic similarity.

    Uses a banded Needleman-Wunsch dynamic programming algorithm to find
    the optimal global alignment between Document A and Document B. The
    banding restricts the DP matrix computation to a diagonal window,
    reducing time complexity from O(N*M) to O(N*W).

    Memory Trade-off Note (Issue #2001):
        Although the computation is banded, the current implementation
        allocates a full (N+1) × (M+1) NumPy array for simplicity and
        traceback ease. For N, M <= 1000, this consumes ~4 MB of RAM,
        which is acceptable. For larger inputs, this would cause excessive
        memory allocation. A guard is enforced at the top of this function
        to raise ValueError if N > 1000 or M > 1000. Future optimizations
        should implement a true sparse banded matrix allocation.

    Args:
        chunks_a: List of text strings from Document A.
        chunks_b: List of text strings from Document B.
        embeddings_a: Numpy array of shape (N, D) for Document A.
        embeddings_b: Numpy array of shape (M, D) for Document B.
        match_threshold: Minimum cosine similarity to consider a "match".
        gap_penalty: Penalty score for inserting a gap (unmatched chunk).
        band_width: Maximum allowed deviation from the diagonal.

    Returns:
        A list of alignment operations (dictionaries) representing the
        optimal path. Each dict contains:
        - 'type': 'match', 'paraphrase', 'insert_a', or 'insert_b'
        - 'text_a': The chunk from Document A (or None if gap)
        - 'text_b': The chunk from Document B (or None if gap)
        - 'score': The cosine similarity score (0.0 for gaps)

    Raises:
        ValueError: If len(chunks_a) > 1000 or len(chunks_b) > 1000.
    """
    n = len(chunks_a)
    m = len(chunks_b)

    # ── Memory Allocation Guard (Issue #2001) ─────────────────────────────────
    if n > MAX_SEQUENCE_LENGTH or m > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence alignment matrix size limit exceeded. "
            f"Got N={n}, M={m}. Maximum allowed is {MAX_SEQUENCE_LENGTH}x{MAX_SEQUENCE_LENGTH} "
            f"to prevent excessive memory allocation. "
            f"The current implementation allocates a full (N+1)×(M+1) DP matrix. "
            f"Please use smaller chunk sizes or split the documents."
        )

    if n == 0 and m == 0:
        return []
    if n == 0:
        return [
            {"type": "insert_b", "text_a": None, "text_b": c, "score": 0.0}
            for c in chunks_b
        ]
    if m == 0:
        return [
            {"type": "insert_a", "text_a": c, "text_b": None, "score": 0.0}
            for c in chunks_a
        ]

    # Compute similarity matrix
    sim_matrix = _cosine_similarity_matrix(embeddings_a, embeddings_b)

    # Initialize DP matrix with negative infinity
    # Note: This allocates the full (n+1) x (m+1) matrix.
    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0, 0] = 0.0

    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0, j] = j * gap_penalty

    # Fill DP matrix with Sakoe-Chiba band constraint
    for i in range(1, n + 1):
        # Calculate band boundaries for current row
        j_start = max(1, i - band_width)
        j_end = min(m, i + band_width)

        for j in range(j_start, j_end + 1):
            sim_score = float(sim_matrix[i - 1, j - 1])

            # Determine match/paraphrase score
            # If similarity is above threshold, it's a positive match score
            # Otherwise, it's treated as a mismatch (score 0 or slight penalty)
            match_score = sim_score if sim_score >= match_threshold else 0.0

            # Three possible moves:
            # 1. Diagonal (Match/Mismatch): align chunk i with chunk j
            diag = dp[i - 1, j - 1] + match_score

            # 2. Up (Gap in B / Deletion from A): chunk i is unmatched
            up = dp[i - 1, j] + gap_penalty

            # 3. Left (Gap in A / Insertion to B): chunk j is unmatched
            left = dp[i, j - 1] + gap_penalty

            dp[i, j] = max(diag, up, left)

    # Traceback to find the optimal alignment path
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim_score = float(sim_matrix[i - 1, j - 1])
            match_score = sim_score if sim_score >= match_threshold else 0.0

            # Check if diagonal move was chosen
            if abs(dp[i, j] - (dp[i - 1, j - 1] + match_score)) < 1e-6:
                op_type = "match" if sim_score >= match_threshold else "paraphrase"
                alignment.append(
                    {
                        "type": op_type,
                        "text_a": chunks_a[i - 1],
                        "text_b": chunks_b[j - 1],
                        "score": sim_score,
                    }
                )
                i -= 1
                j -= 1
                continue

        if i > 0 and abs(dp[i, j] - (dp[i - 1, j] + gap_penalty)) < 1e-6:
            # Up move: chunk from A is unmatched (deleted)
            alignment.append(
                {
                    "type": "insert_a",
                    "text_a": chunks_a[i - 1],
                    "text_b": None,
                    "score": 0.0,
                }
            )
            i -= 1
        else:
            # Left move: chunk from B is unmatched (inserted)
            alignment.append(
                {
                    "type": "insert_b",
                    "text_a": None,
                    "text_b": chunks_b[j - 1],
                    "score": 0.0,
                }
            )
            j -= 1

    # Reverse to get chronological order
    alignment.reverse()

    logger.info(
        "Semantic alignment complete: %d operations for %d x %d chunks.",
        len(alignment),
        n,
        m,
    )

    return alignment
