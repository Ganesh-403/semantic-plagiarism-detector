"""
src/core/code_similarity_engine.py
----------------------------------
Code Similarity Engine using Abstract Syntax Trees (AST).

Computes structural similarity scores between normalized ASTs using
sequence alignment and node frequency analysis. This detects structural
plagiarism even when variables are renamed or whitespace is altered.
"""

import logging
from typing import List, Dict, Any
from collections import Counter
import math

from src.core.code_ast_parser import parse_and_normalize_code, ast_to_node_sequence

logger = logging.getLogger(__name__)


def compute_sequence_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute the Jaccard similarity between two node sequences.

    This is a fast approximation of structural similarity based on the
    frequency of AST node types (e.g., If, For, While, Assign).

    Args:
        seq_a: Node sequence from the first program.
        seq_b: Node sequence from the second program.

    Returns:
        A similarity score between 0.0 and 1.0.
    """
    if not seq_a and not seq_b:
        return 1.0

    counter_a = Counter(seq_a)
    counter_b = Counter(seq_b)

    # Compute intersection and union of node type frequencies
    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    if union == 0:
        return 0.0

    return intersection / union


def compute_levenshtein_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute the normalized Levenshtein similarity between two node sequences.

    This provides a more accurate measure of structural similarity by
    computing the edit distance between the sequences of AST node types.
    Note: This is O(N*M) and should be used for smaller code snippets.

    Args:
        seq_a: Node sequence from the first program.
        seq_b: Node sequence from the second program.

    Returns:
        A similarity score between 0.0 and 1.0.
    """
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    # Initialize DP matrix for Levenshtein distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    distance = dp[n][m]
    max_len = max(n, m)

    # Normalize to similarity score
    return 1.0 - (distance / max_len)


def compare_code_snippets(code_a: str, code_b: str) -> dict[str, float]:
    """Compare two Python code snippets for structural plagiarism.

    Args:
        code_a: The first source code string.
        code_b: The second source code string.

    Returns:
        A dictionary containing similarity scores:
        - 'jaccard_similarity': Fast frequency-based similarity.
        - 'levenshtein_similarity': Accurate edit-distance similarity.
        - 'overall_score': Weighted average of both scores.
    """
    tree_a = parse_and_normalize_code(code_a)
    tree_b = parse_and_normalize_code(code_b)

    if tree_a is None or tree_b is None:
        logger.warning("Failed to parse one or both code snippets for comparison.")
        return {
            "jaccard_similarity": 0.0,
            "levenshtein_similarity": 0.0,
            "overall_score": 0.0,
        }

    seq_a = ast_to_node_sequence(tree_a)
    seq_b = ast_to_node_sequence(tree_b)

    jaccard = compute_sequence_similarity(seq_a, seq_b)

    # Only compute Levenshtein for reasonably sized snippets to avoid OOM/hang
    if len(seq_a) < 1000 and len(seq_b) < 1000:
        levenshtein = compute_levenshtein_similarity(seq_a, seq_b)
    else:
        levenshtein = jaccard  # Fallback for large files

    # Weighted overall score (Levenshtein is more accurate but slower)
    overall = (0.4 * jaccard) + (0.6 * levenshtein)

    return {
        "jaccard_similarity": round(jaccard, 4),
        "levenshtein_similarity": round(levenshtein, 4),
        "overall_score": round(overall, 4),
    }
