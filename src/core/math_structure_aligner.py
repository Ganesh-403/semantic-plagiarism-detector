"""
src/core/math_structure_aligner.py
----------------------------------
Mathematical Structure Alignment Engine.

Computes tree-edit distance and structural similarity between equation ASTs
to detect structural mathematical plagiarism.
"""

import logging
from typing import List, Dict, Any
from src.core.equation_ast_extractor import MathNode

logger = logging.getLogger(__name__)


def extract_preorder_sequence(node: MathNode) -> List[str]:
    """Extract a pre-order traversal sequence of the AST."""
    seq = [f"{node.node_type}:{node.value}"]
    for child in node.children:
        seq.extend(extract_preorder_sequence(child))
    return seq


def compute_tree_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the Levenshtein distance between two AST pre-order sequences."""
    n = len(seq_a)
    m = len(seq_b)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def compute_math_similarity(tree_a: MathNode, tree_b: MathNode) -> Dict[str, Any]:
    """Compute structural similarity between two mathematical ASTs.

    Args:
        tree_a: AST from equation A.
        tree_b: AST from equation B.

    Returns:
        Dictionary containing edit distance, similarity score, and plagiarism flags.
    """
    seq_a = extract_preorder_sequence(tree_a)
    seq_b = extract_preorder_sequence(tree_b)

    distance = compute_tree_edit_distance(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b), 1)

    similarity = 1.0 - (distance / max_len)
    is_structural_plagiarism = similarity > 0.85

    return {
        "edit_distance": distance,
        "structural_similarity": round(similarity, 4),
        "is_structural_plagiarism": is_structural_plagiarism,
        "node_count_a": len(seq_a),
        "node_count_b": len(seq_b),
    }
