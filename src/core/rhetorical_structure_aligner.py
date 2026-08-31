"""
src/core/rhetorical_structure_aligner.py
----------------------------------------
Rhetorical Structure Alignment Engine.

Computes tree-edit distance and structural alignment between discourse
trees to detect structural idea theft.
"""

import logging
from typing import List, Dict, Any
from src.core.discourse_tree_parser import DiscourseNode

logger = logging.getLogger(__name__)


def extract_rhetorical_sequence(node: DiscourseNode) -> List[str]:
    """Extract a pre-order traversal sequence of rhetorical node types."""
    seq = [node.node_type]
    for child in node.children:
        seq.extend(extract_rhetorical_sequence(child))
    return seq


def compute_tree_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the Levenshtein distance between two rhetorical sequences."""
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


def compute_rhetorical_alignment(
    tree_a: DiscourseNode, tree_b: DiscourseNode
) -> Dict[str, Any]:
    """Compute structural alignment between two discourse trees.

    Args:
        tree_a: Discourse tree from document A.
        tree_b: Discourse tree from document B.

    Returns:
        Dictionary containing structural similarity and plagiarism flags.
    """
    seq_a = extract_rhetorical_sequence(tree_a)
    seq_b = extract_rhetorical_sequence(tree_b)

    distance = compute_tree_edit_distance(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b), 1)

    similarity = 1.0 - (distance / max_len)

    # Flag as structural plagiarism if the rhetorical flow is highly preserved
    is_structural_plagiarism = similarity > 0.80

    return {
        "edit_distance": distance,
        "structural_similarity": round(similarity, 4),
        "is_structural_plagiarism": is_structural_plagiarism,
        "node_count_a": len(seq_a),
        "node_count_b": len(seq_b),
    }
