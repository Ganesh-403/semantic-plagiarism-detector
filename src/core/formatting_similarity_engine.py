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
src/core/formatting_similarity_engine.py
----------------------------------------
Formatting Similarity Engine for Layout Trees.

Computes tree-edit distance and structural similarity scores between
layout trees to detect structural cloning and formatting plagiarism.
"""

import logging
from typing import Any, Dict, List, Tuple

from src.core.layout_tree_extractor import LayoutNode

logger = logging.getLogger(__name__)


def tree_to_sequence(node: LayoutNode) -> List[str]:
    """Flatten a layout tree into a sequence of tags using pre-order traversal."""
    seq = [node.tag]
    for child in node.children:
        seq.extend(tree_to_sequence(child))
    return seq


def compute_tree_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the Levenshtein distance between two layout tag sequences.

    This serves as a proxy for tree-edit distance on the flattened structure.
    """
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


def compute_layout_similarity(tree_a: LayoutNode, tree_b: LayoutNode) -> Dict[str, Any]:
    """Compute structural similarity between two layout trees.

    Args:
        tree_a: First layout tree.
        tree_b: Second layout tree.

    Returns:
        Dictionary containing edit distance and similarity score.
    """
    seq_a = tree_to_sequence(tree_a)
    seq_b = tree_to_sequence(tree_b)

    distance = compute_tree_edit_distance(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b), 1)

    similarity = 1.0 - (distance / max_len)

    return {
        "edit_distance": distance,
        "structural_similarity": round(similarity, 4),
        "node_count_a": len(seq_a),
        "node_count_b": len(seq_b),
        "is_structural_clone": similarity > 0.85,  # Threshold for structural cloning
    }
