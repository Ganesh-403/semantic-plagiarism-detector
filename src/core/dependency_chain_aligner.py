"""
src/core/dependency_chain_aligner.py
-------------------------------------
Dependency Chain Alignment Engine.

Computes graph-edit distance and sequence alignment between API call
graphs to detect algorithmic cloning based on external resource utilization.
"""

import logging
from typing import List, Dict, Any
from src.core.api_call_graph_extractor import APICallGraph

logger = logging.getLogger(__name__)


def compute_api_graph_similarity(
    graph_a: APICallGraph, graph_b: APICallGraph
) -> Dict[str, Any]:
    """Compute similarity between two API call graphs.

    Combines node set Jaccard similarity with sequence alignment of the
    call chains to determine if the same external dependencies and call
    patterns are being utilized.

    Args:
        graph_a: API call graph from document A.
        graph_b: API call graph from document B.

    Returns:
        Dictionary containing structural and sequence similarity scores.
    """
    if not graph_a.nodes and not graph_b.nodes:
        return {
            "node_similarity": 1.0,
            "sequence_similarity": 1.0,
            "overall_score": 1.0,
            "is_clone": False,
        }
    if not graph_a.nodes or not graph_b.nodes:
        return {
            "node_similarity": 0.0,
            "sequence_similarity": 0.0,
            "overall_score": 0.0,
            "is_clone": False,
        }

    # 1. Node Set Jaccard Similarity
    nodes_a = set(graph_a.nodes.keys())
    nodes_b = set(graph_b.nodes.keys())

    intersection = len(nodes_a.intersection(nodes_b))
    union = len(nodes_a.union(nodes_b))
    node_sim = intersection / union if union > 0 else 0.0

    # 2. Sequence Alignment (Levenshtein distance on call sequences)
    seq_a = graph_a.call_sequence
    seq_b = graph_b.call_sequence

    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        seq_sim = 1.0
    elif n == 0 or m == 0:
        seq_sim = 0.0
    else:
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
                )

        edit_distance = dp[n][m]
        max_len = max(n, m)
        seq_sim = 1.0 - (edit_distance / max_len)

    # Overall score weights sequence alignment higher, as call order matters
    overall_score = (node_sim * 0.4) + (seq_sim * 0.6)

    # Flag as clone if both node usage and call sequence are highly similar
    is_clone = node_sim > 0.60 and seq_sim > 0.70

    return {
        "node_similarity": round(node_sim, 4),
        "sequence_similarity": round(seq_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_clone": is_clone,
    }
