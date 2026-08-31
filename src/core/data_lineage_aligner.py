"""
src/core/data_lineage_aligner.py
--------------------------------
Data Lineage Alignment Engine.

Computes Directed Acyclic Graph (DAG) edit distance and lineage similarity
between notebook workflows to detect cloned computational logic.
"""

import logging
from typing import List, Dict, Any
from src.core.notebook_graph_extractor import NotebookGraph

logger = logging.getLogger(__name__)


def compute_execution_sequence_distance(seq_a: List[int], seq_b: List[int]) -> int:
    """Compute the Levenshtein distance between two execution count sequences."""
    n, m = len(seq_a), len(seq_b)
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


def compute_lineage_similarity(
    graph_a: NotebookGraph, graph_b: NotebookGraph
) -> Dict[str, Any]:
    """Compute structural and lineage similarity between two notebook graphs.

    Args:
        graph_a: NotebookGraph from notebook A.
        graph_b: NotebookGraph from notebook B.

    Returns:
        Dictionary containing execution distance, lineage similarity, and plagiarism flags.
    """
    if not graph_a.cells and not graph_b.cells:
        return {
            "execution_distance": 0,
            "lineage_similarity": 1.0,
            "is_cloned_workflow": False,
        }

    # 1. Execution Sequence Distance
    exec_dist = compute_execution_sequence_distance(
        graph_a.execution_sequence, graph_b.execution_sequence
    )
    max_exec_len = max(
        len(graph_a.execution_sequence), len(graph_b.execution_sequence), 1
    )
    exec_sim = 1.0 - (exec_dist / max_exec_len)

    # 2. Lineage Edge Similarity (Jaccard)
    # Normalize edges by ignoring specific cell IDs and just looking at the
    # relative dependency structure (e.g., edge index offsets)
    # For simplicity here, we compare the raw edge counts and node degrees
    edges_a = set(graph_a.lineage_edges)
    edges_b = set(graph_b.lineage_edges)

    # Since cell IDs differ, we compare the total number of dependencies
    # relative to the number of code cells.
    code_cells_a = len([c for c in graph_a.cells if c.cell_type == "code"])
    code_cells_b = len([c for c in graph_b.cells if c.cell_type == "code"])

    density_a = len(edges_a) / max(code_cells_a, 1)
    density_b = len(edges_b) / max(code_cells_b, 1)

    density_diff = abs(density_a - density_b)
    lineage_sim = max(0.0, 1.0 - (density_diff / max(density_a, density_b, 1.0)))

    # Overall score
    overall_score = (exec_sim * 0.5) + (lineage_sim * 0.5)

    # Flag as cloned if both execution flow and dependency density match closely
    is_cloned = overall_score > 0.85

    return {
        "execution_distance": exec_dist,
        "execution_similarity": round(exec_sim, 4),
        "lineage_similarity": round(lineage_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_cloned_workflow": is_cloned,
    }
