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
src/core/graph_isomorphism_engine.py
------------------------------------
Graph Isomorphism and Edit Distance Engine for CFG Comparison.

Computes graph-edit distance and subgraph isomorphism scores between
Control Flow Graphs to detect algorithmic cloning and logic obfuscation.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def compute_graph_edit_distance(
    adj_list_a: Dict[int, List[int]], adj_list_b: Dict[int, List[int]]
) -> int:
    """Compute an approximation of the graph-edit distance between two CFGs.

    This is a simplified heuristic based on the difference in node counts,
    edge counts, and degree distributions. True graph-edit distance is
    NP-hard, so we use a structural similarity proxy.

    Args:
        adj_list_a: Adjacency list for graph A.
        adj_list_b: Adjacency list for graph B.

    Returns:
        Integer representing the approximate edit distance.
    """
    nodes_a = set(adj_list_a.keys())
    nodes_b = set(adj_list_b.keys())

    edges_a = sum(len(successors) for successors in adj_list_a.values())
    edges_b = sum(len(successors) for successors in adj_list_b.values())

    # Node difference
    node_diff = abs(len(nodes_a) - len(nodes_b))

    # Edge difference
    edge_diff = abs(edges_a - edges_b)

    # Degree distribution difference
    degrees_a = sorted([len(successors) for successors in adj_list_a.values()])
    degrees_b = sorted([len(successors) for successors in adj_list_b.values()])

    # Pad shorter list with zeros for comparison
    max_len = max(len(degrees_a), len(degrees_b))
    degrees_a += [0] * (max_len - len(degrees_a))
    degrees_b += [0] * (max_len - len(degrees_b))

    degree_diff = sum(abs(a - b) for a, b in zip(degrees_a, degrees_b))

    # Weighted sum of differences
    distance = node_diff + edge_diff + (degree_diff // 2)
    return distance


def compute_structural_similarity(
    adj_list_a: Dict[int, List[int]], adj_list_b: Dict[int, List[int]]
) -> float:
    """Compute a normalized structural similarity score between two CFGs.

    Uses the Jaccard similarity of edge sets (normalized by node pairs)
    as a proxy for structural isomorphism.

    Args:
        adj_list_a: Adjacency list for graph A.
        adj_list_b: Adjacency list for graph B.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not adj_list_a and not adj_list_b:
        return 1.0
    if not adj_list_a or not adj_list_b:
        return 0.0

    # Extract edge sets (as tuples of sorted node IDs to ignore directionality)
    edges_a = set()
    for node, successors in adj_list_a.items():
        for succ in successors:
            edge = tuple(sorted([node, succ]))
            edges_a.add(edge)

    edges_b = set()
    for node, successors in adj_list_b.items():
        for succ in successors:
            edge = tuple(sorted([node, succ]))
            edges_b.add(edge)

    if not edges_a and not edges_b:
        # Both graphs have nodes but no edges (e.g., linear sequence)
        # Compare node counts
        node_diff = abs(len(adj_list_a) - len(adj_list_b))
        max_nodes = max(len(adj_list_a), len(adj_list_b), 1)
        return 1.0 - (node_diff / max_nodes)

    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))

    return intersection / union if union > 0 else 0.0


def compare_cfgs(blocks_a: Dict[int, Any], blocks_b: Dict[int, Any]) -> Dict[str, Any]:
    """Compare two Control Flow Graphs for algorithmic cloning.

    Args:
        blocks_a: Dictionary of BasicBlock objects for graph A.
        blocks_b: Dictionary of BasicBlock objects for graph B.

    Returns:
        Dictionary containing edit distance, similarity score, and match flags.
    """
    # Convert to adjacency lists
    from src.core.cfg_generator import cfg_to_adjacency_list, compute_cfg_hash

    adj_a = cfg_to_adjacency_list(blocks_a)
    adj_b = cfg_to_adjacency_list(blocks_b)

    edit_distance = compute_graph_edit_distance(adj_a, adj_b)
    similarity = compute_structural_similarity(adj_a, adj_b)

    hash_a = compute_cfg_hash(blocks_a)
    hash_b = compute_cfg_hash(blocks_b)

    is_exact_clone = (hash_a == hash_b) and (hash_a != "")

    return {
        "edit_distance": edit_distance,
        "structural_similarity": round(similarity, 4),
        "hash_a": hash_a,
        "hash_b": hash_b,
        "is_exact_clone": is_exact_clone,
        "node_count_a": len(adj_a),
        "node_count_b": len(adj_b),
    }
