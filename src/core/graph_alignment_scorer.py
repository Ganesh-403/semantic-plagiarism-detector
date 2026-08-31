"""
src/core/graph_alignment_scorer.py
----------------------------------
Graph Alignment Scorer for Conceptual Plagiarism.

Computes subgraph isomorphism and conceptual overlap scores between
extracted knowledge graphs to detect idea theft.
"""

import logging
from typing import List, Dict, Any, Tuple, Set

logger = logging.getLogger(__name__)


def compute_graph_jaccard_similarity(
    graph_a: Dict[str, Any], graph_b: Dict[str, Any]
) -> float:
    """Compute Jaccard similarity between the edge sets of two graphs.

    Args:
        graph_a: First knowledge graph.
        graph_b: Second knowledge graph.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    edges_a = set(tuple(e) for e in graph_a.get("edges", []))
    edges_b = set(tuple(e) for e in graph_b.get("edges", []))

    if not edges_a and not edges_b:
        return 1.0

    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))

    return intersection / union if union > 0 else 0.0


def compute_conceptual_overlap(
    graph_a: Dict[str, Any], graph_b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute detailed conceptual overlap metrics between two graphs.

    Args:
        graph_a: First knowledge graph.
        graph_b: Second knowledge graph.

    Returns:
        Dictionary containing overlap metrics and conceptual plagiarism flags.
    """
    nodes_a = set(graph_a.get("nodes", []))
    nodes_b = set(graph_b.get("nodes", []))

    edges_a = set(tuple(e) for e in graph_a.get("edges", []))
    edges_b = set(tuple(e) for e in graph_b.get("edges", []))

    node_intersection = len(nodes_a.intersection(nodes_b))
    node_union = len(nodes_a.union(nodes_b))
    node_jaccard = node_intersection / node_union if node_union > 0 else 0.0

    edge_intersection = len(edges_a.intersection(edges_b))
    edge_union = len(edges_a.union(edges_b))
    edge_jaccard = edge_intersection / edge_union if edge_union > 0 else 0.0

    # Conceptual plagiarism is flagged if edge overlap is high, even if node vocabulary differs slightly
    # (since students might use synonyms for nodes but keep the same relational structure)
    conceptual_score = (node_jaccard * 0.4) + (edge_jaccard * 0.6)
    is_conceptual_plagiarism = edge_jaccard > 0.40

    return {
        "node_jaccard": round(node_jaccard, 4),
        "edge_jaccard": round(edge_jaccard, 4),
        "conceptual_score": round(conceptual_score, 4),
        "is_conceptual_plagiarism": is_conceptual_plagiarism,
        "shared_edges": edge_intersection,
    }
