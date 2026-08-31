"""
src/core/covert_collaboration_analyzer.py
-----------------------------------------
Covert Collaboration and Shared Repository Analyzer.

Computes commit graph similarity, timestamp clustering, and authorship
anomaly scores to detect shared hidden repositories.
"""

import logging
from typing import List, Dict, Any
from src.core.git_graph_extractor import GitGraph, compute_timezone_entropy

logger = logging.getLogger(__name__)


def compute_timestamp_clustering(graph: GitGraph) -> float:
    """Compute the variance of inter-commit timestamps.

    Covert collaboration often results in bursty commit patterns (high variance)
    when students push their squashed or rewritten histories at the same time.
    """
    if len(graph.commits) < 2:
        return 0.0

    timestamps = sorted([c.timestamp for c in graph.commits])
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    if not deltas:
        return 0.0

    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)

    # Normalize variance (higher variance = more bursty = more suspicious)
    return round(min(1.0, variance / 1e8), 4)


def analyze_covert_collaboration(
    graph_a: GitGraph, graph_b: GitGraph
) -> Dict[str, Any]:
    """Analyze two Git logs for signs of covert collaboration."""
    if not graph_a.commits or not graph_b.commits:
        return {
            "graph_similarity": 0.0,
            "timestamp_burstiness": 0.0,
            "overall_score": 0.0,
            "is_covert_collaboration": False,
        }

    # 1. Graph Structure Similarity (Edge count and depth)
    edges_a = len(graph_a.edges)
    edges_b = len(graph_b.edges)
    graph_sim = min(edges_a, edges_b) / max(edges_a, edges_b, 1)

    # 2. Timestamp Burstiness
    burst_a = compute_timestamp_clustering(graph_a)
    burst_b = compute_timestamp_clustering(graph_b)
    avg_burst = (burst_a + burst_b) / 2.0

    # 3. Timezone and Author Entropy
    metrics_a = compute_timezone_entropy(graph_a)
    metrics_b = compute_timezone_entropy(graph_b)

    # High author entropy + high timezone entropy in a student repo is suspicious
    entropy_score = (metrics_a["author_entropy"] + metrics_b["author_entropy"]) / 2.0

    overall_score = (
        (graph_sim * 0.3) + (avg_burst * 0.4) + (min(1.0, entropy_score / 3.0) * 0.3)
    )
    is_covert = overall_score > 0.75

    return {
        "graph_similarity": round(graph_sim, 4),
        "timestamp_burstiness": round(avg_burst, 4),
        "entropy_score": round(entropy_score, 4),
        "overall_score": round(overall_score, 4),
        "is_covert_collaboration": is_covert,
    }
