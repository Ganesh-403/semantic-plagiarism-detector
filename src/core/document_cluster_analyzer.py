"""
src/core/document_cluster_analyzer.py
--------------------------------------
Document cluster analysis for plagiarism detection.

Groups documents into semantic clusters based on embedding similarity,
enabling detection of collusion rings, shared-source families, and
template-based plagiarism patterns that pairwise comparison alone
may miss.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import (
    DEFAULT_THRESHOLDS,
    HIGH_SEVERITY,
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    SimilarityThresholds,
    normalize_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ClusterMethod(str, Enum):
    """Supported clustering algorithms."""
    AGGLOMERATIVE = "agglomerative"
    GRAPH_COMMUNITY = "graph_community"
    THRESHOLD = "threshold"


class ClusterRiskLevel(str, Enum):
    """Risk classification for a document cluster."""
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DocumentPair:
    """A pairwise similarity record between two documents."""
    doc_a: str
    doc_b: str
    similarity: float

    def __post_init__(self) -> None:
        if self.similarity < 0.0 or self.similarity > 1.0:
            raise ValueError(
                f"Similarity must be in [0.0, 1.0], got {self.similarity}"
            )

    def canonical_key(self) -> Tuple[str, str]:
        """Return a hashable key with documents in lexicographic order."""
        return (min(self.doc_a, self.doc_b), max(self.doc_a, self.doc_b))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "similarity": round(self.similarity, 6),
        }


@dataclass
class Cluster:
    """A group of semantically related documents."""
    cluster_id: int
    documents: List[str]
    avg_internal_similarity: float
    max_internal_similarity: float
    risk_level: ClusterRiskLevel
    centroid_embedding: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    inter_cluster_distances: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.documents)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "documents": sorted(self.documents),
            "size": self.size,
            "avg_internal_similarity": round(self.avg_internal_similarity, 6),
            "max_internal_similarity": round(self.max_internal_similarity, 6),
            "risk_level": self.risk_level.value,
            "metadata": self.metadata,
        }


@dataclass
class ClusterAnalysisResult:
    """Complete output of a cluster analysis run."""
    clusters: List[Cluster]
    method: ClusterMethod
    similarity_threshold: float
    total_documents: int
    clustered_documents: int
    singleton_documents: List[str]
    recommendations: List[str]
    processing_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_clusters(self) -> int:
        return len(self.clusters)

    @property
    def high_risk_clusters(self) -> List[Cluster]:
        return [
            c for c in self.clusters
            if c.risk_level in (ClusterRiskLevel.HIGH, ClusterRiskLevel.CRITICAL)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "similarity_threshold": self.similarity_threshold,
            "total_documents": self.total_documents,
            "total_clusters": self.total_clusters,
            "clustered_documents": self.clustered_documents,
            "singleton_count": len(self.singleton_documents),
            "high_risk_cluster_count": len(self.high_risk_clusters),
            "clusters": [c.to_dict() for c in self.clusters],
            "singleton_documents": sorted(self.singleton_documents),
            "recommendations": self.recommendations,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def _build_similarity_graph(
    doc_names: List[str],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build an adjacency-list representation of the similarity graph.

    Only edges with similarity >= *threshold* are included.

    Args:
        doc_names: Ordered document names matching matrix indices.
        similarity_matrix: NxN similarity matrix.
        threshold: Minimum similarity to create an edge.

    Returns:
        Adjacency list: {doc_name: [(neighbor, similarity), ...]}.
    """
    n = len(doc_names)
    graph: Dict[str, List[Tuple[str, float]]] = {name: [] for name in doc_names}
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                graph[doc_names[i]].append((doc_names[j], sim))
                graph[doc_names[j]].append((doc_names[i], sim))
    return graph


def _connected_components(
    graph: Dict[str, List[Tuple[str, float]]],
) -> List[List[str]]:
    """Find connected components via BFS."""
    visited: set[str] = set()
    components: List[List[str]] = []
    for start in graph:
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        component: List[str] = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _compute_cluster_similarity_stats(
    cluster_docs: List[str],
    doc_index: Dict[str, int],
    similarity_matrix: np.ndarray,
) -> Tuple[float, float]:
    """Compute average and max internal similarity for a set of documents."""
    if len(cluster_docs) < 2:
        return 1.0, 1.0

    pairs = []
    for i in range(len(cluster_docs)):
        for j in range(i + 1, len(cluster_docs)):
            idx_a = doc_index.get(cluster_docs[i])
            idx_b = doc_index.get(cluster_docs[j])
            if idx_a is not None and idx_b is not None:
                pairs.append(float(similarity_matrix[idx_a, idx_b]))

    if not pairs:
        return 0.0, 0.0
    return float(np.mean(pairs)), float(np.max(pairs))


def _compute_centroid(
    cluster_docs: List[str],
    doc_index: Dict[str, int],
    embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute the mean embedding centroid for a cluster."""
    indices = [doc_index[d] for d in cluster_docs if d in doc_index]
    if not indices:
        return None
    vectors = embeddings[indices]
    return np.mean(vectors, axis=0)


def _classify_cluster_risk(
    avg_sim: float,
    max_sim: float,
    size: int,
    thresholds: SimilarityThresholds,
) -> ClusterRiskLevel:
    """Determine cluster risk based on internal similarity and size."""
    if max_sim >= thresholds.high and size >= 3:
        return ClusterRiskLevel.CRITICAL
    if max_sim >= thresholds.high or (avg_sim >= thresholds.medium and size >= 4):
        return ClusterRiskLevel.HIGH
    if avg_sim >= thresholds.plagiarism and size >= 2:
        return ClusterRiskLevel.ELEVATED
    return ClusterRiskLevel.LOW


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def cluster_by_threshold(
    doc_names: List[str],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> List[List[str]]:
    """Group documents using connected-components at a fixed threshold.

    Any two documents with similarity >= *threshold* are linked.  Connected
    components become clusters.

    Args:
        doc_names: Ordered document names.
        similarity_matrix: NxN cosine similarity matrix.
        threshold: Minimum similarity to link documents.

    Returns:
        List of clusters, each a list of document names.
    """
    graph = _build_similarity_graph(doc_names, similarity_matrix, threshold)
    components = _connected_components(graph)
    return [sorted(c) for c in components if len(c) >= 2]


def cluster_by_agglomerative(
    doc_names: List[str],
    similarity_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> List[List[str]]:
    """Agglomerative hierarchical clustering on a similarity matrix.

    Converts similarity to distance (1 - sim) and performs single-linkage
    agglomerative clustering without requiring scipy.

    Args:
        doc_names: Ordered document names.
        similarity_matrix: NxN cosine similarity matrix.
        n_clusters: Desired number of clusters.  If *None*,
            *distance_threshold* must be supplied.
        distance_threshold: Merge cutoff distance.  Clusters separated
            by more than this distance are not merged.

    Returns:
        List of clusters, each a list of document names.
    """
    n = len(doc_names)
    if n == 0:
        return []

    if n_clusters is None and distance_threshold is None:
        distance_threshold = 0.25  # default: merge when sim >= 0.75

    # Convert similarity to distance
    dist_matrix = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)

    # Initialise each document as its own cluster
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    next_id = n

    def _cluster_distance(c1: List[int], c2: List[int]) -> float:
        """Single-linkage distance: minimum pairwise distance."""
        best = float("inf")
        for a in c1:
            for b in c2:
                if dist_matrix[a, b] < best:
                    best = dist_matrix[a, b]
        return best

    active_ids = list(clusters.keys())

    while len(active_ids) > 1:
        # Find the closest pair of clusters
        best_dist = float("inf")
        best_pair = (-1, -1)
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                d = _cluster_distance(
                    clusters[active_ids[i]], clusters[active_ids[j]]
                )
                if d < best_dist:
                    best_dist = d
                    best_pair = (active_ids[i], active_ids[j])

        if distance_threshold is not None and best_dist > distance_threshold:
            break

        if n_clusters is not None and len(active_ids) <= n_clusters:
            break

        # Merge
        id_a, id_b = best_pair
        merged = clusters[id_a] + clusters[id_b]
        clusters[next_id] = merged
        del clusters[id_a]
        del clusters[id_b]
        active_ids.remove(id_a)
        active_ids.remove(id_b)
        active_ids.append(next_id)
        next_id += 1

    return [
        sorted(doc_names[idx] for idx in cid_list)
        for cid_list in clusters.values()
        if len(cid_list) >= 2
    ]


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class DocumentClusterAnalyzer:
    """Orchestrates document clustering and cluster-level risk assessment.

    Usage::

        analyzer = DocumentClusterAnalyzer()
        result = analyzer.analyze(
            doc_names=["a.pdf", "b.pdf", "c.pdf"],
            similarity_matrix=sim_matrix,
            embeddings=emb_matrix,
        )
        print(result.to_dict())
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLDS.plagiarism,
        thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
        method: ClusterMethod = ClusterMethod.THRESHOLD,
    ) -> None:
        self.threshold = normalize_score(threshold)
        self.thresholds = thresholds
        self.method = method

    def analyze(
        self,
        doc_names: List[str],
        similarity_matrix: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        method: Optional[ClusterMethod] = None,
    ) -> ClusterAnalysisResult:
        """Run the full cluster analysis pipeline.

        Args:
            doc_names: Ordered document names matching matrix dimensions.
            similarity_matrix: NxN cosine similarity matrix.
            embeddings: Optional (N, D) embedding matrix for centroid computation.
            method: Override the clustering method for this run.

        Returns:
            ClusterAnalysisResult with clusters, risk levels, and recommendations.
        """
        t_start = datetime.now()
        active_method = method or self.method

        if len(doc_names) < 2:
            return self._empty_result(len(doc_names), active_method, t_start)

        # Validate matrix shape
        n = len(doc_names)
        if similarity_matrix.shape != (n, n):
            raise ValueError(
                f"similarity_matrix shape {similarity_matrix.shape} "
                f"does not match {n} documents"
            )

        # Run clustering
        raw_clusters = self._run_clustering(
            doc_names, similarity_matrix, active_method
        )

        # Build structured cluster objects
        doc_index = {name: i for i, name in enumerate(doc_names)}
        clustered_set: set[str] = set()
        clusters: List[Cluster] = []

        for cid, cluster_docs in enumerate(raw_clusters):
            avg_sim, max_sim = _compute_cluster_similarity_stats(
                cluster_docs, doc_index, similarity_matrix
            )
            risk = _classify_cluster_risk(
                avg_sim, max_sim, len(cluster_docs), self.thresholds
            )

            centroid = None
            if embeddings is not None and embeddings.size > 0:
                centroid = _compute_centroid(cluster_docs, doc_index, embeddings)

            cluster = Cluster(
                cluster_id=cid,
                documents=cluster_docs,
                avg_internal_similarity=avg_sim,
                max_internal_similarity=max_sim,
                risk_level=risk,
                centroid_embedding=centroid,
            )
            clusters.append(cluster)
            clustered_set.update(cluster_docs)

        singletons = sorted(set(doc_names) - clustered_set)

        # Sort clusters: highest risk first, then by size
        risk_order = {
            ClusterRiskLevel.CRITICAL: 0,
            ClusterRiskLevel.HIGH: 1,
            ClusterRiskLevel.ELEVATED: 2,
            ClusterRiskLevel.LOW: 3,
        }
        clusters.sort(key=lambda c: (risk_order[c.risk_level], -c.size))

        # Re-number after sorting
        for idx, cluster in enumerate(clusters):
            cluster.cluster_id = idx

        elapsed = (datetime.now() - t_start).total_seconds()
        recommendations = self._generate_recommendations(clusters, doc_names)

        return ClusterAnalysisResult(
            clusters=clusters,
            method=active_method,
            similarity_threshold=self.threshold,
            total_documents=n,
            clustered_documents=len(clustered_set),
            singleton_documents=singletons,
            recommendations=recommendations,
            processing_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_clustering(
        self,
        doc_names: List[str],
        similarity_matrix: np.ndarray,
        method: ClusterMethod,
    ) -> List[List[str]]:
        if method == ClusterMethod.THRESHOLD:
            return cluster_by_threshold(
                doc_names, similarity_matrix, self.threshold
            )
        if method == ClusterMethod.AGGLOMERATIVE:
            return cluster_by_agglomerative(
                doc_names, similarity_matrix, distance_threshold=1.0 - self.threshold
            )
        if method == ClusterMethod.GRAPH_COMMUNITY:
            return cluster_by_threshold(
                doc_names, similarity_matrix, self.threshold
            )
        raise ValueError(f"Unsupported clustering method: {method}")

    def _generate_recommendations(
        self,
        clusters: List[Cluster],
        doc_names: List[str],
    ) -> List[str]:
        """Produce human-readable recommendations from cluster findings."""
        recs: List[str] = []

        critical = [c for c in clusters if c.risk_level == ClusterRiskLevel.CRITICAL]
        high = [c for c in clusters if c.risk_level == ClusterRiskLevel.HIGH]
        elevated = [c for c in clusters if c.risk_level == ClusterRiskLevel.ELEVATED]

        if critical:
            total_docs = sum(c.size for c in critical)
            recs.append(
                f"🔴 {len(critical)} critical cluster(s) involving "
                f"{total_docs} documents. Immediate investigation recommended."
            )
        if high:
            total_docs = sum(c.size for c in high)
            recs.append(
                f"🟠 {len(high)} high-risk cluster(s) involving "
                f"{total_docs} documents. Manual review strongly advised."
            )
        if elevated:
            total_docs = sum(c.size for c in elevated)
            recs.append(
                f"🟡 {len(elevated)} elevated-risk cluster(s) involving "
                f"{total_docs} documents. Monitor for further patterns."
            )

        large_clusters = [c for c in clusters if c.size >= 5]
        if large_clusters:
            recs.append(
                f"⚠️ {len(large_clusters)} cluster(s) have 5+ documents, "
                "suggesting possible shared source or template usage."
            )

        if not recs:
            recs.append(
                "✅ No significant document clusters detected. "
                "The collection appears diverse."
            )

        return recs

    def _empty_result(
        self,
        total_docs: int,
        method: ClusterMethod,
        t_start: datetime,
    ) -> ClusterAnalysisResult:
        elapsed = (datetime.now() - t_start).total_seconds()
        return ClusterAnalysisResult(
            clusters=[],
            method=method,
            similarity_threshold=self.threshold,
            total_documents=total_docs,
            clustered_documents=0,
            singleton_documents=[],
            recommendations=["✅ Fewer than 2 documents; clustering not applicable."],
            processing_time_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_cluster(
    doc_names: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.59,
) -> List[Dict[str, Any]]:
    """Quick threshold-based clustering returning dicts.

    Useful for one-off analysis or Streamlit integration.

    Args:
        doc_names: Ordered document names.
        similarity_matrix: NxN similarity matrix.
        threshold: Minimum similarity for linkage.

    Returns:
        List of cluster dicts with 'documents', 'size', 'avg_similarity'.
    """
    raw = cluster_by_threshold(doc_names, similarity_matrix, threshold)
    doc_index = {name: i for i, name in enumerate(doc_names)}
    results = []
    for docs in raw:
        avg_sim, max_sim = _compute_cluster_similarity_stats(
            docs, doc_index, similarity_matrix
        )
        results.append({
            "documents": sorted(docs),
            "size": len(docs),
            "avg_similarity": round(avg_sim, 6),
            "max_similarity": round(max_sim, 6),
        })
    return sorted(results, key=lambda c: c["avg_similarity"], reverse=True)


def compute_document_risk_scores(
    doc_names: List[str],
    similarity_matrix: np.ndarray,
    thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-document risk scores from the similarity matrix.

    For each document, the risk score is the maximum similarity it has
    with any other document.  This identifies the most at-risk documents
    regardless of clustering.

    Args:
        doc_names: Ordered document names.
        similarity_matrix: NxN similarity matrix.
        thresholds: Severity thresholds for classification.

    Returns:
        Dict mapping document name to risk metadata.
    """
    n = len(doc_names)
    scores: Dict[str, Dict[str, Any]] = {}

    for i in range(n):
        max_sim = 0.0
        most_similar_doc = ""
        similarity_count = 0
        high_similarity_count = 0

        for j in range(n):
            if i == j:
                continue
            sim = float(similarity_matrix[i, j])
            if sim > max_sim:
                max_sim = sim
                most_similar_doc = doc_names[j]
            if sim >= thresholds.plagiarism:
                similarity_count += 1
            if sim >= thresholds.high:
                high_similarity_count += 1

        # Risk score: weighted combination
        risk_score = max_sim
        if high_similarity_count >= 3:
            risk_score = min(1.0, risk_score + 0.1)
        if similarity_count >= 5:
            risk_score = min(1.0, risk_score + 0.05)

        from src.core.config import severity_from_score
        severity = severity_from_score(max_sim, thresholds)

        scores[doc_names[i]] = {
            "max_similarity": round(max_sim, 6),
            "most_similar_document": most_similar_doc,
            "similar_pair_count": similarity_count,
            "high_similarity_pair_count": high_similarity_count,
            "risk_score": round(risk_score, 6),
            "severity": severity,
        }

    return scores


def generate_risk_summary(
    risk_scores: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-document risk scores into a summary report.

    Args:
        risk_scores: Output of :func:`compute_document_risk_scores`.

    Returns:
        Summary dict with statistics and flagged documents.
    """
    if not risk_scores:
        return {
            "total_documents": 0,
            "avg_risk_score": 0.0,
            "max_risk_score": 0.0,
            "flagged_count": 0,
            "critical_documents": [],
            "high_risk_documents": [],
        }

    all_scores = [v["risk_score"] for v in risk_scores.values()]
    flagged = {
        name: data
        for name, data in risk_scores.items()
        if data["severity"] in (MEDIUM_SEVERITY, HIGH_SEVERITY)
    }
    critical = [
        name for name, data in risk_scores.items()
        if data["severity"] == HIGH_SEVERITY
    ]
    high_risk = [
        name for name, data in risk_scores.items()
        if data["severity"] == MEDIUM_SEVERITY
    ]

    return {
        "total_documents": len(risk_scores),
        "avg_risk_score": round(float(np.mean(all_scores)), 6),
        "max_risk_score": round(float(np.max(all_scores)), 6),
        "flagged_count": len(flagged),
        "critical_documents": sorted(critical),
        "high_risk_documents": sorted(high_risk),
    }
