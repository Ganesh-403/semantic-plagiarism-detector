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
similarity_heatmap.py
---------------------
Engine for computing pairwise document similarity heatmaps and
agglomerative document clustering.

Provides:
  • Pairwise cosine similarity matrix from stored embeddings
  • Hierarchical (agglomerative) clustering with configurable linkage
  • Heatmap data structures optimised for SVG/Canvas rendering
  • Cluster assignment per document with silhouette scoring
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HeatmapCell:
    """Single cell in the similarity heatmap matrix."""

    row_label: str
    col_label: str
    similarity: float  # 0.0 – 1.0
    row_index: int
    col_index: int


@dataclass
class HeatmapMatrix:
    """Full heatmap data structure for rendering."""

    labels: list[str]
    matrix: list[list[float]]  # NxN similarity values
    min_similarity: float
    max_similarity: float
    mean_similarity: float
    document_count: int
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "matrix": self.matrix,
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "document_count": self.document_count,
            "computed_at": self.computed_at,
        }


@dataclass
class Cluster:
    """A cluster of documents from hierarchical clustering."""

    cluster_id: int
    documents: list[str]
    centroid_score: float  # average intra-cluster similarity
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "documents": self.documents,
            "centroid_score": round(self.centroid_score, 4),
            "size": self.size,
        }


@dataclass
class ClusteringResult:
    """Complete clustering output."""

    clusters: list[Cluster]
    num_clusters: int
    silhouette_score: float
    linkage_method: str
    distance_threshold: float
    document_assignments: dict[str, int]  # filename → cluster_id
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "num_clusters": self.num_clusters,
            "silhouette_score": round(self.silhouette_score, 4),
            "linkage_method": self.linkage_method,
            "distance_threshold": self.distance_threshold,
            "document_assignments": self.document_assignments,
            "computed_at": self.computed_at,
        }


# ---------------------------------------------------------------------------
# Similarity Matrix Computation
# ---------------------------------------------------------------------------


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix for a set of embeddings.

    Args:
        embeddings: (N, D) float32 array of document embeddings.

    Returns:
        (N, N) float32 similarity matrix with 1.0 on the diagonal.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalised = embeddings / norms
    sim_matrix = normalised @ normalised.T
    # Clamp to [-1, 1] for numerical safety
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix.astype(np.float32)


def compute_heatmap(
    filenames: list[str],
    embeddings: np.ndarray,
) -> HeatmapMatrix:
    """
    Compute a full similarity heatmap from document embeddings.

    Args:
        filenames: List of document filenames (labels).
        embeddings: (N, D) array of document-level embeddings.

    Returns:
        HeatmapMatrix with the NxN similarity grid.
    """
    if len(filenames) < 2:
        return HeatmapMatrix(
            labels=filenames,
            matrix=[[1.0]] if len(filenames) == 1 else [],
            min_similarity=1.0,
            max_similarity=1.0,
            mean_similarity=1.0,
            document_count=len(filenames),
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    sim = _cosine_similarity_matrix(embeddings)

    # Extract upper triangle (excluding diagonal) for statistics
    n = len(filenames)
    upper_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_vals.append(float(sim[i, j]))

    matrix_list = [[round(float(sim[i, j]), 4) for j in range(n)] for i in range(n)]

    return HeatmapMatrix(
        labels=filenames,
        matrix=matrix_list,
        min_similarity=min(upper_vals) if upper_vals else 0.0,
        max_similarity=max(upper_vals) if upper_vals else 0.0,
        mean_similarity=sum(upper_vals) / len(upper_vals) if upper_vals else 0.0,
        document_count=n,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Agglomerative Clustering (single-linkage)
# ---------------------------------------------------------------------------


def _pairwise_distance(sim_matrix: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to distance matrix: d = 1 - s."""
    return (1.0 - sim_matrix).astype(np.float32)


def _agglomerative_clustering(
    dist_matrix: np.ndarray,
    labels: list[str],
    distance_threshold: float = 0.5,
) -> dict[str, int]:
    """
    Simple agglomerative (single-linkage) clustering.

    Merges the two closest clusters at each step until the minimum
    inter-cluster distance exceeds the threshold.

    Returns:
        Dict mapping filename → cluster_id.
    """
    n = len(labels)
    # Each document starts in its own cluster
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    next_id = n

    # Work copy of the distance matrix
    D = dist_matrix.copy()
    active = list(range(n))

    while len(active) > 1:
        # Find the minimum distance pair among active clusters
        min_dist = float("inf")
        min_i, min_j = -1, -1

        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                ci, cj = active[idx_i], active[idx_j]
                d = float(D[ci, cj])
                if d < min_dist:
                    min_dist = d
                    min_i = idx_i
                    min_j = idx_j

        if min_dist > distance_threshold:
            break

        ci, cj = active[min_i], active[min_j]

        # Merge cj into ci
        clusters[ci].extend(clusters.pop(cj, []))
        active.pop(min_j)

        # Update distances (single-linkage: take minimum)
        for k in active:
            if k != ci:
                new_dist = min(float(D[ci, k]), float(D[cj, k]))
                D[ci, k] = new_dist
                D[k, ci] = new_dist

    # Build label → cluster_id mapping
    assignments: dict[str, int] = {}
    for cid, members in clusters.items():
        for member_idx in members:
            assignments[labels[member_idx]] = cid

    return assignments


def _compute_silhouette(
    dist_matrix: np.ndarray,
    labels: list[str],
    assignments: dict[str, int],
) -> float:
    """Compute the average silhouette score for the clustering."""
    n = len(labels)
    if n < 2:
        return 0.0

    label_to_idx = {label: i for i, label in enumerate(labels)}
    unique_clusters = set(assignments.values())
    if len(unique_clusters) <= 1:
        return 0.0

    silhouettes: list[float] = []

    for i, label_i in enumerate(labels):
        ci = assignments.get(label_i)
        if ci is None:
            continue

        # a(i) = mean distance to same-cluster members
        same_cluster = [
            j
            for j, label_j in enumerate(labels)
            if assignments.get(label_j) == ci and j != i
        ]
        if not same_cluster:
            silhouettes.append(0.0)
            continue
        a_i = sum(float(dist_matrix[i, j]) for j in same_cluster) / len(same_cluster)

        # b(i) = min mean distance to other clusters
        b_i = float("inf")
        for other_cluster in unique_clusters:
            if other_cluster == ci:
                continue
            other_members = [
                j
                for j, label_j in enumerate(labels)
                if assignments.get(label_j) == other_cluster
            ]
            if not other_members:
                continue
            mean_dist = sum(float(dist_matrix[i, j]) for j in other_members) / len(
                other_members
            )
            b_i = min(b_i, mean_dist)

        if b_i == float("inf"):
            silhouettes.append(0.0)
        else:
            silhouettes.append(
                (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            )

    return sum(silhouettes) / len(silhouettes) if silhouettes else 0.0


def cluster_documents(
    filenames: list[str],
    embeddings: np.ndarray,
    distance_threshold: float = 0.5,
    linkage: str = "single",
) -> ClusteringResult:
    """
    Perform agglomerative clustering on document embeddings.

    Args:
        filenames: List of document filenames.
        embeddings: (N, D) document embedding array.
        distance_threshold: Maximum inter-cluster distance to merge.
        linkage: Linkage method (currently only 'single' supported).

    Returns:
        ClusteringResult with cluster assignments and quality metrics.
    """
    if len(filenames) < 2:
        return ClusteringResult(
            clusters=[],
            num_clusters=len(filenames),
            silhouette_score=0.0,
            linkage_method=linkage,
            distance_threshold=distance_threshold,
            document_assignments={f: 0 for f in filenames},
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    sim = _cosine_similarity_matrix(embeddings)
    dist = _pairwise_distance(sim)
    assignments = _agglomerative_clustering(dist, filenames, distance_threshold)

    # Build cluster objects
    cluster_members: dict[int, list[str]] = {}
    for fname, cid in assignments.items():
        cluster_members.setdefault(cid, []).append(fname)

    clusters: list[Cluster] = []
    for cid, members in sorted(cluster_members.items()):
        # Compute average intra-cluster similarity
        member_indices = [filenames.index(m) for m in members if m in filenames]
        if len(member_indices) > 1:
            sims = []
            for i in range(len(member_indices)):
                for j in range(i + 1, len(member_indices)):
                    sims.append(float(sim[member_indices[i], member_indices[j]]))
            centroid = sum(sims) / len(sims) if sims else 0.0
        else:
            centroid = 1.0

        clusters.append(
            Cluster(
                cluster_id=cid,
                documents=members,
                centroid_score=centroid,
                size=len(members),
            )
        )

    silhouette = _compute_silhouette(dist, filenames, assignments)

    return ClusteringResult(
        clusters=clusters,
        num_clusters=len(clusters),
        silhouette_score=silhouette,
        linkage_method=linkage,
        distance_threshold=distance_threshold,
        document_assignments=assignments,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Hotspot Detection
# ---------------------------------------------------------------------------


def detect_similarity_hotspots(
    filenames: list[str],
    sim_matrix: np.ndarray,
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Identify document pairs with suspiciously high similarity.

    Returns:
        List of hotspot dicts with doc_a, doc_b, similarity.
    """
    hotspots: list[dict[str, Any]] = []
    n = len(filenames)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                hotspots.append(
                    {
                        "doc_a": filenames[i],
                        "doc_b": filenames[j],
                        "similarity": round(s, 4),
                    }
                )
    hotspots.sort(key=lambda h: h["similarity"], reverse=True)
    return hotspots


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def heatmap_to_svg_data(
    heatmap: HeatmapMatrix,
    cell_size: int = 20,
) -> dict[str, Any]:
    """
    Convert a HeatmapMatrix to SVG-ready data for client rendering.

    Returns dict with cell_size, width, height, and cells (each with
    x, y, color, label, value).
    """
    n = heatmap.document_count
    # Short labels (truncate to 12 chars)
    short_labels = [l[:12] + "…" if len(l) > 12 else l for l in heatmap.labels]

    width = n * cell_size + 120  # extra for labels
    height = n * cell_size + 120

    cells = []
    for i in range(n):
        for j in range(n):
            val = heatmap.matrix[i][j]
            # Color: white (0) → amber (0.5) → red (1.0)
            if val <= 0.5:
                r = 255
                g = int(255 - val * 2 * 100)
                b = int(245 - val * 2 * 100)
            else:
                r = int(255 - (val - 0.5) * 2 * 155)
                g = int(155 - (val - 0.5) * 2 * 105)
                b = int(145 - (val - 0.5) * 2 * 100)
            color = f"rgb({r},{g},{b})"

            cells.append(
                {
                    "x": j * cell_size + 120,
                    "y": i * cell_size + 120,
                    "color": color,
                    "row_label": short_labels[i],
                    "col_label": short_labels[j],
                    "value": round(val, 4),
                }
            )

    return {
        "cell_size": cell_size,
        "width": width,
        "height": height,
        "labels": short_labels,
        "cells": cells,
    }
