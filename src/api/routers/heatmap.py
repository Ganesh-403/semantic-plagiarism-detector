"""src/api/routers/heatmap.py - Similarity heatmap and clustering API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import (
    HeatmapSnapshotResponse,
    HeatmapSnapshotListResponse,
    ClusteringResultResponse,
    HotspotResponse,
    HotspotListResponse,
    HotspotSummaryResponse,
    ErrorResponse,
)
from src.db.heatmap_db import heatmap_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Similarity Heatmap & Clustering"])


# ---------------------------------------------------------------------------
# Compute Heatmap
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/heatmap/compute",
    response_model=HeatmapSnapshotResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def compute_heatmap(
    request: Request,
    notes: Optional[str] = Query(None, description="Optional notes for this snapshot"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """
    Compute the pairwise similarity heatmap from the current corpus embeddings
    and persist a snapshot.
    """
    try:
        from src.db.corpus_db import get_all_documents, get_all_embeddings
        from src.core.similarity_heatmap import compute_heatmap, detect_similarity_hotspots

        docs = get_all_documents()
        if len(docs) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Need at least 2 documents to compute a heatmap",
            )

        filenames = [d.filename for d in docs]
        embeddings = get_all_embeddings()

        if embeddings.shape[0] < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough embeddings to compute similarity",
            )

        heatmap = compute_heatmap(filenames, embeddings)

        # Save snapshot
        snapshot_id = heatmap_repo.save_snapshot(
            labels=heatmap.labels,
            matrix=heatmap.matrix,
            min_similarity=heatmap.min_similarity,
            max_similarity=heatmap.max_similarity,
            mean_similarity=heatmap.mean_similarity,
            computed_by=_user.get("sub", "unknown"),
            notes=notes,
        )

        # Detect and save hotspots
        import numpy as np
        mat = np.array(heatmap.matrix, dtype=np.float32)
        hotspots = detect_similarity_hotspots(filenames, mat, threshold=0.7)
        if hotspots:
            heatmap_repo.save_hotspots_batch(snapshot_id, hotspots)

        return HeatmapSnapshotResponse(
            snapshot_id=snapshot_id,
            document_count=heatmap.document_count,
            min_similarity=heatmap.min_similarity,
            max_similarity=heatmap.max_similarity,
            mean_similarity=heatmap.mean_similarity,
            computed_at=heatmap.computed_at,
            hotspots_found=len(hotspots),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compute heatmap: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/heatmap/snapshots/{snapshot_id}",
    response_model=HeatmapSnapshotResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Snapshot not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_heatmap_snapshot(
    snapshot_id: int,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Retrieve a saved heatmap snapshot with its full matrix."""
    try:
        snap = heatmap_repo.get_snapshot(snapshot_id)
        if not snap:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Heatmap snapshot #{snapshot_id} not found",
            )
        return HeatmapSnapshotResponse(
            snapshot_id=snap["snapshot_id"],
            labels=snap.get("labels_json", []),
            matrix=snap.get("matrix_json", []),
            document_count=snap["document_count"],
            min_similarity=snap["min_similarity"],
            max_similarity=snap["max_similarity"],
            mean_similarity=snap["mean_similarity"],
            computed_at=snap["computed_at"],
            computed_by=snap.get("computed_by"),
            notes=snap.get("notes"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/heatmap/snapshots",
    response_model=HeatmapSnapshotListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_heatmap_snapshots(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """List all heatmap snapshots."""
    offset = (page - 1) * per_page
    snapshots = heatmap_repo.list_snapshots(limit=per_page, offset=offset)
    total = heatmap_repo.count_snapshots()
    total_pages = (total + per_page - 1) // per_page if total > 0 else 0
    return HeatmapSnapshotListResponse(
        snapshots=snapshots,
        page=page,
        per_page=per_page,
        total_items=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1,
    )


@router.delete(
    "/api/v1/heatmap/snapshots/{snapshot_id}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Snapshot not found"},
    },
)
async def delete_heatmap_snapshot(
    snapshot_id: int,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete a heatmap snapshot."""
    deleted = heatmap_repo.delete_snapshot(snapshot_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot #{snapshot_id} not found",
        )
    return {"status": "deleted", "snapshot_id": snapshot_id}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/heatmap/cluster",
    response_model=ClusteringResultResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def compute_clustering(
    request: Request,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Distance threshold"),
    linkage: str = Query("single", description="Linkage method"),
    snapshot_id: Optional[int] = Query(None, description="Link to a heatmap snapshot"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Compute document clustering from corpus embeddings."""
    try:
        from src.db.corpus_db import get_all_documents, get_all_embeddings
        from src.core.similarity_heatmap import cluster_documents

        docs = get_all_documents()
        if len(docs) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Need at least 2 documents to cluster",
            )

        filenames = [d.filename for d in docs]
        embeddings = get_all_embeddings()

        if embeddings.shape[0] < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough embeddings for clustering",
            )

        result = cluster_documents(
            filenames=filenames,
            embeddings=embeddings,
            distance_threshold=threshold,
            linkage=linkage,
        )

        # Persist
        result_id = heatmap_repo.save_clustering(
            snapshot_id=snapshot_id,
            num_clusters=result.num_clusters,
            silhouette_score=result.silhouette_score,
            linkage_method=result.linkage_method,
            distance_threshold=result.distance_threshold,
            clusters=[c.to_dict() for c in result.clusters],
            assignments=result.document_assignments,
        )

        return ClusteringResultResponse(
            result_id=result_id,
            num_clusters=result.num_clusters,
            silhouette_score=result.silhouette_score,
            linkage_method=result.linkage_method,
            distance_threshold=result.distance_threshold,
            clusters=[c.to_dict() for c in result.clusters],
            document_assignments=result.document_assignments,
            computed_at=result.computed_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compute clustering: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/heatmap/clusters",
    response_model=list[ClusteringResultResponse],
    status_code=status.HTTP_200_OK,
)
async def list_clustering_results(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """List past clustering results."""
    results = heatmap_repo.list_clusterings(limit=limit)
    return [
        ClusteringResultResponse(
            result_id=r["result_id"],
            num_clusters=r["num_clusters"],
            silhouette_score=r["silhouette_score"],
            linkage_method=r["linkage_method"],
            distance_threshold=r["distance_threshold"],
            clusters=r.get("clusters_json", []),
            document_assignments=r.get("assignments_json", {}),
            computed_at=r["computed_at"],
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/heatmap/hotspots",
    response_model=HotspotListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_hotspots(
    request: Request,
    unresolved_only: bool = Query(False, description="Only unresolved"),
    min_similarity: Optional[float] = Query(None, description="Min similarity"),
    limit: int = Query(50, ge=1, le=200),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """List similarity hotspots (high-similarity document pairs)."""
    hotspots = heatmap_repo.get_hotspots(
        unresolved_only=unresolved_only,
        min_similarity=min_similarity,
        limit=limit,
    )
    return HotspotListResponse(
        hotspots=[HotspotResponse(**h) for h in hotspots],
        total=len(hotspots),
    )


@router.put(
    "/api/v1/heatmap/hotspots/{hotspot_id}/resolve",
    status_code=status.HTTP_200_OK,
)
async def resolve_hotspot(
    hotspot_id: int,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark a similarity hotspot as resolved."""
    resolved = heatmap_repo.resolve_hotspot(hotspot_id)
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotspot #{hotspot_id} not found",
        )
    return {"status": "resolved", "hotspot_id": hotspot_id}


@router.get(
    "/api/v1/heatmap/hotspots/summary",
    response_model=HotspotSummaryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_hotspot_summary(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get summary statistics for similarity hotspots."""
    summary = heatmap_repo.get_hotspot_summary()
    return HotspotSummaryResponse(**summary)


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/heatmap/purge",
    status_code=status.HTTP_200_OK,
)
async def purge_old_data(
    request: Request,
    days: int = Query(90, ge=7, le=365),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Purge old heatmap snapshots, clusterings, and resolved hotspots."""
    deleted = heatmap_repo.purge_old_data(days=days)
    return {"status": "purged", **deleted}
