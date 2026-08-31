"""src/api/routers/anomalies.py - Anomaly Detection API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import (
    AnomalyAlertResponse,
    AnomalyAlertListResponse,
    AnomalyScanResponse,
    AnomalyScanListResponse,
    AnomalySummaryResponse,
    AnomalySeverityDistResponse,
    AnomalyConfigResponse,
    ErrorResponse,
)
from src.db.anomaly_alerts_db import anomaly_repo, init_anomaly_alerts_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Anomaly Detection"])

try:
    init_anomaly_alerts_db()
except Exception:
    logger.warning("Anomaly alerts DB init deferred")


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------

@router.post(
    "/api/v1/anomalies/scans",
    status_code=status.HTTP_201_CREATED,
    responses={500: {"model": ErrorResponse}},
)
async def create_scan(
    request: Request,
    scan_type: str = Query("full", description="Scan type"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Start a new anomaly detection scan."""
    try:
        scan_id = anomaly_repo.create_scan(
            scan_type=scan_type,
            triggered_by=_user.get("sub", "unknown"),
        )
        return {"scan_id": scan_id, "status": "running", "scan_type": scan_type}
    except Exception as e:
        logger.error("Failed to create scan: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/anomalies/scans",
    responses={200: {"description": "List of scans"}, 500: {"model": ErrorResponse}},
)
async def list_scans(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    scan_status: Optional[str] = Query(None, alias="status"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """List anomaly detection scans."""
    try:
        return anomaly_repo.list_scans(page=page, per_page=per_page, status=scan_status)
    except Exception as e:
        logger.error("Failed to list scans: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/anomalies/scans/{scan_id}",
    responses={200: {"description": "Scan details"}, 404: {"description": "Not found"}},
)
async def get_scan(
    scan_id: int,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get details of a specific scan."""
    scan = anomaly_repo.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@router.post(
    "/api/v1/anomalies/scans/{scan_id}/complete",
    responses={200: {"description": "Scan completed"}, 404: {"description": "Not found"}},
)
async def complete_scan(
    scan_id: int,
    documents_scanned: int = Query(0, ge=0),
    anomalies_found: int = Query(0, ge=0),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark a scan as completed."""
    scan = anomaly_repo.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    anomaly_repo.complete_scan(scan_id, documents_scanned, anomalies_found)
    return {"scan_id": scan_id, "status": "completed"}


@router.post(
    "/api/v1/anomalies/scans/{scan_id}/fail",
    responses={200: {"description": "Scan failed"}, 404: {"description": "Not found"}},
)
async def fail_scan(
    scan_id: int,
    error_message: str = Query("Unknown error"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark a scan as failed."""
    scan = anomaly_repo.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    anomaly_repo.fail_scan(scan_id, error_message)
    return {"scan_id": scan_id, "status": "failed"}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.post(
    "/api/v1/anomalies/alerts",
    status_code=status.HTTP_201_CREATED,
    responses={500: {"model": ErrorResponse}},
)
async def create_alert(
    request: Request,
    scan_id: Optional[int] = Query(None),
    anomaly_type: str = Query(..., description="Anomaly type"),
    severity: str = Query("info", description="Severity level"),
    title: str = Query(..., description="Alert title"),
    description: str = Query("", description="Detailed description"),
    confidence: float = Query(0.0, ge=0.0, le=1.0),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Create a new anomaly alert."""
    try:
        alert_id = anomaly_repo.create_alert(
            scan_id=scan_id,
            anomaly_type=anomaly_type,
            severity=severity,
            title=title,
            description=description,
            confidence=confidence,
        )
        return {"alert_id": alert_id, "anomaly_type": anomaly_type, "severity": severity}
    except Exception as e:
        logger.error("Failed to create alert: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/anomalies/alerts",
    responses={200: {"description": "List of alerts"}, 500: {"model": ErrorResponse}},
)
async def list_alerts(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    resolved: Optional[bool] = Query(None),
    scan_id: Optional[int] = Query(None),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """List anomaly alerts with filtering."""
    try:
        return anomaly_repo.list_alerts(
            page=page, per_page=per_page,
            severity=severity, anomaly_type=anomaly_type,
            acknowledged=acknowledged, resolved=resolved,
            scan_id=scan_id,
        )
    except Exception as e:
        logger.error("Failed to list alerts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/anomalies/alerts/{alert_id}",
    responses={200: {"description": "Alert details"}, 404: {"description": "Not found"}},
)
async def get_alert(
    alert_id: int,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get details of a specific alert."""
    alert = anomaly_repo.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.put(
    "/api/v1/anomalies/alerts/{alert_id}/acknowledge",
    responses={200: {"description": "Acknowledged"}, 404: {"description": "Not found"}},
)
async def acknowledge_alert(
    alert_id: int,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Acknowledge an alert."""
    acked = anomaly_repo.acknowledge_alert(alert_id, by=_user.get("sub", "system"))
    if not acked:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    return {"alert_id": alert_id, "acknowledged": True}


@router.put(
    "/api/v1/anomalies/alerts/{alert_id}/resolve",
    responses={200: {"description": "Resolved"}, 404: {"description": "Not found"}},
)
async def resolve_alert(
    alert_id: int,
    notes: str = Query("", description="Resolution notes"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Resolve an alert."""
    resolved = anomaly_repo.resolve_alert(alert_id, by=_user.get("sub", "system"), notes=notes)
    if not resolved:
        raise HTTPException(status_code=404, detail="Alert not found or already resolved")
    return {"alert_id": alert_id, "resolved": True}


@router.put(
    "/api/v1/anomalies/alerts/read-all",
    responses={200: {"description": "All acknowledged"}},
)
async def acknowledge_all_alerts(
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Acknowledge all unacknowledged alerts."""
    count = anomaly_repo.acknowledge_all(by=_user.get("sub", "system"))
    return {"acknowledged_count": count}


@router.delete(
    "/api/v1/anomalies/alerts/{alert_id}",
    responses={200: {"description": "Deleted"}, 404: {"description": "Not found"}},
)
async def delete_alert(
    alert_id: int,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete an anomaly alert."""
    deleted = anomaly_repo.delete_alert(alert_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"deleted": True, "alert_id": alert_id}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@router.get(
    "/api/v1/anomalies/analytics/summary",
    responses={200: {"description": "Analytics summary"}},
)
async def analytics_summary(
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return aggregate anomaly statistics."""
    return anomaly_repo.analytics_summary()


@router.get(
    "/api/v1/anomalies/analytics/severity",
    responses={200: {"description": "Severity distribution"}},
)
async def severity_distribution(
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return alert counts per severity level."""
    return {"distribution": anomaly_repo.severity_distribution()}


@router.get(
    "/api/v1/anomalies/analytics/types",
    responses={200: {"description": "Type distribution"}},
)
async def type_distribution(
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return alert counts per anomaly type."""
    return {"distribution": anomaly_repo.type_distribution()}


@router.get(
    "/api/v1/anomalies/analytics/high-confidence",
    responses={200: {"description": "High-confidence alerts"}},
)
async def high_confidence_alerts(
    min_confidence: float = Query(0.8, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=50),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return high-confidence unresolved alerts."""
    return {"alerts": anomaly_repo.high_confidence_alerts(min_confidence, limit)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@router.get(
    "/api/v1/anomalies/config",
    responses={200: {"description": "Detection config"}},
)
async def get_config(
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get the current anomaly detection configuration."""
    return anomaly_repo.get_config()


@router.put(
    "/api/v1/anomalies/config",
    responses={200: {"description": "Updated config"}},
)
async def update_config(
    request: Request,
    z_score_threshold: Optional[float] = Query(None),
    cluster_min_size: Optional[int] = Query(None),
    cluster_similarity: Optional[float] = Query(None),
    collusion_threshold: Optional[float] = Query(None),
    template_threshold: Optional[float] = Query(None),
    enable_statistical: Optional[bool] = Query(None),
    enable_cluster: Optional[bool] = Query(None),
    enable_pattern: Optional[bool] = Query(None),
    enable_collusion: Optional[bool] = Query(None),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Update anomaly detection configuration."""
    updates = {}
    if z_score_threshold is not None:
        updates["z_score_threshold"] = z_score_threshold
    if cluster_min_size is not None:
        updates["cluster_min_size"] = cluster_min_size
    if cluster_similarity is not None:
        updates["cluster_similarity"] = cluster_similarity
    if collusion_threshold is not None:
        updates["collusion_threshold"] = collusion_threshold
    if template_threshold is not None:
        updates["template_threshold"] = template_threshold
    if enable_statistical is not None:
        updates["enable_statistical"] = int(enable_statistical)
    if enable_cluster is not None:
        updates["enable_cluster"] = int(enable_cluster)
    if enable_pattern is not None:
        updates["enable_pattern"] = int(enable_pattern)
    if enable_collusion is not None:
        updates["enable_collusion"] = int(enable_collusion)

    return anomaly_repo.update_config(**updates)
