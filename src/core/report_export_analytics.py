"""
report_export_analytics.py
--------------------------
Analytics engine computing trend reports, severity distributions,
per-document risk summaries, and time-series aggregations from
corpus database scan_history and plagiarism_incidents tables.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendPoint:
    """Single data point in a time-series trend."""
    timestamp: str
    document_count: int
    avg_similarity: float
    max_similarity: float
    flagged_count: int
    threshold_used: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityBucket:
    """Aggregated severity distribution bucket."""
    label: str
    count: int
    percentage: float
    avg_score: float
    min_score: float
    max_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRiskProfile:
    """Risk profile for a single document across all incidents."""
    filename: str
    incident_count: int
    avg_similarity: float
    max_similarity: float
    severity: str
    last_flagged: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticsSummary:
    """Top-level summary of all computed analytics."""
    total_scans: int
    total_incidents: int
    total_documents: int
    avg_similarity_overall: float
    max_similarity_overall: float
    flagged_rate: float
    severity_distribution: List[SeverityBucket]
    top_risk_documents: List[DocumentRiskProfile]
    daily_trends: List[TrendPoint]
    weekly_trends: List[TrendPoint]
    monthly_trends: List[TrendPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scans": self.total_scans,
            "total_incidents": self.total_incidents,
            "total_documents": self.total_documents,
            "avg_similarity_overall": self.avg_similarity_overall,
            "max_similarity_overall": self.max_similarity_overall,
            "flagged_rate": self.flagged_rate,
            "severity_distribution": [b.to_dict() for b in self.severity_distribution],
            "top_risk_documents": [d.to_dict() for d in self.top_risk_documents],
            "daily_trends": [t.to_dict() for t in self.daily_trends],
            "weekly_trends": [t.to_dict() for t in self.weekly_trends],
            "monthly_trends": [t.to_dict() for t in self.monthly_trends],
        }


# ── Severity thresholds ──────────────────────────────────────────────────────

_SEVERITY_THRESHOLDS = [("High", 0.90), ("Medium", 0.75), ("Low", 0.0)]


def classify_severity(score: float) -> str:
    """Classify a similarity score into a severity label."""
    clamped = max(0.0, min(1.0, float(score)))
    for label, threshold in _SEVERITY_THRESHOLDS:
        if clamped >= threshold:
            return label
    return "Low"


# ── Trend aggregation ────────────────────────────────────────────────────────


def _aggregate_trends(scan_history: List[Dict[str, Any]], period: str) -> List[TrendPoint]:
    """Aggregate scan_history records into daily, weekly, or monthly trends."""
    if not scan_history:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in scan_history:
        ts = record.get("timestamp", "")
        if not ts:
            continue
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue

        if period == "daily":
            key = dt.strftime("%Y-%m-%d")
        elif period == "weekly":
            iso = dt.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
        else:
            key = dt.strftime("%Y-%m")
        buckets[key].append(record)

    points: List[TrendPoint] = []
    for key in sorted(buckets):
        group = buckets[key]
        points.append(TrendPoint(
            timestamp=key,
            document_count=sum(r.get("document_count", 0) for r in group),
            avg_similarity=round(statistics.mean([r.get("avg_similarity", 0.0) for r in group]), 4),
            max_similarity=round(max(r.get("max_similarity", 0.0) for r in group), 4),
            flagged_count=sum(r.get("flagged_count", 0) for r in group),
            threshold_used=round(statistics.mean([r.get("threshold_used", 0.59) for r in group]), 4),
        ))
    return points


# ── Severity distribution ────────────────────────────────────────────────────


def compute_severity_distribution(incidents: List[Dict[str, Any]]) -> List[SeverityBucket]:
    """Compute severity distribution from plagiarism incident records."""
    if not incidents:
        return []

    severity_map: Dict[str, List[float]] = defaultdict(list)
    for inc in incidents:
        try:
            score = max(0.0, min(1.0, float(inc.get("similarity_score", 0.0))))
        except (TypeError, ValueError):
            continue
        severity_map[classify_severity(score)].append(score)

    total = sum(len(v) for v in severity_map.values())
    if total == 0:
        return []

    buckets: List[SeverityBucket] = []
    for label in ("High", "Medium", "Low"):
        group = severity_map.get(label, [])
        if not group:
            continue
        buckets.append(SeverityBucket(
            label=label, count=len(group),
            percentage=round(len(group) / total * 100, 2),
            avg_score=round(statistics.mean(group), 4),
            min_score=round(min(group), 4),
            max_score=round(max(group), 4),
        ))
    buckets.sort(key=lambda b: b.count, reverse=True)
    return buckets


# ── Per-document risk profiling ──────────────────────────────────────────────


def compute_document_risk_profiles(incidents: List[Dict[str, Any]], top_n: int = 10) -> List[DocumentRiskProfile]:
    """Build per-document risk profiles from plagiarism incidents."""
    doc_data: Dict[str, Dict[str, Any]] = {}

    for inc in incidents:
        for doc_key in ("document_a", "document_b"):
            filename = inc.get(doc_key, "")
            if not filename:
                continue
            try:
                score = max(0.0, min(1.0, float(inc.get("similarity_score", 0.0))))
            except (TypeError, ValueError):
                continue

            if filename not in doc_data:
                doc_data[filename] = {"scores": [], "last_flagged": inc.get("date_flagged", "")}
            doc_data[filename]["scores"].append(score)
            flagged = inc.get("date_flagged", "")
            if flagged > doc_data[filename]["last_flagged"]:
                doc_data[filename]["last_flagged"] = flagged

    profiles = []
    for filename, data in doc_data.items():
        scores = data["scores"]
        profiles.append(DocumentRiskProfile(
            filename=filename, incident_count=len(scores),
            avg_similarity=round(statistics.mean(scores), 4),
            max_similarity=round(max(scores), 4),
            severity=classify_severity(max(scores)),
            last_flagged=data["last_flagged"],
        ))
    profiles.sort(key=lambda p: (p.incident_count, p.max_similarity), reverse=True)
    return profiles[:top_n]


# ── Flagged rate ──────────────────────────────────────────────────────────────


def compute_flagged_rate(scan_history: List[Dict[str, Any]]) -> float:
    """Proportion of scan sessions that produced at least one flagged pair."""
    if not scan_history:
        return 0.0
    flagged = sum(1 for r in scan_history if r.get("flagged_count", 0) > 0)
    return round(flagged / len(scan_history), 4)


# ── Rolling averages ─────────────────────────────────────────────────────────


def compute_rolling_averages(trend_points: List[TrendPoint], window: int = 7) -> List[Dict[str, Any]]:
    """Compute rolling averages for similarity and flagged counts."""
    if not trend_points:
        return []
    sims = [p.avg_similarity for p in trend_points]
    flags = [float(p.flagged_count) for p in trend_points]
    results = []
    for i in range(len(trend_points)):
        start = max(0, i - window + 1)
        results.append({
            "timestamp": trend_points[i].timestamp,
            "rolling_avg_similarity": round(statistics.mean(sims[start:i + 1]), 4),
            "rolling_avg_flagged": round(statistics.mean(flags[start:i + 1]), 4),
        })
    return results


# ── Anomaly detection ────────────────────────────────────────────────────────


def detect_scan_anomalies(trend_points: List[TrendPoint], z_threshold: float = 2.0) -> List[Dict[str, Any]]:
    """Detect anomalous scans using Z-score on flagged_count."""
    if len(trend_points) < 3:
        return []
    values = [p.flagged_count for p in trend_points]
    mean_v, std_v = statistics.mean(values), statistics.stdev(values)
    if std_v == 0:
        return []
    return [
        {"timestamp": p.timestamp, "flagged_count": p.flagged_count,
         "z_score": round((p.flagged_count - mean_v) / std_v, 4),
         "direction": "spike" if p.flagged_count > mean_v else "dip"}
        for p in trend_points
        if abs((p.flagged_count - mean_v) / std_v) >= z_threshold
    ]


# ── Trend deltas ─────────────────────────────────────────────────────────────


def compute_trend_deltas(trend_points: List[TrendPoint]) -> Dict[str, Any]:
    """Period-over-period deltas for key metrics."""
    if len(trend_points) < 2:
        return {"avg_similarity_delta": 0.0, "avg_similarity_pct_change": 0.0,
                "flagged_count_delta": 0, "flagged_count_pct_change": 0.0}
    prev, curr = trend_points[-2], trend_points[-1]
    sim_d = round(curr.avg_similarity - prev.avg_similarity, 4)
    sim_p = round(sim_d / prev.avg_similarity * 100, 2) if prev.avg_similarity else 0.0
    fl_d = curr.flagged_count - prev.flagged_count
    fl_p = round(fl_d / prev.flagged_count * 100, 2) if prev.flagged_count else 0.0
    return {"avg_similarity_delta": sim_d, "avg_similarity_pct_change": sim_p,
            "flagged_count_delta": fl_d, "flagged_count_pct_change": fl_p}


# ── Main analytics pipeline ─────────────────────────────────────────────────


def generate_analytics_summary(
    scan_history: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
    total_documents: int = 0,
    top_n_documents: int = 10,
) -> AnalyticsSummary:
    """Run the full analytics pipeline and return a comprehensive summary."""
    return AnalyticsSummary(
        total_scans=len(scan_history),
        total_incidents=len(incidents),
        total_documents=total_documents,
        avg_similarity_overall=round(statistics.mean([r.get("avg_similarity", 0.0) for r in scan_history]), 4) if scan_history else 0.0,
        max_similarity_overall=round(max((r.get("max_similarity", 0.0) for r in scan_history), default=0.0), 4),
        flagged_rate=compute_flagged_rate(scan_history),
        severity_distribution=compute_severity_distribution(incidents),
        top_risk_documents=compute_document_risk_profiles(incidents, top_n=top_n_documents),
        daily_trends=_aggregate_trends(scan_history, "daily"),
        weekly_trends=_aggregate_trends(scan_history, "weekly"),
        monthly_trends=_aggregate_trends(scan_history, "monthly"),
    )


# ── Threshold sensitivity analysis ───────────────────────────────────────────


def threshold_sensitivity_analysis(
    incidents: List[Dict[str, Any]],
    scan_history: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Analyze how many incidents would be flagged at various thresholds."""
    if thresholds is None:
        thresholds = [round(0.30 + i * 0.05, 2) for i in range(14)]

    all_scores = []
    for inc in incidents:
        try:
            all_scores.append(float(inc.get("similarity_score", 0.0)))
        except (TypeError, ValueError):
            continue

    total = len(all_scores)
    return [
        {"threshold": round(t, 2),
         "incident_count": sum(1 for s in all_scores if s >= t),
         "pct_of_total": round(sum(1 for s in all_scores if s >= t) / total * 100, 2) if total else 0.0}
        for t in sorted(thresholds)
    ]
