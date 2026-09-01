"""
src/utils/report_exporter.py
----------------------------
Structured plagiarism analytics report exporter.

Consumes raw similarity matrices, flagged-pair lists, and cluster data
from the detection pipeline and produces machine-readable JSON and CSV
analytics reports suitable for dashboards, LMS integrations, and
compliance auditing.

Report sections:
  1. Similarity score statistics (mean, median, std, percentiles)
  2. Severity distribution breakdown
  3. Per-document risk ranking
  4. Cluster (collusion-ring) summary
  5. Metadata and timestamping
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.core.config import (
    DEFAULT_THRESHOLDS,
    HIGH_SEVERITY,
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    PLAGIARISM_THRESHOLD,
    SimilarityThresholds,
    normalize_score,
    severity_from_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes – canonical report schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityStatistics:
    """Aggregate statistics for a collection of similarity scores."""

    count: int
    mean: float
    median: float
    std_dev: float
    min_score: float
    max_score: float
    p25: float
    p75: float
    p90: float
    p95: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityDistribution:
    """Count and percentage of scores in each severity band."""

    total: int
    low_count: int
    low_pct: float
    medium_count: int
    medium_pct: float
    high_count: int
    high_pct: float
    flagged_count: int
    flagged_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRiskProfile:
    """Risk assessment for a single document against all others."""

    document_name: str
    max_similarity: float
    mean_similarity: float
    flagged_pair_count: int
    risk_level: str  # Low / Medium / High / Critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterSummary:
    """Summary for a detected plagiarism cluster."""

    cluster_id: int
    member_count: int
    documents: List[str]
    avg_internal_similarity: float
    max_internal_similarity: float
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlaggedPairRecord:
    """A single flagged document-pair record with explainable scoring."""

    doc_a: str
    doc_b: str
    similarity: float
    severity: str
    threshold_at_flag: float
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    semantic_contribution: Optional[float] = None
    lexical_contribution: Optional[float] = None
    hybrid_weight: Optional[float] = None
    threshold_margin: Optional[float] = None
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticsReport:
    """Top-level container for the full analytics report."""

    generated_at: str
    threshold: float
    total_documents: int
    total_evaluated_pairs: int
    similarity_stats: SimilarityStatistics
    severity_distribution: SeverityDistribution
    document_risk_profiles: List[DocumentRiskProfile]
    flagged_pairs: List[FlaggedPairRecord]
    cluster_summaries: List[ClusterSummary]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "threshold": self.threshold,
            "total_documents": self.total_documents,
            "total_evaluated_pairs": self.total_evaluated_pairs,
            "similarity_statistics": self.similarity_stats.to_dict(),
            "severity_distribution": self.severity_distribution.to_dict(),
            "document_risk_profiles": [p.to_dict() for p in self.document_risk_profiles],
            "flagged_pairs": [f.to_dict() for f in self.flagged_pairs],
            "cluster_summaries": [c.to_dict() for c in self.cluster_summaries],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float:
    """Coerce a value to float, returning 0.0 on failure."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Compute a percentile from an already-sorted sequence."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_similarity_statistics(
    scores: Union[pd.DataFrame, np.ndarray, List[float]],
) -> SimilarityStatistics:
    """Compute aggregate statistics for a set of similarity scores.

    When a DataFrame is provided, off-diagonal values are extracted
    (to ignore self-similarity = 1.0 on the diagonal).

    Args:
        scores: Similarity scores as a DataFrame, ndarray, or plain list.

    Returns:
        SimilarityStatistics with all computed metrics.
    """
    flat: List[float] = []

    if isinstance(scores, pd.DataFrame):
        arr = scores.values
        n = arr.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    flat.append(_safe_float(arr[i, j]))
    elif isinstance(scores, np.ndarray):
        flat = [_safe_float(v) for v in scores.flatten()]
    elif isinstance(scores, (list, tuple)):
        flat = [_safe_float(v) for v in scores]
    else:
        flat = [_safe_float(scores)]

    if not flat:
        return SimilarityStatistics(
            count=0, mean=0.0, median=0.0, std_dev=0.0,
            min_score=0.0, max_score=0.0,
            p25=0.0, p75=0.0, p90=0.0, p95=0.0,
        )

    sorted_vals = sorted(flat)
    n = len(sorted_vals)
    mean_val = statistics.mean(sorted_vals)
    median_val = statistics.median(sorted_vals)
    std_val = statistics.pstdev(sorted_vals) if n > 1 else 0.0

    return SimilarityStatistics(
        count=n,
        mean=round(mean_val, 6),
        median=round(median_val, 6),
        std_dev=round(std_val, 6),
        min_score=round(sorted_vals[0], 6),
        max_score=round(sorted_vals[-1], 6),
        p25=round(_percentile(sorted_vals, 25), 6),
        p75=round(_percentile(sorted_vals, 75), 6),
        p90=round(_percentile(sorted_vals, 90), 6),
        p95=round(_percentile(sorted_vals, 95), 6),
    )


# ---------------------------------------------------------------------------
# Severity distribution
# ---------------------------------------------------------------------------


def compute_severity_distribution(
    similarity_df: pd.DataFrame,
    thresholds: Optional[SimilarityThresholds] = None,
) -> SeverityDistribution:
    """Count documents pairs by severity band.

    Args:
        similarity_df: Square N×N similarity matrix.
        thresholds: Optional custom thresholds; defaults to module-level.

    Returns:
        SeverityDistribution with counts and percentages.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    total_pairs = 0
    low_cnt = 0
    medium_cnt = 0
    high_cnt = 0
    flagged_cnt = 0

    n = similarity_df.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            score = _safe_float(similarity_df.iloc[i, j])
            total_pairs += 1
            if score >= thr.plagiarism:
                flagged_cnt += 1
                sev = severity_from_score(score, thr)
                if sev == HIGH_SEVERITY:
                    high_cnt += 1
                elif sev == MEDIUM_SEVERITY:
                    medium_cnt += 1
                else:
                    low_cnt += 1
            else:
                low_cnt += 1

    def _pct(cnt: int) -> float:
        return round(cnt / total_pairs * 100, 2) if total_pairs > 0 else 0.0

    return SeverityDistribution(
        total=total_pairs,
        low_count=low_cnt,
        low_pct=_pct(low_cnt),
        medium_count=medium_cnt,
        medium_pct=_pct(medium_cnt),
        high_count=high_cnt,
        high_pct=_pct(high_cnt),
        flagged_count=flagged_cnt,
        flagged_pct=_pct(flagged_cnt),
    )


# ---------------------------------------------------------------------------
# Document risk ranking
# ---------------------------------------------------------------------------


def _risk_level_for(max_sim: float, flagged_count: int) -> str:
    """Map a document's worst similarity score to a risk label."""
    if max_sim >= 0.90 or flagged_count >= 5:
        return "Critical"
    if max_sim >= 0.75 or flagged_count >= 3:
        return "High"
    if max_sim >= 0.59 or flagged_count >= 1:
        return "Medium"
    return "Low"


def rank_document_risk(
    similarity_df: pd.DataFrame,
    thresholds: Optional[SimilarityThresholds] = None,
) -> List[DocumentRiskProfile]:
    """Produce a risk profile for every document in the similarity matrix.

    Profiles are sorted by ``max_similarity`` descending (highest risk first).

    Args:
        similarity_df: Square N×N similarity DataFrame.
        thresholds: Optional custom thresholds.

    Returns:
        Sorted list of DocumentRiskProfile objects.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    doc_names = similarity_df.columns.tolist()
    profiles: List[DocumentRiskProfile] = []

    for idx, name in enumerate(doc_names):
        row_vals: List[float] = []
        flagged_count = 0

        for j in range(len(doc_names)):
            if j == idx:
                continue
            score = _safe_float(similarity_df.iloc[idx, j])
            row_vals.append(score)
            if score >= thr.plagiarism:
                flagged_count += 1

        max_sim = max(row_vals) if row_vals else 0.0
        mean_sim = statistics.mean(row_vals) if row_vals else 0.0
        risk = _risk_level_for(max_sim, flagged_count)

        profiles.append(
            DocumentRiskProfile(
                document_name=name,
                max_similarity=round(max_sim, 6),
                mean_similarity=round(mean_sim, 6),
                flagged_pair_count=flagged_count,
                risk_level=risk,
            )
        )

    profiles.sort(key=lambda p: (-p.max_similarity, -p.flagged_pair_count))
    return profiles


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------


def summarize_clusters(
    clusters: Dict[int, List[str]],
    similarity_df: pd.DataFrame,
    thresholds: Optional[SimilarityThresholds] = None,
) -> List[ClusterSummary]:
    """Summarize plagiarism clusters with internal similarity metrics.

    Args:
        clusters: Mapping from cluster_id to list of document names.
        similarity_df: The N×N similarity matrix.
        thresholds: Optional thresholds for severity assignment.

    Returns:
        List of ClusterSummary objects sorted by severity then member count.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    summaries: List[ClusterSummary] = []

    for cid, members in clusters.items():
        if len(members) < 2:
            continue

        pairwise_scores: List[float] = []
        max_score = 0.0

        doc_names = similarity_df.columns.tolist()
        name_to_idx = {n: i for i, n in enumerate(doc_names)}

        for i, a in enumerate(members):
            for b in members[i + 1:]:
                idx_a = name_to_idx.get(a)
                idx_b = name_to_idx.get(b)
                if idx_a is not None and idx_b is not None:
                    s = _safe_float(similarity_df.iloc[idx_a, idx_b])
                    pairwise_scores.append(s)
                    max_score = max(max_score, s)

        avg_internal = statistics.mean(pairwise_scores) if pairwise_scores else 0.0

        severity = severity_from_score(max_score, thr)

        summaries.append(
            ClusterSummary(
                cluster_id=cid,
                member_count=len(members),
                documents=sorted(members),
                avg_internal_similarity=round(avg_internal, 6),
                max_internal_similarity=round(max_score, 6),
                severity=severity,
            )
        )

    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    summaries.sort(key=lambda s: (severity_order.get(s.severity, 9), -s.member_count))
    return summaries


# ---------------------------------------------------------------------------
# Flagged-pair formatting
# ---------------------------------------------------------------------------


def format_flagged_pairs(
    flags: List[Dict[str, Any]],
    thresholds: Optional[SimilarityThresholds] = None,
) -> List[FlaggedPairRecord]:
    """Normalize raw flagged-pair dicts into typed FlaggedPairRecord objects.

    Args:
        flags: List of dicts from ``flag_plagiarism()``.
        thresholds: Optional thresholds for severity.

    Returns:
        List of FlaggedPairRecord sorted by similarity descending.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    records: List[FlaggedPairRecord] = []

    for flag in flags:
        doc_a = str(flag.get("doc_a", flag.get("document_a", "")))
        doc_b = str(flag.get("doc_b", flag.get("document_b", "")))
        score = _safe_float(flag.get("similarity", flag.get("similarity_score", 0.0)))
        sev = flag.get("severity", severity_from_score(score, thr))
        thr_at_flag = _safe_float(flag.get("threshold_at_flag", thr.plagiarism))

        records.append(
            FlaggedPairRecord(
                doc_a=doc_a,
                doc_b=doc_b,
                similarity=round(score, 6),
                severity=str(sev),
                threshold_at_flag=round(thr_at_flag, 6),
                semantic_score=(
                    round(_safe_float(flag["semantic_score"]), 6)
                    if flag.get("semantic_score") is not None
                    else None
                ),
                lexical_score=(
                    round(_safe_float(flag["lexical_score"]), 6)
                    if flag.get("lexical_score") is not None
                    else None
                ),
                semantic_contribution=(
                    round(
                        _safe_float(
                            flag["semantic_contribution"]
                        ),
                        6,
                    )
                    if flag.get("semantic_contribution") is not None
                    else None
                ),
                lexical_contribution=(
                    round(
                        _safe_float(
                            flag["lexical_contribution"]
                        ),
                        6,
                    )
                    if flag.get("lexical_contribution") is not None
                    else None
                ),
                hybrid_weight=(
                    round(_safe_float(flag["alpha"]), 6)
                    if flag.get("alpha") is not None
                    else None
                ),
                threshold_margin=(
                    round(
                        _safe_float(flag["threshold_margin"]),
                        6,
                    )
                    if flag.get("threshold_margin") is not None
                    else None
                ),
            )
        )
    records.sort(key=lambda r: -r.similarity)
    return records


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_analytics_report(
    similarity_df: pd.DataFrame,
    flagged_pairs: Optional[List[Dict[str, Any]]] = None,
    clusters: Optional[Dict[int, List[str]]] = None,
    thresholds: Optional[SimilarityThresholds] = None,
    metadata: Optional[Dict[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> AnalyticsReport:
    """Construct the complete analytics report from pipeline outputs.

    This is the main entry point for report generation.

    Args:
        similarity_df: Square N×N similarity DataFrame.
        flagged_pairs: List of flagged dicts from ``flag_plagiarism()``.
        clusters: Mapping from cluster_id to member document names.
        thresholds: Optional custom similarity thresholds.
        metadata: Arbitrary metadata (model name, upload session, etc.).
        generated_at: ISO-8601 timestamp string; auto-generated if omitted.

    Returns:
        A fully populated AnalyticsReport.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    ts = generated_at or datetime.now(timezone.utc).isoformat()
    n_docs = similarity_df.shape[0]
    n_pairs = n_docs * (n_docs - 1) // 2

    stats = compute_similarity_statistics(similarity_df)
    sev_dist = compute_severity_distribution(similarity_df, thr)
    risk_profiles = rank_document_risk(similarity_df, thr)
    formatted_flags = format_flagged_pairs(flagged_pairs or [], thr)

    cluster_summaries: List[ClusterSummary] = []
    if clusters:
        cluster_summaries = summarize_clusters(clusters, similarity_df, thr)

    return AnalyticsReport(
        generated_at=ts,
        threshold=thr.plagiarism,
        total_documents=n_docs,
        total_evaluated_pairs=n_pairs,
        similarity_stats=stats,
        severity_distribution=sev_dist,
        document_risk_profiles=risk_profiles,
        flagged_pairs=formatted_flags,
        cluster_summaries=cluster_summaries,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_report_json(
    report: AnalyticsReport,
    indent: int = 2,
) -> str:
    """Serialize the analytics report to a JSON string.

    Args:
        report: The AnalyticsReport to serialize.
        indent: JSON indentation level.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps(report.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _write_csv_string(
    headers: List[str],
    rows: List[List[Any]],
) -> str:
    """Write CSV content to an in-memory string buffer."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def export_report_csv_flags(report: AnalyticsReport) -> str:
    """Export flagged pairs as a CSV string.

    Returns:
        CSV-formatted string with columns:
        doc_a, doc_b, similarity, severity, threshold_at_flag.
    """
    headers = ["doc_a", "doc_b", "similarity", "severity", "threshold_at_flag"]
    rows = [
        [f.doc_a, f.doc_b, f.similarity, f.severity, f.threshold_at_flag]
        for f in report.flagged_pairs
    ]
    return _write_csv_string(headers, rows)


def export_report_csv_documents(report: AnalyticsReport) -> str:
    """Export document risk profiles as a CSV string.

    Returns:
        CSV-formatted string with columns:
        document_name, max_similarity, mean_similarity, flagged_pair_count, risk_level.
    """
    headers = [
        "document_name",
        "max_similarity",
        "mean_similarity",
        "flagged_pair_count",
        "risk_level",
    ]
    rows = [
        [
            p.document_name,
            p.max_similarity,
            p.mean_similarity,
            p.flagged_pair_count,
            p.risk_level,
        ]
        for p in report.document_risk_profiles
    ]
    return _write_csv_string(headers, rows)


def export_report_csv_clusters(report: AnalyticsReport) -> str:
    """Export cluster summaries as a CSV string.

    Returns:
        CSV-formatted string with columns:
        cluster_id, member_count, avg_internal_similarity,
        max_internal_similarity, severity, documents.
    """
    headers = [
        "cluster_id",
        "member_count",
        "avg_internal_similarity",
        "max_internal_similarity",
        "severity",
        "documents",
    ]
    rows = []
    for c in report.cluster_summaries:
        rows.append([
            c.cluster_id,
            c.member_count,
            c.avg_internal_similarity,
            c.max_internal_similarity,
            c.severity,
            "; ".join(c.documents),
        ])
    return _write_csv_string(headers, rows)


# ---------------------------------------------------------------------------
# File generation helper
# ---------------------------------------------------------------------------


def generate_report_files(
    similarity_df: pd.DataFrame,
    flagged_pairs: Optional[List[Dict[str, Any]]] = None,
    clusters: Optional[Dict[int, List[str]]] = None,
    thresholds: Optional[SimilarityThresholds] = None,
    output_dir: str = "reports",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build the analytics report and write all output files to disk.

    Creates the following files inside ``output_dir``:
      - analytics_report.json
      - flagged_pairs.csv
      - document_risk_profiles.csv
      - cluster_summaries.csv

    Args:
        similarity_df: Square N×N similarity DataFrame.
        flagged_pairs: Flagged-pair dicts.
        clusters: Cluster membership mapping.
        thresholds: Optional custom thresholds.
        output_dir: Destination directory.
        metadata: Arbitrary metadata.

    Returns:
        Dict mapping logical name to the file path that was written.
    """
    report = build_analytics_report(
        similarity_df,
        flagged_pairs=flagged_pairs,
        clusters=clusters,
        thresholds=thresholds,
        metadata=metadata,
    )

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"plagiarism_analytics_{timestamp}"

    paths: Dict[str, str] = {}

    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(export_report_json(report))
    paths["json"] = json_path
    logger.info("Wrote JSON report to %s", json_path)

    flags_csv_path = os.path.join(output_dir, f"{prefix}_flagged_pairs.csv")
    with open(flags_csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(export_report_csv_flags(report))
    paths["flags_csv"] = flags_csv_path

    docs_csv_path = os.path.join(output_dir, f"{prefix}_document_risk.csv")
    with open(docs_csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(export_report_csv_documents(report))
    paths["documents_csv"] = docs_csv_path

    clusters_csv_path = os.path.join(output_dir, f"{prefix}_clusters.csv")
    with open(clusters_csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(export_report_csv_clusters(report))
    paths["clusters_csv"] = clusters_csv_path

    return paths


# ---------------------------------------------------------------------------
# Convenience: quick summary string
# ---------------------------------------------------------------------------


def format_summary_text(report: AnalyticsReport) -> str:
    """Return a human-readable one-paragraph summary of the report.

    Useful for logging or embedding in notification emails.

    Args:
        report: The populated AnalyticsReport.

    Returns:
        A concise plain-text summary.
    """
    s = report.similarity_stats
    d = report.severity_distribution
    return (
        f"Plagiarism Analytics Report ({report.total_documents} docs, "
        f"{report.total_evaluated_pairs} pairs): "
        f"mean similarity {s.mean:.1%}, median {s.median:.1%}, "
        f"max {s.max_score:.1%}. "
        f"Flagged {d.flagged_count} pairs ({d.flagged_pct:.1f}%): "
        f"{d.high_count} high, {d.medium_count} medium severity. "
        f"{len(report.cluster_summaries)} collusion cluster(s) detected."
    )


# ---------------------------------------------------------------------------
# Incremental / session-based report accumulator
# ---------------------------------------------------------------------------


class ReportAccumulator:
    """Accumulate similarity results across multiple upload sessions.

    Useful for dashboards that track plagiarism trends over time.
    Each ``add_session()`` call records one analysis run; ``build_trend()``
    then returns time-series data suitable for charting.
    """

    def __init__(self) -> None:
        self._sessions: List[Dict[str, Any]] = []

    def add_session(
        self,
        report: AnalyticsReport,
        session_label: Optional[str] = None,
    ) -> None:
        """Record a single analysis session.

        Args:
            report: The analytics report from one upload batch.
            session_label: Optional human-readable label for this session.
        """
        label = session_label or report.generated_at
        self._sessions.append({
            "label": label,
            "generated_at": report.generated_at,
            "total_documents": report.total_documents,
            "total_pairs": report.total_evaluated_pairs,
            "mean_similarity": report.similarity_stats.mean,
            "median_similarity": report.similarity_stats.median,
            "max_similarity": report.similarity_stats.max_score,
            "flagged_count": report.severity_distribution.flagged_count,
            "high_severity": report.severity_distribution.high_count,
            "medium_severity": report.severity_distribution.medium_count,
            "cluster_count": len(report.cluster_summaries),
        })

    @property
    def session_count(self) -> int:
        """Return the number of accumulated sessions."""
        return len(self._sessions)

    def build_trend(self) -> Dict[str, Any]:
        """Build a trend summary across all recorded sessions.

        Returns:
            Dict with ``sessions`` list and ``aggregate`` statistics.
        """
        if not self._sessions:
            return {"sessions": [], "aggregate": {}}

        means = [s["mean_similarity"] for s in self._sessions]
        flags = [s["flagged_count"] for s in self._sessions]
        highs = [s["high_severity"] for s in self._sessions]
        doc_counts = [s["total_documents"] for s in self._sessions]

        aggregate = {
            "total_sessions": len(self._sessions),
            "total_documents_scanned": sum(doc_counts),
            "avg_mean_similarity": round(statistics.mean(means), 6) if means else 0.0,
            "total_flagged_pairs": sum(flags),
            "total_high_severity": sum(highs),
            "trend_direction": (
                "increasing" if len(means) >= 2 and means[-1] > means[0]
                else "decreasing" if len(means) >= 2 and means[-1] < means[0]
                else "stable"
            ),
        }

        return {"sessions": list(self._sessions), "aggregate": aggregate}

    def export_trend_json(self) -> str:
        """Serialize the trend data to JSON."""
        return json.dumps(self.build_trend(), indent=2, default=str)

    def reset(self) -> None:
        """Clear all accumulated sessions."""
        self._sessions.clear()
