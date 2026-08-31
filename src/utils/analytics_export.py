"""
analytics_export.py
-------------------
Multi-format export utilities for plagiarism analytics reports.
Generates CSV, JSON, and HTML outputs from AnalyticsSummary data.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.report_export_analytics import (
    AnalyticsSummary,
    DocumentRiskProfile,
    SeverityBucket,
    TrendPoint,
)

logger = logging.getLogger(__name__)


# ── CSV Export ────────────────────────────────────────────────────────────────


def export_trends_csv(trends: List[TrendPoint]) -> str:
    if not trends:
        return ""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["period", "document_count", "avg_similarity", "max_similarity", "flagged_count", "threshold_used"])
    for t in trends:
        w.writerow([t.timestamp, t.document_count, t.avg_similarity, t.max_similarity, t.flagged_count, t.threshold_used])
    return buf.getvalue()


def export_severity_csv(buckets: List[SeverityBucket]) -> str:
    if not buckets:
        return ""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["severity", "count", "percentage", "avg_score", "min_score", "max_score"])
    for b in buckets:
        w.writerow([b.label, b.count, b.percentage, b.avg_score, b.min_score, b.max_score])
    return buf.getvalue()


def export_risk_profiles_csv(profiles: List[DocumentRiskProfile]) -> str:
    if not profiles:
        return ""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "incident_count", "avg_similarity", "max_similarity", "severity", "last_flagged"])
    for p in profiles:
        w.writerow([p.filename, p.incident_count, p.avg_similarity, p.max_similarity, p.severity, p.last_flagged])
    return buf.getvalue()


def export_summary_csv(summary: AnalyticsSummary) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["=== Overview ==="])
    w.writerow(["metric", "value"])
    for k, v in [("total_scans", summary.total_scans), ("total_incidents", summary.total_incidents),
                  ("total_documents", summary.total_documents), ("avg_similarity_overall", summary.avg_similarity_overall),
                  ("max_similarity_overall", summary.max_similarity_overall), ("flagged_rate", summary.flagged_rate)]:
        w.writerow([k, v])
    w.writerow([])
    w.writerow(["=== Severity Distribution ==="])
    w.writerow(["severity", "count", "percentage", "avg_score"])
    for b in summary.severity_distribution:
        w.writerow([b.label, b.count, b.percentage, b.avg_score])
    w.writerow([])
    w.writerow(["=== Top Risk Documents ==="])
    w.writerow(["filename", "incident_count", "avg_similarity", "max_similarity", "severity"])
    for p in summary.top_risk_documents:
        w.writerow([p.filename, p.incident_count, p.avg_similarity, p.max_similarity, p.severity])
    return buf.getvalue()


# ── JSON Export ───────────────────────────────────────────────────────────────


def export_summary_json(summary: AnalyticsSummary, include_trends: bool = True, indent: int = 2) -> str:
    data = summary.to_dict()
    if not include_trends:
        for k in ("daily_trends", "weekly_trends", "monthly_trends"):
            data.pop(k, None)
    data["_metadata"] = {"generated_at": datetime.now().isoformat(), "format_version": "1.0", "engine": "report_export_analytics"}
    return json.dumps(data, indent=indent, default=str)


def export_threshold_analysis_json(analysis: List[Dict[str, Any]], indent: int = 2) -> str:
    return json.dumps({
        "threshold_analysis": analysis,
        "_metadata": {"generated_at": datetime.now().isoformat(), "total_thresholds": len(analysis)},
    }, indent=indent, default=str)


# ── HTML Export ───────────────────────────────────────────────────────────────


def _badge_cls(label: str) -> str:
    return {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(label, "badge-low")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Plagiarism Analytics Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}}
.container{{max-width:960px;margin:0 auto}}
h1{{color:#f1f5f9;font-size:1.75rem;margin-bottom:8px}}
h2{{color:#94a3b8;font-size:1.1rem;font-weight:500;margin-bottom:24px}}
.section{{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:20px}}
.section h3{{color:#38bdf8;font-size:1rem;margin-bottom:12px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}}
.metric-card{{background:#0f172a;border-radius:8px;padding:16px;text-align:center}}
.metric-value{{font-size:1.75rem;font-weight:700;color:#38bdf8}}
.metric-label{{font-size:.8rem;color:#94a3b8;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th{{text-align:left;padding:10px 12px;color:#94a3b8;border-bottom:1px solid #334155;font-weight:500}}
td{{padding:10px 12px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:rgba(56,189,248,.05)}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}}
.badge-high{{background:#7f1d1d;color:#fca5a5}}.badge-medium{{background:#78350f;color:#fcd34d}}.badge-low{{background:#14532d;color:#86efac}}
.footer{{text-align:center;margin-top:32px;padding-top:16px;border-top:1px solid #1e293b;color:#64748b;font-size:.8rem}}
</style>
</head>
<body>
<div class="container">
<h1>Plagiarism Analytics Report</h1>
<h2>Generated {generated_at}</h2>
<div class="section"><h3>Overview</h3><div class="metrics">
<div class="metric-card"><div class="metric-value">{total_scans}</div><div class="metric-label">Total Scans</div></div>
<div class="metric-card"><div class="metric-value">{total_incidents}</div><div class="metric-label">Flagged Incidents</div></div>
<div class="metric-card"><div class="metric-value">{total_documents}</div><div class="metric-label">Documents</div></div>
<div class="metric-card"><div class="metric-value">{avg_similarity}%</div><div class="metric-label">Avg Similarity</div></div>
<div class="metric-card"><div class="metric-value">{flagged_rate}%</div><div class="metric-label">Flagged Rate</div></div>
</div></div>
<div class="section"><h3>Severity Distribution</h3>
<table><thead><tr><th>Severity</th><th>Count</th><th>Percentage</th><th>Avg Score</th></tr></thead>
<tbody>{severity_rows}</tbody></table></div>
<div class="section"><h3>Top Risk Documents</h3>
<table><thead><tr><th>Document</th><th>Incidents</th><th>Avg Score</th><th>Max Score</th><th>Severity</th></tr></thead>
<tbody>{risk_rows}</tbody></table></div>
<div class="footer">Semantic Plagiarism Detector — Analytics Report Engine v1.0</div>
</div></body></html>"""


def export_summary_html(summary: AnalyticsSummary) -> str:
    sev_rows = "".join(
        f'<tr><td><span class="badge {_badge_cls(b.label)}">{b.label}</span></td>'
        f'<td>{b.count}</td><td>{b.percentage}%</td><td>{b.avg_score}</td></tr>'
        for b in summary.severity_distribution
    )
    risk_rows = "".join(
        f'<tr><td>{p.filename}</td><td>{p.incident_count}</td>'
        f'<td>{p.avg_similarity}</td><td>{p.max_similarity}</td>'
        f'<td><span class="badge {_badge_cls(p.severity)}">{p.severity}</span></td></tr>'
        for p in summary.top_risk_documents
    )
    return _HTML_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_scans=summary.total_scans, total_incidents=summary.total_incidents,
        total_documents=summary.total_documents,
        avg_similarity=round(summary.avg_similarity_overall * 100, 1),
        flagged_rate=round(summary.flagged_rate * 100, 1),
        severity_rows=sev_rows, risk_rows=risk_rows,
    )


# ── Unified dispatchers ──────────────────────────────────────────────────────


def export_analytics(summary: AnalyticsSummary, format: str = "json") -> str:
    fmt = format.lower().strip()
    if fmt == "json":
        return export_summary_json(summary)
    elif fmt == "csv":
        return export_summary_csv(summary)
    elif fmt == "html":
        return export_summary_html(summary)
    raise ValueError(f"Unsupported export format: '{format}'. Supported: json, csv, html.")


def export_trends(trends: List[TrendPoint], format: str = "csv") -> str:
    fmt = format.lower().strip()
    if fmt == "csv":
        return export_trends_csv(trends)
    elif fmt == "json":
        return json.dumps([t.to_dict() for t in trends], indent=2, default=str)
    raise ValueError(f"Unsupported trend export format: '{format}'. Supported: csv, json.")
