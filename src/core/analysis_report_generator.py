"""
src/core/analysis_report_generator.py
--------------------------------------
Comprehensive plagiarism analysis report generator.

Produces structured JSON and HTML reports from scan results,
including summary statistics, severity breakdowns, per-document
risk assessments, cluster insights, and trend comparisons.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import (
    CRITICAL_SEVERITY,
    DEFAULT_THRESHOLDS,
    HIGH_SEVERITY,
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    SEVERITY_ORDER,
    SimilarityThresholds,
    normalize_score,
    severity_from_score,
)
from src.core.document_cluster_analyzer import (
    ClusterAnalysisResult,
    ClusterRiskLevel,
    DocumentClusterAnalyzer,
    compute_document_risk_scores,
    generate_risk_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PairwiseFinding:
    """A single flagged document pair."""
    doc_a: str
    doc_b: str
    similarity: float
    severity: str
    flagged_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "similarity": round(self.similarity, 6),
            "severity": self.severity,
            "flagged_chunks": self.flagged_chunks,
        }


@dataclass
class ScanSummary:
    """Aggregate statistics for a single scan session."""
    total_documents: int
    total_pairs: int
    flagged_pairs: int
    avg_similarity: float
    max_similarity: float
    median_similarity: float
    severity_distribution: Dict[str, int]
    threshold_used: float
    scan_timestamp: str
    processing_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_pairs": self.total_pairs,
            "flagged_pairs": self.flagged_pairs,
            "flagging_rate": (
                round(self.flagged_pairs / self.total_pairs, 6)
                if self.total_pairs > 0
                else 0.0
            ),
            "avg_similarity": round(self.avg_similarity, 6),
            "max_similarity": round(self.max_similarity, 6),
            "median_similarity": round(self.median_similarity, 6),
            "severity_distribution": self.severity_distribution,
            "threshold_used": self.threshold_used,
            "scan_timestamp": self.scan_timestamp,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
        }


@dataclass
class AnalysisReport:
    """Complete analysis report combining all analysis components."""
    scan_summary: ScanSummary
    findings: List[PairwiseFinding]
    document_risks: Dict[str, Dict[str, Any]]
    risk_summary: Dict[str, Any]
    cluster_result: Optional[ClusterAnalysisResult]
    recommendations: List[str]
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    report_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "report_version": self.report_version,
            "generated_at": self.generated_at,
            "scan_summary": self.scan_summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "document_risks": self.document_risks,
            "risk_summary": self.risk_summary,
            "recommendations": self.recommendations,
        }
        if self.cluster_result is not None:
            result["cluster_analysis"] = self.cluster_result.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines: List[str] = []
        lines.append("# Plagiarism Analysis Report")
        lines.append("")
        lines.append(f"**Generated:** {self.generated_at}")
        lines.append(f"**Report Version:** {self.report_version}")
        lines.append("")

        # Scan summary
        s = self.scan_summary
        lines.append("## Scan Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Documents analyzed | {s.total_documents} |")
        lines.append(f"| Total pairs | {s.total_pairs} |")
        lines.append(f"| Flagged pairs | {s.flagged_pairs} |")
        flag_rate = (
            f"{s.flagged_pairs / s.total_pairs:.1%}" if s.total_pairs > 0 else "N/A"
        )
        lines.append(f"| Flagging rate | {flag_rate} |")
        lines.append(f"| Average similarity | {s.avg_similarity:.4f} |")
        lines.append(f"| Maximum similarity | {s.max_similarity:.4f} |")
        lines.append(f"| Median similarity | {s.median_similarity:.4f} |")
        lines.append(f"| Threshold used | {s.threshold_used} |")
        lines.append("")

        # Severity distribution
        if s.severity_distribution:
            lines.append("### Severity Distribution")
            lines.append("")
            for sev in SEVERITY_ORDER:
                count = s.severity_distribution.get(sev, 0)
                if count > 0:
                    lines.append(f"- **{sev}:** {count}")
            lines.append("")

        # Top findings
        if self.findings:
            lines.append("## Top Findings")
            lines.append("")
            lines.append("| # | Document A | Document B | Similarity | Severity |")
            lines.append("|---|-----------|-----------|-----------|---------|")
            for i, f in enumerate(self.findings[:20], 1):
                lines.append(
                    f"| {i} | {f.doc_a} | {f.doc_b} "
                    f"| {f.similarity:.4f} | {f.severity} |"
                )
            lines.append("")

        # Risk summary
        rs = self.risk_summary
        if rs.get("total_documents", 0) > 0:
            lines.append("## Document Risk Summary")
            lines.append("")
            lines.append(f"- **Total documents:** {rs['total_documents']}")
            lines.append(f"- **Average risk score:** {rs['avg_risk_score']:.4f}")
            lines.append(f"- **Max risk score:** {rs['max_risk_score']:.4f}")
            lines.append(f"- **Flagged documents:** {rs['flagged_count']}")
            if rs.get("critical_documents"):
                lines.append(
                    f"- **Critical:** {', '.join(rs['critical_documents'])}"
                )
            if rs.get("high_risk_documents"):
                lines.append(
                    f"- **High risk:** {', '.join(rs['high_risk_documents'])}"
                )
            lines.append("")

        # Cluster analysis
        if self.cluster_result and self.cluster_result.clusters:
            ca = self.cluster_result
            lines.append("## Cluster Analysis")
            lines.append("")
            lines.append(
                f"- **Method:** {ca.method.value}"
            )
            lines.append(f"- **Clusters found:** {ca.total_clusters}")
            lines.append(
                f"- **Clustered documents:** {ca.clustered_documents} / "
                f"{ca.total_documents}"
            )
            lines.append("")
            for c in ca.clusters:
                risk_icon = {
                    ClusterRiskLevel.CRITICAL: "🔴",
                    ClusterRiskLevel.HIGH: "🟠",
                    ClusterRiskLevel.ELEVATED: "🟡",
                    ClusterRiskLevel.LOW: "🟢",
                }.get(c.risk_level, "⚪")
                lines.append(
                    f"### {risk_icon} Cluster {c.cluster_id} "
                    f"({c.risk_level.value}, {c.size} docs)"
                )
                lines.append(
                    f"- Avg similarity: {c.avg_internal_similarity:.4f}, "
                    f"Max: {c.max_internal_similarity:.4f}"
                )
                lines.append(
                    f"- Documents: {', '.join(sorted(c.documents))}"
                )
                lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class AnalysisReportGenerator:
    """Generates comprehensive plagiarism analysis reports.

    Usage::

        generator = AnalysisReportGenerator()
        report = generator.generate(
            doc_names=["a.pdf", "b.pdf"],
            similarity_matrix=sim_matrix,
            embeddings=emb_matrix,
        )
        print(report.to_json())
    """

    def __init__(
        self,
        thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
        enable_clustering: bool = True,
        max_findings: int = 50,
    ) -> None:
        self.thresholds = thresholds
        self.enable_clustering = enable_clustering
        self.max_findings = max_findings

    def generate(
        self,
        doc_names: List[str],
        similarity_matrix: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        processing_time: float = 0.0,
        chunk_level_data: Optional[Dict[str, Any]] = None,
    ) -> AnalysisReport:
        """Generate a full analysis report.

        Args:
            doc_names: Ordered document names.
            similarity_matrix: NxN cosine similarity matrix.
            embeddings: Optional (N, D) embedding matrix.
            processing_time: Total scan processing time in seconds.
            chunk_level_data: Optional per-pair chunk-level matches.

        Returns:
            Complete AnalysisReport.
        """
        t_start = datetime.now()
        n = len(doc_names)

        # Validate
        if n == 0:
            return self._empty_report(processing_time)

        if similarity_matrix.shape != (n, n):
            raise ValueError(
                f"similarity_matrix shape {similarity_matrix.shape} "
                f"does not match {n} documents"
            )

        # Extract pairwise similarities (upper triangle only)
        pairs = self._extract_pairs(doc_names, similarity_matrix)

        # Compute scan summary
        scan_summary = self._compute_scan_summary(
            doc_names, pairs, processing_time
        )

        # Find flagged pairs
        findings = self._extract_findings(
            pairs, chunk_level_data
        )

        # Per-document risk scores
        doc_risks = compute_document_risk_scores(
            doc_names, similarity_matrix, self.thresholds
        )
        risk_summary = generate_risk_summary(doc_risks)

        # Cluster analysis
        cluster_result = None
        if self.enable_clustering and n >= 3:
            analyzer = DocumentClusterAnalyzer(
                threshold=self.thresholds.plagiarism,
                thresholds=self.thresholds,
            )
            cluster_result = analyzer.analyze(
                doc_names, similarity_matrix, embeddings
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            scan_summary, findings, risk_summary, cluster_result
        )

        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info(
            "Report generated in %.3fs: %d findings from %d documents",
            elapsed,
            len(findings),
            n,
        )

        return AnalysisReport(
            scan_summary=scan_summary,
            findings=findings,
            document_risks=doc_risks,
            risk_summary=risk_summary,
            cluster_result=cluster_result,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pairs(
        self,
        doc_names: List[str],
        similarity_matrix: np.ndarray,
    ) -> List[PairwiseFinding]:
        """Extract all pairwise similarities from the upper triangle."""
        n = len(doc_names)
        pairs: List[PairwiseFinding] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarity_matrix[i, j])
                severity = severity_from_score(sim, self.thresholds)
                pairs.append(
                    PairwiseFinding(
                        doc_a=doc_names[i],
                        doc_b=doc_names[j],
                        similarity=sim,
                        severity=severity,
                    )
                )
        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs

    def _extract_findings(
        self,
        pairs: List[PairwiseFinding],
        chunk_level_data: Optional[Dict[str, Any]] = None,
    ) -> List[PairwiseFinding]:
        """Filter to flagged pairs and attach chunk-level data."""
        flagged = [
            p for p in pairs
            if p.similarity >= self.thresholds.plagiarism
        ]

        if chunk_level_data:
            for finding in flagged:
                key = f"{finding.doc_a}|{finding.doc_b}"
                alt_key = f"{finding.doc_b}|{finding.doc_a}"
                chunks = chunk_level_data.get(key) or chunk_level_data.get(
                    alt_key, []
                )
                finding.flagged_chunks = chunks

        return flagged[: self.max_findings]

    def _compute_scan_summary(
        self,
        doc_names: List[str],
        pairs: List[PairwiseFinding],
        processing_time: float,
    ) -> ScanSummary:
        """Compute aggregate statistics."""
        n = len(doc_names)
        total_pairs = n * (n - 1) // 2

        if not pairs:
            return ScanSummary(
                total_documents=n,
                total_pairs=total_pairs,
                flagged_pairs=0,
                avg_similarity=0.0,
                max_similarity=0.0,
                median_similarity=0.0,
                severity_distribution={},
                threshold_used=self.thresholds.plagiarism,
                scan_timestamp=datetime.now().isoformat(),
                processing_time_seconds=processing_time,
            )

        all_scores = [p.similarity for p in pairs]
        flagged = [
            p for p in pairs
            if p.similarity >= self.thresholds.plagiarism
        ]

        severity_counts = Counter(p.severity for p in flagged)

        return ScanSummary(
            total_documents=n,
            total_pairs=total_pairs,
            flagged_pairs=len(flagged),
            avg_similarity=float(np.mean(all_scores)),
            max_similarity=float(np.max(all_scores)),
            median_similarity=float(np.median(all_scores)),
            severity_distribution=dict(severity_counts),
            threshold_used=self.thresholds.plagiarism,
            scan_timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time,
        )

    def _generate_recommendations(
        self,
        summary: ScanSummary,
        findings: List[PairwiseFinding],
        risk_summary: Dict[str, Any],
        cluster_result: Optional[ClusterAnalysisResult],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs: List[str] = []

        # Overall flagging rate
        if summary.total_pairs > 0:
            rate = summary.flagged_pairs / summary.total_pairs
            if rate > 0.3:
                recs.append(
                    f"⚠️ High flagging rate ({rate:.0%}). Consider "
                    "tightening the plagiarism threshold or reviewing "
                    "assignment diversity."
                )
            elif rate == 0:
                recs.append(
                    "✅ No plagiarism flags detected across all document pairs."
                )

        # Critical findings
        critical = [f for f in findings if f.severity == HIGH_SEVERITY]
        if critical:
            recs.append(
                f"🔴 {len(critical)} pair(s) flagged at HIGH severity "
                f"(similarity ≥ {self.thresholds.high}). "
                "Immediate manual review required."
            )

        # Repeat offenders
        flagged_docs: Counter[str] = Counter()
        for f in findings:
            flagged_docs[f.doc_a] += 1
            flagged_docs[f.doc_b] += 1
        repeat_offenders = [
            doc for doc, count in flagged_docs.most_common(5) if count >= 3
        ]
        if repeat_offenders:
            recs.append(
                f"🟡 Documents flagged in 3+ pairs: "
                f"{', '.join(repeat_offenders)}. "
                "These may be common source materials or template documents."
            )

        # Cluster recommendations
        if cluster_result:
            recs.extend(
                rec.replace(
                    "🔴",
                    "🔴 [Cluster]",
                ).replace(
                    "🟠",
                    "🟠 [Cluster]",
                ).replace(
                    "🟡",
                    "🟡 [Cluster]",
                )
                for rec in cluster_result.recommendations
                if "✅" not in rec
            )

        # Risk summary recommendations
        crit_docs = risk_summary.get("critical_documents", [])
        if len(crit_docs) > 2:
            recs.append(
                f"📋 {len(crit_docs)} documents have critical risk scores. "
                "Consider investigating potential collusion networks."
            )

        if not recs:
            recs.append(
                "✅ Analysis complete. No significant plagiarism patterns detected."
            )

        return recs

    def _empty_report(self, processing_time: float) -> AnalysisReport:
        """Create a minimal report for empty input."""
        summary = ScanSummary(
            total_documents=0,
            total_pairs=0,
            flagged_pairs=0,
            avg_similarity=0.0,
            max_similarity=0.0,
            median_similarity=0.0,
            severity_distribution={},
            threshold_used=self.thresholds.plagiarism,
            scan_timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time,
        )
        return AnalysisReport(
            scan_summary=summary,
            findings=[],
            document_risks={},
            risk_summary=generate_risk_summary({}),
            cluster_result=None,
            recommendations=["ℹ️ No documents to analyze."],
        )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_report_json(
    report: AnalysisReport,
    output_path: str,
    indent: int = 2,
) -> str:
    """Write a report to a JSON file.

    Args:
        report: The analysis report to export.
        output_path: Destination file path.
        indent: JSON indentation level.

    Returns:
        The path that was written.
    """
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.to_json(indent=indent), encoding="utf-8")
    logger.info("Report exported to %s", destination)
    return str(destination)


def export_report_markdown(
    report: AnalysisReport,
    output_path: str,
) -> str:
    """Write a report to a Markdown file.

    Args:
        report: The analysis report to export.
        output_path: Destination file path.

    Returns:
        The path that was written.
    """
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.to_markdown(), encoding="utf-8")
    logger.info("Markdown report exported to %s", destination)
    return str(destination)


def export_report_html(
    report: AnalysisReport,
    output_path: str,
    title: str = "Plagiarism Analysis Report",
) -> str:
    """Write a self-contained HTML report.

    The HTML includes embedded CSS for a clean, print-friendly layout.

    Args:
        report: The analysis report to export.
        output_path: Destination file path.
        title: Page title.

    Returns:
        The path that was written.
    """
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    summary = report.scan_summary
    findings_html = ""
    for i, f in enumerate(report.findings[:30], 1):
        severity_class = f.severity.lower().replace(" ", "-")
        findings_html += (
            f'<tr class="severity-{severity_class}">'
            f"<td>{i}</td>"
            f"<td>{f.doc_a}</td>"
            f"<td>{f.doc_b}</td>"
            f"<td>{f.similarity:.4f}</td>"
            f'<td><span class="badge badge-{severity_class}">{f.severity}</span></td>'
            f"</tr>\n"
        )

    cluster_html = ""
    if report.cluster_result and report.cluster_result.clusters:
        for c in report.cluster_result.clusters:
            cluster_html += (
                f'<div class="cluster-card risk-{c.risk_level.value}">'
                f"<h4>Cluster {c.cluster_id} "
                f'({c.risk_level.value}, {c.size} docs)</h4>'
                f"<p>Avg sim: {c.avg_internal_similarity:.4f}, "
                f"Max: {c.max_internal_similarity:.4f}</p>"
                f"<p>Documents: {', '.join(sorted(c.documents))}</p>"
                f"</div>\n"
            )

    risk_rows = ""
    for doc_name in sorted(report.document_risks.keys()):
        r = report.document_risks[doc_name]
        risk_rows += (
            f"<tr>"
            f"<td>{doc_name}</td>"
            f"<td>{r['max_similarity']:.4f}</td>"
            f"<td>{r['most_similar_document']}</td>"
            f"<td>{r['similar_pair_count']}</td>"
            f'<td><span class="badge badge-{r["severity"].lower()}">'
            f'{r["severity"]}</span></td>'
            f"</tr>\n"
        )

    recs_html = "\n".join(f"<li>{rec}</li>" for rec in report.recommendations)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #1a1a2e;
         background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1e3a8a; padding-bottom: 0.5rem; }}
  h2 {{ color: #1e3a8a; margin-top: 2rem; }}
  h3 {{ color: #374151; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #1e3a8a; color: white; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f3f4f6; }}
  .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
  .badge-high {{ background: #fee2e2; color: #991b1b; }}
  .badge-medium {{ background: #fef3c7; color: #92400e; }}
  .badge-low {{ background: #d1fae5; color: #065f46; }}
  .cluster-card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem;
                   margin: 0.75rem 0; }}
  .risk-critical {{ border-left: 4px solid #dc2626; }}
  .risk-high {{ border-left: 4px solid #ea580c; }}
  .risk-elevated {{ border-left: 4px solid #d97706; }}
  .risk-low {{ border-left: 4px solid #16a34a; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 1rem; margin: 1rem 0; }}
  .stat-card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px;
                padding: 1.25rem; text-align: center; }}
  .stat-card .value {{ font-size: 1.75rem; font-weight: 700; color: #1e3a8a; }}
  .stat-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem; }}
  .recommendations {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
                      padding: 1.25rem; margin: 1rem 0; }}
  .recommendations li {{ margin: 0.5rem 0; }}
  .meta {{ color: #6b7280; font-size: 0.9rem; }}
  @media print {{
    body {{ padding: 1rem; background: white; }}
    .stat-card {{ box-shadow: none; }}
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {report.generated_at} | Report v{report.report_version}</p>

<h2>Scan Summary</h2>
<div class="summary-grid">
  <div class="stat-card"><div class="value">{summary.total_documents}</div><div class="label">Documents</div></div>
  <div class="stat-card"><div class="value">{summary.total_pairs}</div><div class="label">Pairs Analyzed</div></div>
  <div class="stat-card"><div class="value">{summary.flagged_pairs}</div><div class="label">Flagged Pairs</div></div>
  <div class="stat-card"><div class="value">{summary.max_similarity:.4f}</div><div class="label">Max Similarity</div></div>
  <div class="stat-card"><div class="value">{summary.avg_similarity:.4f}</div><div class="label">Avg Similarity</div></div>
</div>

<h2>Flagged Pairs</h2>
<table>
<thead><tr><th>#</th><th>Document A</th><th>Document B</th><th>Similarity</th><th>Severity</th></tr></thead>
<tbody>
{findings_html}
</tbody>
</table>

<h2>Document Risk Scores</h2>
<table>
<thead><tr><th>Document</th><th>Max Similarity</th><th>Most Similar To</th><th>Flagged Pairs</th><th>Severity</th></tr></thead>
<tbody>
{risk_rows}
</tbody>
</table>

<h2>Cluster Analysis</h2>
{cluster_html if cluster_html else "<p>No document clusters detected.</p>"}

<div class="recommendations">
<h2>Recommendations</h2>
<ul>{recs_html}</ul>
</div>

</body>
</html>"""

    destination.write_text(html, encoding="utf-8")
    logger.info("HTML report exported to %s", destination)
    return str(destination)
