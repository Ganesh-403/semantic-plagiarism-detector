"""
Plagiarism Report Generator Engine.

Generates comprehensive plagiarism detection reports with
visualizations, statistics, and actionable insights.
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Available report formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"


class ReportType(Enum):
    """Types of plagiarism reports."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    EXECUTIVE = "executive"
    BATCH = "batch"


@dataclass
class ReportSection:
    """A section in the plagiarism report."""

    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    charts: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "data": self.data,
            "charts": self.charts,
            "severity": self.severity,
        }


@dataclass
class PlagiarismReport:
    """Complete plagiarism detection report."""

    report_id: str
    report_type: ReportType
    title: str
    generated_at: str
    summary: Dict[str, Any]
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    recommendations: List[str]
    raw_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
            "recommendations": self.recommendations,
        }


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    include_visualizations: bool = True
    include_raw_data: bool = False
    include_recommendations: bool = True
    max_matches_displayed: int = 50
    severity_threshold: float = 0.5
    company_name: str = "Plagiarism Detection System"
    logo_path: Optional[str] = None
    custom_footer: str = ""


class ReportGenerator:
    """
    Generates comprehensive plagiarism detection reports.

    Creates detailed reports with sections for summary, matches,
    statistics, and recommendations.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._template_cache: Dict[str, str] = {}

    def generate_report(
        self,
        detection_results: Dict[str, Any],
        report_type: ReportType = ReportType.DETAILED,
        title: Optional[str] = None,
    ) -> PlagiarismReport:
        """
        Generate a complete plagiarism report.

        Args:
            detection_results: Results from plagiarism detection
            report_type: Type of report to generate
            title: Optional custom title

        Returns:
            PlagiarismReport instance
        """
        report_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{report_type.value}".encode()
        ).hexdigest()[:12]
        timestamp = datetime.now().isoformat()

        summary = self._generate_summary(detection_results)
        sections = self._generate_sections(detection_results, report_type)
        recommendations = (
            self._generate_recommendations(detection_results)
            if self.config.include_recommendations
            else []
        )

        return PlagiarismReport(
            report_id=report_id,
            report_type=report_type,
            title=title or f"Plagiarism Detection Report — {report_type.value.title()}",
            generated_at=timestamp,
            summary=summary,
            sections=sections,
            metadata={
                "config": self.config.__dict__,
                "detection_engine": "semantic-plagiarism-detector",
            },
            recommendations=recommendations,
            raw_data=detection_results if self.config.include_raw_data else {},
        )

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary statistics."""
        matches = results.get("matches", results.get("flagged", []))
        total_docs = results.get("total_documents", results.get("n_docs", 0))
        total_pairs = results.get(
            "total_pairs", total_docs * (total_docs - 1) // 2 if total_docs > 1 else 0
        )

        severities = {"critical": 0, "high": 0, "moderate": 0, "low": 0, "clean": 0}
        scores = []
        for match in matches:
            score = match.get("similarity", match.get("overall_score", 0))
            scores.append(score)
            if score >= 0.9:
                severities["critical"] += 1
            elif score >= 0.75:
                severities["high"] += 1
            elif score >= 0.5:
                severities["moderate"] += 1
            elif score >= 0.3:
                severities["low"] += 1
            else:
                severities["clean"] += 1

        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0

        return {
            "total_documents": total_docs,
            "total_pairs": total_pairs,
            "total_matches": len(matches),
            "average_similarity": round(avg_score, 4),
            "max_similarity": round(max_score, 4),
            "severity_distribution": severities,
            "plagiarism_rate": round(len(matches) / total_pairs * 100, 2)
            if total_pairs > 0
            else 0,
            "high_severity_count": severities["critical"] + severities["high"],
        }

    def _generate_sections(
        self, results: Dict[str, Any], report_type: ReportType
    ) -> List[ReportSection]:
        """Generate report sections based on type."""
        sections = []
        sections.append(self._create_overview_section(results))
        if report_type in (ReportType.DETAILED, ReportType.BATCH):
            sections.append(self._create_matches_section(results))
        if report_type == ReportType.DETAILED:
            sections.append(self._create_statistics_section(results))
            sections.append(self._create_document_breakdown_section(results))
        if report_type == ReportType.EXECUTIVE:
            sections.append(self._create_executive_insights_section(results))
        return sections

    def _create_overview_section(self, results: Dict[str, Any]) -> ReportSection:
        """Create overview section."""
        summary = self._generate_summary(results)
        content = f"""
# Report Overview

**Total Documents Analyzed:** {summary["total_documents"]}
**Total Document Pairs:** {summary["total_pairs"]}
**Flagged Matches:** {summary["total_matches"]}
**Average Similarity:** {summary["average_similarity"]:.1%}
**Maximum Similarity:** {summary["max_similarity"]:.1%}
**Plagiarism Rate:** {summary["plagiarism_rate"]:.1f}%

## Severity Breakdown
- 🔴 **Critical (≥90%):** {summary["severity_distribution"]["critical"]}
- 🟠 **High (75-89%):** {summary["severity_distribution"]["high"]}
- 🟡 **Moderate (50-74%):** {summary["severity_distribution"]["moderate"]}
- 🟢 **Low (30-49%):** {summary["severity_distribution"]["low"]}
"""
        return ReportSection(
            title="Overview", content=content, data=summary, severity="info"
        )

    def _create_matches_section(self, results: Dict[str, Any]) -> ReportSection:
        """Create matches section."""
        matches = results.get("matches", results.get("flagged", []))
        sorted_matches = sorted(
            matches,
            key=lambda m: m.get("similarity", m.get("overall_score", 0)),
            reverse=True,
        )
        displayed = sorted_matches[: self.config.max_matches_displayed]

        content = "# Detected Matches\n\n"
        for i, match in enumerate(displayed, 1):
            score = match.get("similarity", match.get("overall_score", 0))
            doc_a = match.get("doc_a", match.get("source_doc", "Unknown"))
            doc_b = match.get("doc_b", match.get("target_doc", "Unknown"))
            severity = (
                "🔴"
                if score >= 0.9
                else "🟠"
                if score >= 0.75
                else "🟡"
                if score >= 0.5
                else "🟢"
            )
            content += f"{severity} **#{i}** {doc_a} ↔ {doc_b} — **{score:.1%}**\n"

        if len(matches) > self.config.max_matches_displayed:
            content += f"\n*...and {len(matches) - self.config.max_matches_displayed} more matches.*"

        return ReportSection(
            title="Detected Matches",
            content=content,
            data={"matches": displayed},
            severity="warning",
        )

    def _create_statistics_section(self, results: Dict[str, Any]) -> ReportSection:
        """Create statistics section."""
        matches = results.get("matches", results.get("flagged", []))
        scores = [m.get("similarity", m.get("overall_score", 0)) for m in matches]

        import numpy as np

        stats = {}
        if scores:
            scores_arr = np.array(scores)
            stats = {
                "mean": round(float(np.mean(scores_arr)), 4),
                "median": round(float(np.median(scores_arr)), 4),
                "std": round(float(np.std(scores_arr)), 4),
                "min": round(float(np.min(scores_arr)), 4),
                "max": round(float(np.max(scores_arr)), 4),
                "percentile_90": round(float(np.percentile(scores_arr, 90)), 4),
                "percentile_95": round(float(np.percentile(scores_arr, 95)), 4),
            }

        content = "# Statistical Analysis\n\n"
        if stats:
            for key, val in stats.items():
                content += (
                    f"- **{key.replace('_', ' ').title()}:** {val:.1%}\n"
                    if isinstance(val, float)
                    else f"- **{key}:** {val}\n"
                )
        else:
            content += "No matches to analyze.\n"

        return ReportSection(
            title="Statistics", content=content, data=stats, severity="info"
        )

    def _create_document_breakdown_section(
        self, results: Dict[str, Any]
    ) -> ReportSection:
        """Create document breakdown section."""
        matches = results.get("matches", results.get("flagged", []))
        doc_scores: Dict[str, List[float]] = {}
        for match in matches:
            doc_a = match.get("doc_a", match.get("source_doc", ""))
            doc_b = match.get("doc_b", match.get("target_doc", ""))
            score = match.get("similarity", match.get("overall_score", 0))
            if doc_a not in doc_scores:
                doc_scores[doc_a] = []
            doc_scores[doc_a].append(score)
            if doc_b not in doc_scores:
                doc_scores[doc_b] = []
            doc_scores[doc_b].append(score)

        content = "# Document Breakdown\n\n"
        ranked = sorted(
            doc_scores.items(), key=lambda x: max(x[1]) if x[1] else 0, reverse=True
        )
        for doc, scores in ranked:
            avg = sum(scores) / len(scores) if scores else 0
            max_s = max(scores) if scores else 0
            content += (
                f"- **{doc}:** {len(scores)} matches, avg {avg:.1%}, max {max_s:.1%}\n"
            )

        return ReportSection(
            title="Document Breakdown",
            content=content,
            data=doc_scores,
            severity="info",
        )

    def _create_executive_insights_section(
        self, results: Dict[str, Any]
    ) -> ReportSection:
        """Create executive insights section."""
        summary = self._generate_summary(results)
        plagiarism_rate = summary["plagiarism_rate"]
        high_count = summary["high_severity_count"]

        if plagiarism_rate > 50:
            insight = "⚠️ **HIGH RISK:** Over half of document pairs show significant similarity. Immediate investigation recommended."
        elif plagiarism_rate > 20:
            insight = "🟠 **MODERATE RISK:** A notable portion of documents show concerning similarity levels. Review recommended."
        elif plagiarism_rate > 5:
            insight = "🟡 **LOW RISK:** Some similarity detected but within acceptable range for most academic contexts."
        else:
            insight = "✅ **MINIMAL RISK:** Very low plagiarism indicators detected. Documents appear largely original."

        content = f"# Executive Insights\n\n{insight}\n\n"
        content += f"**Key Metrics:**\n- Plagiarism Rate: {plagiarism_rate:.1f}%\n- High Severity Matches: {high_count}\n"
        return ReportSection(
            title="Executive Insights",
            content=content,
            data=summary,
            severity="critical" if high_count > 0 else "info",
        )

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        summary = self._generate_summary(results)
        recs = []
        if summary["high_severity_count"] > 0:
            recs.append("🔴 Immediate review required for high-severity matches.")
        if summary["plagiarism_rate"] > 30:
            recs.append("🟠 Consider implementing stricter plagiarism policies.")
        if summary["total_documents"] < 5:
            recs.append("📊 Upload more documents for more reliable analysis.")
        if not recs:
            recs.append("✅ No immediate action required. Continue regular monitoring.")
        return recs

    def export_report(
        self, report: PlagiarismReport, format: ReportFormat, output_path: str
    ) -> str:
        """Export report to specified format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if format == ReportFormat.JSON:
            return self._export_json(report, output_path)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report, output_path)
        elif format == ReportFormat.TEXT:
            return self._export_text(report, output_path)
        elif format == ReportFormat.HTML:
            return self._export_html(report, output_path)
        return output_path

    def _export_json(self, report: PlagiarismReport, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str, ensure_ascii=False)
        return path

    def _export_markdown(self, report: PlagiarismReport, path: str) -> str:
        content = f"# {report.title}\n\nGenerated: {report.generated_at}\n\n"
        for section in report.sections:
            content += f"\n{section.content}\n"
        if report.recommendations:
            content += "\n## Recommendations\n\n"
            for rec in report.recommendations:
                content += f"- {rec}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _export_text(self, report: PlagiarismReport, path: str) -> str:
        content = f"{report.title}\n{'=' * 50}\nGenerated: {report.generated_at}\n\n"
        for section in report.sections:
            clean = section.content.replace("#", "").strip()
            content += f"\n{clean}\n\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _export_html(self, report: PlagiarismReport, path: str) -> str:
        html = f"""<!DOCTYPE html>
<html><head><title>{report.title}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}}
.section{{margin:20px 0;padding:15px;border:1px solid #ddd;border-radius:8px;}}
h1{{color:#1a1a2e;}}h2{{color:#16213e;}}</style></head>
<body><h1>{report.title}</h1><p>Generated: {report.generated_at}</p>"""
        for section in report.sections:
            html += f'<div class="section"><h2>{section.title}</h2><pre>{section.content}</pre></div>'
        html += "<h2>Recommendations</h2><ul>"
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul></body></html>"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path
