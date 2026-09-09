"""
Main report generator for the Report Generation Module
Orchestrates the creation of reports in multiple formats.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import tempfile
from copy import deepcopy
from dataclasses import replace
import numpy as np

from src.models.report import (
    Report,
    ReportConfig,
    ReportRequest,
    ReportResponse,
    ReportStatus,
    ReportFormat,
)
from src.reports.html_generator import HTMLGenerator
from src.reports.pdf_generator import PDFGenerator
from src.reports.csv_exporter import CSVExporter
from src.reports.visualizations import ReportVisualizer


class ReportGenerator:
    """
    Main report generator that orchestrates the creation of reports.
    Supports HTML, PDF, CSV, and JSON formats.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.html_generator = HTMLGenerator()
        self.pdf_generator = PDFGenerator()
        self.csv_exporter = CSVExporter()
        self.visualizer = ReportVisualizer()

        self._reports: Dict[str, Report] = {}
        self._generation_stats: Dict[str, Any] = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "average_time_ms": 0,
        }

    def generate_report(self, request: ReportRequest) -> ReportResponse:
        """
        Generate a report based on the request.

        Args:
            request: Report request

        Returns:
            ReportResponse
        """
        start_time = time.perf_counter()
        report = None
        try:
            report = self._create_report_from_request(request)
            report.status = ReportStatus.GENERATING
            report.update_progress(10)
            data = self._build_report_data(request)
            report.statistics = data["statistics"]
            report.document_names = data["document_names"]
            report.similarity_matrix = data["similarity_matrix"]
            report.matches = data["matches"]
            report.update_progress(40)

            config = replace(request.config)
            if not request.include_details:
                config.include_matches = False
                config.include_heatmap = False
            visualizations = (
                self._generate_visualizations(data, config)
                if request.format in (ReportFormat.HTML, ReportFormat.PDF)
                else {}
            )
            content = self._generate_content(data, visualizations, config)
            report.update_progress(80)
            file_path, file_size = self._export_report(report, content, request.format)
            report.mark_completed(file_path, file_size)
            self._generation_stats["successful"] += 1
            return ReportResponse(
                success=True,
                message="Report generated successfully",
                report_id=report.id,
                file_path=file_path,
                file_size=file_size,
                format=request.format.value,
                record_count=len(content["matches"]),
                report=report,
            )
        except Exception as exc:
            if report is not None:
                report.mark_failed(str(exc))
            self._generation_stats["failed"] += 1
            return ReportResponse(
                success=False,
                message=f"Report generation failed: {exc}",
                report_id=report.id if report is not None else None,
                report=report,
                error=str(exc),
            )
        finally:
            stats = self._generation_stats
            previous_count = stats["total_generated"]
            stats["total_generated"] += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            stats["average_time_ms"] = (
                stats["average_time_ms"] * previous_count + elapsed_ms
            ) / stats["total_generated"]

    def _create_report_from_request(self, request: ReportRequest) -> Report:
        """Create a report object from request."""
        title = (
            request.title
            or f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        report = Report(
            title=title,
            description=request.description or "Plagiarism detection analysis report",
            report_type=request.report_type,
            format=request.format,
            metadata={
                "analysis_id": request.analysis_id,
                "document_ids": list(request.document_ids),
                "include_details": request.include_details,
                "created_at": datetime.now().isoformat(),
                "config": request.config.to_dict() if request.config else None,
            },
        )

        self._reports[report.id] = report
        return report

    def _build_report_data(self, request: ReportRequest) -> Dict[str, Any]:
        """Validate caller results and derive statistics from unique document pairs."""
        data = deepcopy(request.analysis_data)
        if not isinstance(data, dict) or not {
            "document_names",
            "similarity_matrix",
        }.issubset(data):
            raise ValueError(
                "analysis_data must include document_names and similarity_matrix"
            )
        names = data.get("document_names", [])
        if not isinstance(names, list) or any(
            not isinstance(n, str) or not n for n in names
        ):
            raise ValueError("document_names must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("document_names must be unique")
        matrix = np.asarray(data.get("similarity_matrix", []), dtype=float)
        if not names and matrix.size == 0:
            matrix = matrix.reshape(0, 0)
        if matrix.shape != (len(names), len(names)):
            raise ValueError(
                "similarity_matrix must be square and match document_names"
            )
        if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError(
                "similarity_matrix scores must be finite and between zero and one"
            )
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-8):
            raise ValueError("similarity_matrix must be symmetric")

        matches = data.get("matches")
        if matches is None:
            matches = [
                {
                    "source_document": names[i],
                    "target_document": names[j],
                    "hybrid_score": float(matrix[i, j]),
                }
                for i in range(len(names))
                for j in range(i + 1, len(names))
                if matrix[i, j] >= 0.3
            ]
        if not isinstance(matches, list):
            raise ValueError("matches must be a list")
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError("each match must be an object")
            source, target = match.get("source_document"), match.get("target_document")
            if source not in names or target not in names or source == target:
                raise ValueError("matches must refer to two different known documents")
            score = match.get("hybrid_score", match.get("score"))
            if score is None:
                raise ValueError("each match must include a similarity score")
            for key in ("hybrid_score", "score", "lexical_score", "semantic_score"):
                if key in match:
                    value = float(match[key])
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(
                            "match scores must be finite and between zero and one"
                        )
                    match[key] = value
            score = float(score)
            match["hybrid_score"] = score
            match["severity"] = (
                "high"
                if score >= 0.8
                else "medium"
                if score >= 0.5
                else "low"
                if score >= 0.3
                else "none"
            )
        pair_scores = matrix[np.triu_indices(len(names), k=1)]
        stats = {
            "total_documents": len(names),
            "total_comparisons": len(pair_scores),
            "total_matches": len(matches),
        }
        for name, operation in (
            ("average_similarity", np.mean),
            ("median_similarity", np.median),
            ("max_similarity", np.max),
            ("min_similarity", np.min),
            ("std_similarity", np.std),
        ):
            stats[name] = float(operation(pair_scores)) if pair_scores.size else 0.0
        for severity in ("high", "medium", "low", "none"):
            stats[f"{severity}_severity_count"] = sum(
                m["severity"] == severity for m in matches
            )
        return {
            "document_names": names,
            "similarity_matrix": matrix.tolist(),
            "matches": matches,
            "statistics": stats,
        }

    def _generate_visualizations(
        self, data: Dict[str, Any], config: ReportConfig
    ) -> Dict[str, str]:
        """Generate visualizations for the report."""
        visualizations = {}

        if config.include_heatmap and data.get("similarity_matrix"):
            visualizations["heatmap"] = self.visualizer.create_heatmap(
                data["similarity_matrix"],
                data["document_names"],
                title="Document Similarity Matrix",
            )

        if config.include_matches and config.max_matches and data.get("matches"):
            scores = [
                m.get("hybrid_score", m.get("score", 0))
                for m in data["matches"][: config.max_matches]
            ]
            labels = [
                f"{m.get('source_document', '')} → {m.get('target_document', '')}"
                for m in data["matches"][: config.max_matches]
            ]

            visualizations["chart"] = self.visualizer.create_similarity_chart(
                scores,
                labels,
                title="Similarity Scores",
                threshold=config.highlight_threshold,
            )

            visualizations["distribution"] = self.visualizer.create_distribution_chart(
                scores, title="Score Distribution"
            )

            visualizations["severity"] = self.visualizer.create_severity_distribution(
                data["matches"][: config.max_matches], title="Severity Distribution"
            )

        if config.include_summary_stats and data.get("statistics"):
            visualizations["summary"] = self.visualizer.create_summary_dashboard(
                data["statistics"], data.get("matches", []), title="Analysis Summary"
            )

        return visualizations

    def _generate_content(
        self, data: Dict[str, Any], visualizations: Dict[str, str], config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate report content."""
        content = {
            "sections": [],
            "summary": data.get("statistics", {})
            if config.include_summary_stats
            else {},
            "visualizations": visualizations,
            "document_names": data.get("document_names", []),
            "matches": data.get("matches", [])[: config.max_matches]
            if config.include_matches
            else [],
            "similarity_matrix": data.get("similarity_matrix", [])
            if config.include_heatmap
            else [],
        }

        # Add summary section
        if config.include_summary_stats:
            content["sections"].append(
                {
                    "title": "Executive Summary",
                    "content": self._generate_summary_content(data),
                    "data": data.get("statistics", {}),
                }
            )

        # Add heatmap section
        if config.include_heatmap and "heatmap" in visualizations:
            content["sections"].append(
                {
                    "title": "Similarity Heatmap",
                    "content": "This heatmap shows the pairwise similarity scores between all documents.",
                    "visualization": "heatmap",
                }
            )

        # Add matches section
        if config.include_matches and config.max_matches and data.get("matches"):
            content["sections"].append(
                {
                    "title": "Detected Matches",
                    "content": f"Found {len(data['matches'])} matches across documents.",
                    "matches": data["matches"][: config.max_matches],
                }
            )

        # Add statistics section
        if config.include_summary_stats:
            content["sections"].append(
                {
                    "title": "Detailed Statistics",
                    "content": self._generate_statistics_table(data),
                    "data": data.get("statistics", {}),
                }
            )

        return content

    def _generate_summary_content(self, data: Dict[str, Any]) -> str:
        """Generate summary content."""
        stats = data.get("statistics", {})

        return f"""
        <div class="summary-content">
            <p>This report analyzes <strong>{stats.get("total_documents", 0)}</strong> documents with 
            <strong>{stats.get("total_comparisons", 0)}</strong> pairwise comparisons.</p>
            
            <p>The average similarity score is <strong>{stats.get("average_similarity", 0):.2%}</strong>, 
            with a maximum of <strong>{stats.get("max_similarity", 0):.2%}</strong>.</p>
            
            <p>Found <strong>{stats.get("high_severity_count", 0)}</strong> high-severity matches 
            and <strong>{stats.get("medium_severity_count", 0)}</strong> medium-severity matches.</p>
            
            <div class="alert alert-info">
                <strong>💡 Key Insight:</strong> 
                {self._generate_insight(stats)}
            </div>
        </div>
        """

    def _generate_insight(self, stats: Dict[str, Any]) -> str:
        """Generate key insight from statistics."""
        high = stats.get("high_severity_count", 0)
        total = stats.get("total_matches", 0)

        if high > 0:
            if total > 0 and high / total > 0.5:
                return "More than half of the matches are high severity. Immediate review recommended."
            else:
                return f"{high} high-severity matches detected. These require priority review."
        elif stats.get("average_similarity", 0) > 0.3:
            return "Moderate similarity detected. Review matches to ensure proper attribution."
        else:
            return "Low overall similarity detected. No urgent action required."

    def _generate_statistics_table(self, data: Dict[str, Any]) -> str:
        """Generate statistics table."""
        stats = data.get("statistics", {})

        rows = [
            ("📄 Total Documents", str(stats.get("total_documents", 0))),
            ("🔄 Total Comparisons", str(stats.get("total_comparisons", 0))),
            ("🎯 Total Matches", str(stats.get("total_matches", 0))),
            ("📈 Average Similarity", f"{stats.get('average_similarity', 0):.2%}"),
            ("📈 Median Similarity", f"{stats.get('median_similarity', 0):.2%}"),
            ("📈 Max Similarity", f"{stats.get('max_similarity', 0):.2%}"),
            ("📉 Min Similarity", f"{stats.get('min_similarity', 0):.2%}"),
            ("📊 Std Deviation", f"{stats.get('std_similarity', 0):.2%}"),
            ("🔴 High Severity (≥80%)", str(stats.get("high_severity_count", 0))),
            ("🟡 Medium Severity (50-80%)", str(stats.get("medium_severity_count", 0))),
            ("🟢 Low Severity (30-50%)", str(stats.get("low_severity_count", 0))),
            ("⚪ Very Low (<30%)", str(stats.get("none_severity_count", 0))),
        ]

        html = '<table class="stats-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        for label, value in rows:
            html += f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"
        html += "</tbody></table>"

        return html

    def _export_report(
        self, report: Report, content: Dict[str, Any], format: ReportFormat
    ) -> tuple:
        """Render a completed snapshot and atomically publish its bytes."""
        format = ReportFormat(format)
        snapshot = replace(
            report,
            status=ReportStatus.COMPLETED,
            progress=100,
            matches=content["matches"],
            statistics=content["summary"],
            similarity_matrix=content["similarity_matrix"],
        )
        if format == ReportFormat.HTML:
            payload = self.html_generator.generate(snapshot, content).encode("utf-8")
        elif format == ReportFormat.PDF:
            payload = self.pdf_generator.generate(snapshot, content)
        elif format == ReportFormat.CSV:
            payload = self.csv_exporter.export(snapshot, content).encode("utf-8")
        else:
            data = snapshot.to_dict()
            # Artifact bytes do not contain their own size or a host-local path.
            data.pop("file_path")
            data.pop("file_size")
            payload = json.dumps(data, indent=2, allow_nan=False).encode("utf-8")
        # Report IDs can be edited by callers of get_report; validate the output path.
        file_path = (self.output_dir / f"{report.id}.{format.value}").resolve()
        if file_path.parent != self.output_dir:
            raise ValueError("report path must remain inside the output directory")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.output_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(payload)
            os.replace(temporary_path, file_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return str(file_path), len(payload)

    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def list_reports(self, limit: int = 50, offset: int = 0) -> List[Report]:
        """List all generated reports."""
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be nonnegative")
        reports = list(self._reports.values())
        reports.sort(key=lambda r: r.generated_at, reverse=True)
        return reports[offset : offset + limit]

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its file."""
        report = self._reports.get(report_id)
        if not report:
            return False

        if report.file_path:
            file_path = Path(report.file_path).resolve()
            if file_path.parent != self.output_dir:
                raise ValueError("report path must remain inside the output directory")
            file_path.unlink(missing_ok=True)

        del self._reports[report_id]
        return True

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get report generation statistics."""
        return {
            **self._generation_stats,
            "reports_in_memory": len(self._reports),
            "success_rate": (
                self._generation_stats["successful"]
                / self._generation_stats["total_generated"]
                * 100
                if self._generation_stats["total_generated"] > 0
                else 0
            ),
        }

    def cleanup_old_reports(self, days: int = 30) -> int:
        """Clean up reports older than specified days."""
        if days < 0:
            raise ValueError("days must be nonnegative")
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        to_delete = []

        for report_id, report in self._reports.items():
            if report.generated_at.timestamp() < cutoff:
                to_delete.append(report_id)

        for report_id in to_delete:
            self.delete_report(report_id)

        return len(to_delete)
