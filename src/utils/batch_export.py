"""
Batch Export Utilities for Plagiarism Detection Reports.

Provides export functionality for batch processing results in
multiple formats (JSON, CSV, PDF summary).
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for export operations."""

    output_dir: str = "exports"
    include_details: bool = True
    include_metadata: bool = True
    max_results: int = 1000
    date_format: str = "%Y-%m-%d %H:%M:%S"


class BatchExporter:
    """
    Export batch processing results to various formats.

    Supports JSON, CSV, and generates summary reports.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def export_json(
        self, results: Dict[str, Any], filename: str = "batch_results.json"
    ) -> str:
        """
        Export results to JSON format.

        Args:
            results: Results dictionary
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = os.path.join(self.config.output_dir, filename)
        export_data = {"exported_at": datetime.now().isoformat(), "results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported JSON to {output_path}")
        return output_path

    def export_csv(
        self, data: List[Dict[str, Any]], filename: str = "batch_results.csv"
    ) -> str:
        """
        Export results to CSV format.

        Args:
            data: List of result dictionaries
            filename: Output filename

        Returns:
            Path to exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""

        output_path = os.path.join(self.config.output_dir, filename)
        headers = data[0].keys()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data[: self.config.max_results])
        logger.info(f"Exported CSV to {output_path}")
        return output_path

    def export_plagiarism_report(
        self,
        flagged_pairs: List[Dict[str, Any]],
        summary: Dict[str, Any],
        filename: str = "plagiarism_report.json",
    ) -> str:
        """
        Export a comprehensive plagiarism report.

        Args:
            flagged_pairs: List of flagged document pairs
            summary: Summary statistics
            filename: Output filename

        Returns:
            Path to exported file
        """
        report = {
            "report_type": "plagiarism_detection",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_documents": summary.get("total_documents", 0),
                "total_pairs": summary.get("total_pairs", 0),
                "flagged_pairs": len(flagged_pairs),
                "high_severity": summary.get("high_severity", 0),
                "avg_similarity": summary.get("avg_similarity", 0),
            },
            "flagged_pairs": flagged_pairs[: self.config.max_results],
            "metadata": summary.get("metadata", {}),
        }

        output_path = os.path.join(self.config.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported plagiarism report to {output_path}")
        return output_path

    def export_batch_summary(
        self, jobs: List[Dict[str, Any]], filename: str = "batch_summary.csv"
    ) -> str:
        """
        Export batch job summary to CSV.

        Args:
            jobs: List of job dictionaries
            filename: Output filename

        Returns:
            Path to exported file
        """
        if not jobs:
            return ""

        summary_data = []
        for job in jobs:
            summary_data.append(
                {
                    "job_id": job.get("job_id", ""),
                    "name": job.get("name", ""),
                    "status": job.get("status", ""),
                    "documents": job.get("total_documents", 0),
                    "flagged": job.get("flagged_pairs", 0),
                    "progress": job.get("progress", 0),
                    "created_at": job.get("created_at", ""),
                    "duration": job.get("duration_seconds", ""),
                }
            )

        return self.export_csv(summary_data, filename)

    def generate_text_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable text summary.

        Args:
            results: Results dictionary

        Returns:
            Formatted text summary
        """
        lines = [
            "=" * 60,
            "BATCH PROCESSING SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime(self.config.date_format)}",
            "",
            "OVERVIEW:",
            f"  Total Jobs: {results.get('total_jobs', 0)}",
            f"  Completed: {results.get('completed', 0)}",
            f"  Failed: {results.get('failed', 0)}",
            f"  Total Documents: {results.get('total_documents', 0)}",
            f"  Total Flagged: {results.get('total_flagged', 0)}",
            "",
            "-" * 60,
            "END OF REPORT",
            "-" * 60,
        ]
        return "\n".join(lines)

    def export_all(
        self,
        results: Dict[str, Any],
        flagged_pairs: List[Dict[str, Any]],
        jobs: List[Dict[str, Any]],
        base_name: str = "batch_export",
    ) -> Dict[str, str]:
        """
        Export all results in multiple formats.

        Args:
            results: Overall results
            flagged_pairs: Flagged pairs
            jobs: Job list
            base_name: Base filename prefix

        Returns:
            Dictionary mapping format to file path
        """
        exports = {}
        exports["json"] = self.export_json(results, f"{base_name}.json")
        exports["report"] = self.export_plagiarism_report(
            flagged_pairs, results, f"{base_name}_report.json"
        )
        if jobs:
            exports["csv"] = self.export_batch_summary(jobs, f"{base_name}_jobs.csv")
        exports["text"] = ""
        summary_text = self.generate_text_summary(results)
        text_path = os.path.join(self.config.output_dir, f"{base_name}_summary.txt")
        with open(text_path, "w") as f:
            f.write(summary_text)
        exports["text"] = text_path
        logger.info(f"Exported all formats: {list(exports.keys())}")
        return exports


class ReportFormatter:
    """Formats batch processing data for display."""

    @staticmethod
    def format_duration(seconds: Optional[float]) -> str:
        """Format duration in human-readable format."""
        if seconds is None:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    @staticmethod
    def format_status_emoji(status: str) -> str:
        """Get emoji for status."""
        return {
            "completed": "✅",
            "processing": "⚡",
            "pending": "⏳",
            "failed": "❌",
            "cancelled": "🚫",
            "paused": "⏸️",
        }.get(status, "❓")

    @staticmethod
    def format_priority(priority: str) -> str:
        """Format priority with color."""
        colors = {"urgent": "🔴", "high": "🟠", "normal": "🟡", "low": "🟢"}
        return f"{colors.get(priority, '⚪')} {priority.title()}"
