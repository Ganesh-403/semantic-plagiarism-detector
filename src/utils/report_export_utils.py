"""
Report Export Utilities.

Provides batch export functionality for plagiarism reports
in multiple formats with compression support.
"""

import os
import json
import csv
import zipfile
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReportExportManager:
    """Manages batch export of plagiarism reports."""

    def __init__(self, output_dir: str = "exports/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_batch_json(
        self, reports: List[Dict], filename: str = "batch_reports.json"
    ) -> str:
        """Export multiple reports to a single JSON file."""
        path = os.path.join(self.output_dir, filename)
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "report_count": len(reports),
            "reports": reports,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported batch JSON: {path}")
        return path

    def export_comparison_csv(
        self, comparisons: List[Dict], filename: str = "comparison_results.csv"
    ) -> str:
        """Export comparison results to CSV."""
        if not comparisons:
            return ""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["doc_a", "doc_b", "similarity", "severity", "method"]
            )
            writer.writeheader()
            writer.writerows(comparisons)
        logger.info(f"Exported comparison CSV: {path}")
        return path

    def create_zip_archive(
        self, files: List[str], archive_name: str = "reports_archive.zip"
    ) -> str:
        """Create a ZIP archive of multiple report files."""
        path = os.path.join(self.output_dir, archive_name)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                if os.path.exists(file_path):
                    arcname = os.path.basename(file_path)
                    zf.write(file_path, arcname)
        logger.info(f"Created ZIP archive: {path}")
        return path

    def generate_csv_summary(
        self, results: Dict[str, Any], filename: str = "results_summary.csv"
    ) -> str:
        """Generate a CSV summary of all results."""
        path = os.path.join(self.output_dir, filename)
        rows = []
        matches = results.get("matches", results.get("flagged", []))
        for m in matches:
            rows.append(
                {
                    "Document A": m.get("doc_a", m.get("source_doc", "")),
                    "Document B": m.get("doc_b", m.get("target_doc", "")),
                    "Similarity": f"{m.get('similarity', m.get('overall_score', 0)):.1%}",
                    "Severity": m.get("severity", "unknown"),
                }
            )
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"Generated CSV summary: {path}")
        return path
