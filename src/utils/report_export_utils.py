# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Report Export Utilities.

Provides batch export functionality for plagiarism reports
in multiple formats with compression support.
"""

import csv
import json
import logging
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReportExportManager:
    """Manages batch export of plagiarism reports."""

    def __init__(self, output_dir: str = "exports/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_batch_json(
        self, reports: list[dict], filename: str = "batch_reports.json"
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
        self, comparisons: list[dict], filename: str = "comparison_results.csv"
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
        self, files: list[str], archive_name: str = "reports_archive.zip"
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
        self, results: dict[str, Any], filename: str = "results_summary.csv"
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
