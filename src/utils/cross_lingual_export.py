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
Cross-Lingual Plagiarism Export Utilities.

Provides export functionality for cross-lingual detection results
in multiple formats with detailed reporting.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CrossLingualExporter:
    """Export cross-lingual detection results."""

    def __init__(self, output_dir: str = "exports/cross_lingual"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(
        self, result: dict[str, Any], filename: str = "cross_lingual_results.json"
    ) -> str:
        """Export results to JSON."""
        path = os.path.join(self.output_dir, filename)
        export_data = {"exported_at": datetime.now().isoformat(), "result": result}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported JSON: {path}")
        return path

    def export_matches_csv(
        self, matches: list[dict], filename: str = "cross_lingual_matches.csv"
    ) -> str:
        """Export matches to CSV."""
        if not matches:
            return ""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=matches[0].keys())
            writer.writeheader()
            writer.writerows(matches)
        logger.info(f"Exported CSV: {path}")
        return path

    def export_language_report(
        self, result: dict[str, Any], filename: str = "language_report.txt"
    ) -> str:
        """Export human-readable language analysis report."""
        path = os.path.join(self.output_dir, filename)
        summary = result.get("summary", {})
        lang_dist = result.get("language_distribution", {})
        matches = result.get("matches", [])

        lines = [
            "=" * 60,
            "CROSS-LINGUAL PLAGIARISM DETECTION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERVIEW:",
            f"  Total Documents: {summary.get('total_documents', 0)}",
            f"  Languages Detected: {summary.get('languages_detected', 0)}",
            f"  Cross-Lingual Matches: {summary.get('cross_lingual_matches', 0)}",
            f"  Same-Language Matches: {summary.get('same_language_matches', 0)}",
            f"  High Severity Matches: {summary.get('high_severity', 0)}",
            "",
            "LANGUAGE DISTRIBUTION:",
        ]
        for lang, count in lang_dist.items():
            lines.append(f"  {lang}: {count} documents")

        lines.extend(["", "TOP MATCHES:"])
        for i, match in enumerate(matches[:10], 1):
            lines.append(
                f"  #{i} {match.get('source_doc')} ({match.get('source_lang')}) ↔ "
                f"{match.get('target_doc')} ({match.get('target_lang')}) — "
                f"{match.get('similarity', 0):.1%}"
            )

        lines.extend(["", "=" * 60, "END OF REPORT", "=" * 60])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported report: {path}")
        return path

    def export_all(self, result: dict[str, Any]) -> dict[str, str]:
        """Export all formats."""
        exports = {}
        exports["json"] = self.export_json(result)
        exports["csv"] = self.export_matches_csv(result.get("matches", []))
        exports["report"] = self.export_language_report(result)
        return exports
