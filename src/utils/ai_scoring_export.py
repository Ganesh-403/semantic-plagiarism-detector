"""
AI Scoring Export Utilities.

Provides export functionality for AI scoring results in multiple formats.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class AIScoringExporter:
    """Export AI scoring results."""

    def __init__(self, output_dir: str = "exports/ai_scoring"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, scores: list[dict], filename: str = "ai_scores.json") -> str:
        """Export scores to JSON."""
        path = os.path.join(self.output_dir, filename)
        export_data = {"exported_at": datetime.now().isoformat(), "scores": scores}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported JSON: {path}")
        return path

    def export_csv(self, scores: list[dict], filename: str = "ai_scores.csv") -> str:
        """Export scores to CSV."""
        if not scores:
            return ""
        path = os.path.join(self.output_dir, filename)
        rows = []
        for s in scores:
            rows.append(
                {
                    "doc_a": s.get("doc_a", ""),
                    "doc_b": s.get("doc_b", ""),
                    "overall_score": s.get("overall_score", 0),
                    "severity": s.get("severity", ""),
                    "fingerprint_match": s.get("fingerprint_match", False),
                }
            )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Exported CSV: {path}")
        return path

    def export_summary_report(
        self, scores: List[Dict], filename: str = "scoring_summary.txt"
    ) -> str:
        """Export human-readable summary report."""
        path = os.path.join(self.output_dir, filename)
        total = len(scores)
        avg_score = (
            sum(s.get("overall_score", 0) for s in scores) / total if total else 0
        )
        critical = sum(1 for s in scores if s.get("severity") == "critical")
        high = sum(1 for s in scores if s.get("severity") == "high")

        lines = [
            "=" * 60,
            "AI PLAGIARISM SCORING SUMMARY",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Pairs Scored: {total}",
            f"Average Score: {avg_score:.1%}",
            f"Critical Matches: {critical}",
            f"High Severity: {high}",
            "",
            "TOP MATCHES:",
        ]
        for i, s in enumerate(scores[:10], 1):
            lines.append(
                f"  #{i} {s.get('doc_a')} ↔ {s.get('doc_b')} — {s.get('overall_score', 0):.1%} ({s.get('severity')})"
            )
        lines.extend(["", "=" * 60])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported summary: {path}")
        return path

    def export_all(self, scores: list[dict]) -> dict[str, str]:
        """Export all formats."""
        exports = {}
        exports["json"] = self.export_json(scores)
        exports["csv"] = self.export_csv(scores)
        exports["summary"] = self.export_summary_report(scores)
        return exports
