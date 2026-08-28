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
Anomaly Detection Export Utilities.

Provides export functionality for anomaly detection results.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class AnomalyExporter:
    """Export anomaly detection results."""

    def __init__(self, output_dir: str = "exports/anomalies"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, result: dict, filename: str = "anomaly_results.json") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"exported_at": datetime.now().isoformat(), "result": result},
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Exported JSON: {path}")
        return path

    def export_csv(self, anomalies: list[dict], filename: str = "anomalies.csv") -> str:
        if not anomalies:
            return ""
        path = os.path.join(self.output_dir, filename)
        rows = [{k: v for k, v in a.items() if k != "evidence"} for a in anomalies]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Exported CSV: {path}")
        return path

    def export_summary(
        self, result: dict, filename: str = "anomaly_summary.txt"
    ) -> str:
        path = os.path.join(self.output_dir, filename)
        summary = result.get("summary", {})
        lines = [
            "=" * 50,
            "ANOMALY DETECTION SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Anomalies: {summary.get('total_anomalies', 0)}",
            f"High Priority: {summary.get('high_priority_count', 0)}",
            f"Documents Analyzed: {summary.get('documents_analyzed', 0)}",
            "",
            "=" * 50,
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
