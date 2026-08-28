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
src/utils/lms_manifest.py
-------------------------
Export engine for LMS (Canvas/Blackboard) import manifests.

Generates CSV or XML manifests that map plagiarism reports to specific
student IDs and assignment IDs, allowing bulk import into LMS gradebooks.
"""

import csv
import io
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def generate_canvas_manifest(records: list[dict[str, Any]], format: str = "csv") -> str:
    """Generate a Canvas-compatible import manifest.

    Args:
        records: List of dictionaries containing:
            - 'student_id': SIS ID or Canvas user ID.
            - 'assignment_id': Canvas assignment ID.
            - 'report_filename': Name of the generated report file.
            - 'similarity_score': Plagiarism score (0-100).
        format: Output format ('csv' or 'xml').

    Returns:
        A string containing the manifest data.
    """
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "student_id",
                "assignment_id",
                "report_filename",
                "similarity_score",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "student_id": record.get("student_id"),
                    "assignment_id": record.get("assignment_id"),
                    "report_filename": record.get("report_filename"),
                    "similarity_score": record.get("similarity_score"),
                }
            )
        return output.getvalue()

    elif format == "xml":
        root = ET.Element("manifest")
        for record in records:
            entry = ET.SubElement(root, "entry")
            ET.SubElement(entry, "student_id").text = str(record.get("student_id"))
            ET.SubElement(entry, "assignment_id").text = str(
                record.get("assignment_id")
            )
            ET.SubElement(entry, "report_filename").text = record.get("report_filename")
            ET.SubElement(entry, "similarity_score").text = str(
                record.get("similarity_score")
            )

        return ET.tostring(root, encoding="unicode")

    else:
        raise ValueError(f"Unsupported format: {format}")
