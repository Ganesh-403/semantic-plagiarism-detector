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

"""Generate the golden PDF fixture for snapshot testing."""

import hashlib
import json
import os
import sys
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.pdf_report import generate_plagiarism_report

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(FIXTURE_DIR, "pdf_report_golden.hash")

FROZEN_TIME = datetime(2025, 6, 15, 12, 0, 0)

INPUTS = {
    "doc_a": "essay_john_doe.pdf",
    "doc_b": "essay_jane_smith.pdf",
    "overall_similarity": 0.873,
    "threshold": 0.60,
    "top_pairs": [
        (
            "The mitochondria is the powerhouse of the cell and plays a crucial role in energy production.",
            "The mitochondria serves as the cell's primary energy generator through ATP synthesis.",
            0.94,
        ),
        (
            "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
            "Plants transform sunlight into chemical energy via the process of photosynthesis.",
            0.91,
        ),
        (
            "DNA replication occurs during the S phase of the cell cycle before mitosis begins.",
            "The cell replicates its DNA in the synthesis phase prior to mitotic division.",
            0.88,
        ),
    ],
}


def generate_golden():
    with patch("src.utils.pdf_report.datetime") as mock_dt:
        mock_dt.now.return_value = FROZEN_TIME
        mock_dt.strftime = datetime.strftime
        buffer = generate_plagiarism_report(**INPUTS)

    pdf_bytes = buffer.getvalue()
    golden_hash = hashlib.sha256(pdf_bytes).hexdigest()

    fixture_data = {
        "hash": golden_hash,
        "inputs": {
            "doc_a": INPUTS["doc_a"],
            "doc_b": INPUTS["doc_b"],
            "overall_similarity": INPUTS["overall_similarity"],
            "threshold": INPUTS["threshold"],
            "top_pairs_count": len(INPUTS["top_pairs"]),
        },
        "generated_at": FROZEN_TIME.isoformat(),
    }

    with open(GOLDEN_PATH, "w") as f:
        json.dump(fixture_data, f, indent=2)

    print(f"Golden fixture generated: {golden_hash}")
    print(f"Saved to: {GOLDEN_PATH}")
    return golden_hash


if __name__ == "__main__":
    generate_golden()
