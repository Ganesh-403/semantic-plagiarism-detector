import hashlib
import json
from datetime import datetime
from unittest.mock import patch

from src.utils.pdf_report import generate_plagiarism_report

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

hashes = []
for i in range(3):
    with patch("src.utils.pdf_report.datetime") as mock_dt:
        mock_dt.now.return_value = FROZEN_TIME
        mock_dt.strftime = datetime.strftime
        buf = generate_plagiarism_report(**INPUTS)
    h = hashlib.sha256(buf.getvalue()).hexdigest()
    hashes.append(h)
    print(f"Run {i + 1}: {h}", flush=True)

print(f"Deterministic: {len(set(hashes)) == 1}", flush=True)

# Update golden fixture with the latest hash
data = {
    "hash": hashes[-1],
    "inputs": {
        "doc_a": INPUTS["doc_a"],
        "doc_b": INPUTS["doc_b"],
        "overall_similarity": INPUTS["overall_similarity"],
        "threshold": INPUTS["threshold"],
        "top_pairs_count": len(INPUTS["top_pairs"]),
    },
    "generated_at": FROZEN_TIME.isoformat(),
}
with open("tests/fixtures/pdf_report_golden.hash", "w") as f:
    json.dump(data, f, indent=2)
print(f"Golden fixture updated with: {hashes[-1]}", flush=True)
