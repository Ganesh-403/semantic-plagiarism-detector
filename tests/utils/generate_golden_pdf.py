#!/usr/bin/env python3
"""Regenerate the reference PDF using the actual report renderer."""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pdf_report import generate_plagiarism_report


def generate_test_pdf(output_path: Path) -> None:
    with patch("src.utils.pdf_report.datetime") as clock:
        clock.now.return_value = datetime(2025, 6, 15, 12, 0, 0)
        data = generate_plagiarism_report(
            doc_a="student_a.pdf", doc_b="student_b.pdf",
            overall_similarity=0.934, threshold=0.59,
            top_pairs=[("First matching paragraph.", "Second matching paragraph.", 0.96)],
            incident_id="INC-QR-12345",
        )
    output_path.write_bytes(data.getvalue())
    print(f"Generated golden PDF: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate or refresh golden PDF fixtures"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for the golden PDF fixture",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file without confirmation",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Check if file exists
    if output_path.exists():
        if not args.force:
            response = (
                input(f"File '{output_path}' already exists. Overwrite? [y/N]: ")
                .strip()
                .lower()
            )
            if response != "y":
                print("Aborted.")
                sys.exit(0)
        else:
            print(f"Overwriting existing file: {output_path}")

    # Generate the PDF
    generate_test_pdf(output_path)


if __name__ == "__main__":
    main()
