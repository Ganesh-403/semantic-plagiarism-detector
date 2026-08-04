#!/usr/bin/env python3
"""
generate_golden_pdf.py
----------------------
Script to generate or refresh golden PDF fixtures for the semantic plagiarism detector.

Usage:
    # Generate a new golden PDF fixture
    python tests/utils/generate_golden_pdf.py output.pdf

    # Refresh an existing golden fixture (with --force)
    python tests/utils/generate_golden_pdf.py output.pdf --force

This script generates a deterministic PDF report that can be used as a golden fixture
for comparing against future generated PDFs. The generated PDF uses consistent input
parameters to ensure deterministic output.
"""

import argparse
import sys
import io
from pathlib import Path

# We'll use reportlab directly to generate the PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors


def get_similarity_color(score: float) -> HexColor:
    """
    Returns a color based on similarity score.
    - High (>=0.90): Red
    - Medium (>=0.75): Orange
    - Low (<0.75): Green
    """
    if score >= 0.90:
        return HexColor("#ff4b4b")
    elif score >= 0.75:
        return HexColor("#ffa500")
    else:
        return HexColor("#21c55d")


def wrap_text(text: str, max_chars: int = 400) -> str:
    """
    Truncates text to max_chars and adds ellipsis if needed.
    Helps prevent text overflow in PDF cells.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def generate_test_pdf(
    output_path: Path,
) -> None:
    """
    Generate a test PDF report with deterministic content.

    This creates a consistent PDF with known content that can serve as a golden fixture.
    """
    brand_hex = "#1e3a8a"
    brand_clr = HexColor(brand_hex)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Get custom styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=brand_clr,
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=brand_clr,
        spaceAfter=12,
        spaceBefore=20,
    )
    normal_style = styles["Normal"]
    normal_style.fontSize = 10
    normal_style.leading = 14

    # Build story (PDF content)
    story = []

    # Title
    story.append(Paragraph("Plagiarism Detection Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Report metadata - use fixed timestamp for determinism
    timestamp = "2026-07-28 18:15:15"
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", normal_style))
    story.append(Spacer(1, 0.1 * inch))

    # Document comparison header
    story.append(Paragraph("Document Comparison", heading_style))

    # Document details table
    doc_data = [
        ["Document A", "student_a.pdf"],
        ["Document B", "student_b.pdf"],
        ["Overall Similarity", "93.4%"],
        ["Detection Threshold", "59.0%"],
    ]

    doc_table = Table(doc_data, colWidths=[2 * inch, 4 * inch], hAlign=TA_LEFT)
    doc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), HexColor("#f3f4f6")),
                ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#374151")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(doc_table)
    story.append(Spacer(1, 0.3 * inch))

    # Visual similarity bar
    sim_color = get_similarity_color(0.934)
    story.append(Paragraph("Similarity Score Visualization", heading_style))

    # Create similarity bar as a table
    bar_width = 0.934 * 100
    bar_data = [
        ["", ""],
        ["", ""],
    ]
    bar_table = Table(
        bar_data,
        colWidths=[bar_width / 100 * 5 * inch, (100 - bar_width) / 100 * 5 * inch],
        hAlign=TA_LEFT,
    )
    bar_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), sim_color),
                ("BACKGROUND", (1, 0), (1, -1), HexColor("#e5e7eb")),
                ("HEIGHT", (0, 0), (-1, -1), 20),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(bar_table)
    story.append(Paragraph("93.4%", normal_style))
    story.append(Spacer(1, 0.3 * inch))

    # Top suspicious paragraph pairs
    top_pairs = [
        (
            "This is the first paragraph from document A that contains some text about the subject being discussed.",
            "This is the first paragraph from document B that contains similar text about the same subject being discussed.",
            0.96,
        ),
        (
            "The second paragraph discusses the methodology used in the research study and includes various statistical analyses.",
            "Methodology section describes the research approach and includes statistical analysis similar to the previous paragraph.",
            0.87,
        ),
        (
            "In the conclusion, the authors summarize their findings and suggest areas for future research.",
            "The authors conclude by summarizing their key findings and identifying potential areas for further investigation.",
            0.79,
        ),
        (
            "The introduction provides background information on the topic and establishes the context for the study.",
            "Introduction section gives background on the topic and sets up the research context.",
            0.72,
        ),
    ]

    if top_pairs:
        story.append(Paragraph("Top Suspicious Paragraph Pairs", heading_style))
        story.append(
            Paragraph(
                f"Showing top {len(top_pairs)} most similar paragraph pairs above threshold.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.1 * inch))

        # Create side-by-side comparison table for each pair
        for rank, (chunk_a, chunk_b, score) in enumerate(top_pairs, 1):
            # Pair header with similarity score
            pair_color = get_similarity_color(score)
            pair_header = Paragraph(
                f"<b>Pair #{rank}</b> — Similarity: <font color='{pair_color}'>{score:.1%}</font>",
                ParagraphStyle(
                    "PairHeader",
                    parent=styles["Heading3"],
                    fontSize=11,
                    textColor=HexColor("#1f2937"),
                    spaceAfter=8,
                    spaceBefore=15,
                ),
            )
            story.append(pair_header)

            # Side-by-side comparison
            wrapped_a = wrap_text(chunk_a, max_chars=500)
            wrapped_b = wrap_text(chunk_b, max_chars=500)

            pair_data = [
                ["<b>From student_a.pdf:</b>", "<b>From student_b.pdf:</b>"],
                [wrapped_a, wrapped_b],
            ]

            pair_table = Table(
                pair_data, colWidths=[2.5 * inch, 2.5 * inch], hAlign=TA_LEFT
            )
            pair_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f9fafb")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#111827")),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(pair_table)
            story.append(Spacer(1, 0.15 * inch))

            # Add page break if we're in the middle of a long report
            if rank == 3 and len(top_pairs) > 3:
                story.append(PageBreak())
    else:
        story.append(
            Paragraph(
                "No suspicious paragraph pairs found above threshold.", normal_style
            )
        )

    # Footer note
    story.append(PageBreak())
    story.append(Paragraph("Report Notes", heading_style))
    story.append(
        Paragraph(
            "This report was generated by the Semantic Plagiarism Detection System. "
            "Similarity scores are computed using transformer embeddings (all-MiniLM-L6-v2) "
            "and cosine similarity. High similarity scores may indicate plagiarism, "
            "but human review is recommended for final determination.",
            normal_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Threshold used: 59.0%. Pairs with similarity below this threshold are not shown.",
            normal_style,
        )
    )

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    # Write to file
    output_path.write_bytes(buffer.getvalue())
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
            response = input(
                f"File '{output_path}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if response != "y":
                print("Aborted.")
                sys.exit(0)
        else:
            print(f"Overwriting existing file: {output_path}")

    # Generate the PDF
    generate_test_pdf(output_path)


if __name__ == "__main__":
    main()
