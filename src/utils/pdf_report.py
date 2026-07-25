"""
pdf_report.py
-------------
Generates professional PDF plagiarism reports using ReportLab.
Provides side-by-side comparison of suspicious paragraph pairs with visual similarity indicators.
"""

from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Tuple

from reportlab.pdfgen import canvas
from src.core.app_config import get_pdf_footer_text
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False


def get_similarity_color(score: float) -> HexColor:
    """
    Returns a color based on similarity score.
    - High (≥0.90): Red
    - Medium (≥0.75): Orange
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


def compress_pdf_buffer(pdf_buffer: BytesIO) -> BytesIO:
    """
    Compresses a ReportLab generated PDF in-memory buffer using PyMuPDF (fitz)
    or PyPDF2/pypdf as a fallback.
    """
    try:
        # Save original position
        original_pos = pdf_buffer.tell()
        pdf_buffer.seek(0)
        pdf_bytes = pdf_buffer.getvalue()

        # 1. Try PyMuPDF (fitz) which is very powerful for garbage collection and stream compression
        if _HAS_FITZ:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                # garbage=4 performs maximum cleanup including duplicate merging
                compressed_bytes = doc.tobytes(garbage=4, deflate=True)
                doc.close()
                return BytesIO(compressed_bytes)
            except Exception:
                pass

        # Fallback to pypdf if PyMuPDF fails or is unavailable
        try:
            from pypdf import PdfReader, PdfWriter

            reader = PdfReader(BytesIO(pdf_bytes))
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            for page in writer.pages:
                page.compress_content_streams()
            out_buf = BytesIO()
            writer.write(out_buf)
            out_buf.seek(0)
            return out_buf
        except ImportError:
            try:
                from PyPDF2 import PdfReader, PdfWriter

                reader = PdfReader(BytesIO(pdf_bytes))
                writer = PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                for page in writer.pages:
                    page.compress_content_streams()
                out_buf = BytesIO()
                writer.write(out_buf)
                out_buf.seek(0)
                return out_buf
            except ImportError:
                pass

        # If all compression attempts fail, return the original buffer
        pdf_buffer.seek(original_pos)
        return pdf_buffer
    except Exception:
        # Absolute safety fallback
        try:
            pdf_buffer.seek(0)
        except Exception:
            pass
        return pdf_buffer


class NumberedCanvas(canvas.Canvas):
    """
    Canvas that renders dynamic page numbers in the format:
    'Page X of Y'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        # Save the final page before calculating total pages
        self._saved_page_states.append(dict(self.__dict__))

        total_pages = len(self._saved_page_states)

        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(total_pages)
            super().showPage()

        super().save()

    def draw_page_number(self, total_pages):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)

        self.drawRightString(
            A4[0] - 72,
            15,
            f"Page {self._pageNumber} of {total_pages}",
        )


def generate_plagiarism_report(
    doc_a: str,
    doc_b: str,
    overall_similarity: float,
    threshold: float,
    top_pairs: List[Tuple[str, str, float]],
    report_title: str = "Plagiarism Detection Report",
    logo_image: Optional[bytes] = None,
    brand_color: Optional[str] = None,
    dark_mode: Optional[bool] = None,
) -> BytesIO:
    """
    Generates a professional PDF plagiarism report for a document pair.

    Args:
        doc_a: Name of the first document
        doc_b: Name of the second document
        overall_similarity: Overall similarity score between documents (0-1)
        threshold: Plagiarism threshold used for detection
        top_pairs: List of (chunk_a, chunk_b, similarity) tuples for top matches
        report_title: Title for the PDF report
        logo_image: Optional raw bytes of a PNG/JPG logo for the PDF header
        brand_color: Optional hex color string (e.g. "#1e3a8a") for headings
        dark_mode: Optional boolean to enable dark mode themed report

    Returns:
        BytesIO buffer containing the generated PDF
    """
    if dark_mode is None:
        try:
            import streamlit as st

            dark_mode = st.session_state.get("theme", "Light") == "Dark"
        except Exception:
            dark_mode = False

    default_brand = "#2dd4bf" if dark_mode else "#1e3a8a"
    brand_hex = brand_color or default_brand
    brand_clr = HexColor(brand_hex)

    # Determine top margin to leave room for logo header
    logo_height = 0
    if logo_image:
        try:
            reader = ImageReader(BytesIO(logo_image))
            iw, ih = reader.getSize()
            logo_display_w = 1.5 * inch
            logo_display_h = logo_display_w * ih / iw
            logo_height = logo_display_h + 0.25 * inch
        except Exception:
            logo_height = 0

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72 + logo_height,
        bottomMargin=18,
    )

    # Get custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=brand_clr,
        spaceAfter=30,
        alignment=TA_CENTER,
        keepWithNext=True,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=brand_clr,
        spaceAfter=12,
        spaceBefore=20,
        keepWithNext=True,
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=HexColor("#FFFFFF") if dark_mode else HexColor("#31333f"),
    )

    # ── Header / footer callback for logo ──
    def _draw_header(canvas_obj, _doc):
        canvas_obj.saveState()
        if dark_mode:
            canvas_obj.setFillColor(HexColor("#0F172A"))
            canvas_obj.rect(
                0,
                0,
                _doc.pagesize[0],
                _doc.pagesize[1],
                fill=True,
                stroke=False,
            )
        if logo_image:
            try:
                reader = ImageReader(BytesIO(logo_image))
                iw, ih = reader.getSize()
                logo_display_w = 1.5 * inch
                logo_display_h = logo_display_w * ih / iw
                x = _doc.leftMargin
                y = _doc.pagesize[1] - 36 - logo_display_h
                canvas_obj.drawImage(
                    reader,
                    x,
                    y,
                    width=logo_display_w,
                    height=logo_display_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass

        footer_text = get_pdf_footer_text()
        if footer_text:
            canvas_obj.setFont("Helvetica", 9)
            if dark_mode:
                canvas_obj.setFillColor(HexColor("#94a3b8"))
            else:
                canvas_obj.setFillColor(HexColor("#6b7280"))
            canvas_obj.drawCentredString(_doc.pagesize[0] / 2.0, 0.5 * inch, footer_text)

        canvas_obj.restoreState()

    # Build story (PDF content)
    story = []

    # Title
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Report metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", normal_style))
    story.append(Spacer(1, 0.1 * inch))

    # Document comparison header
    story.append(Paragraph("Document Comparison", heading_style))

    # Document details table
    doc_data = [
        ["Document A", doc_a],
        ["Document B", doc_b],
        ["Overall Similarity", f"{overall_similarity:.1%}"],
        ["Detection Threshold", f"{threshold:.1%}"],
    ]

    doc_table = Table(doc_data, colWidths=[2 * inch, 4 * inch], hAlign=TA_LEFT)
    if dark_mode:
        table_style_cmds = [
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#FFFFFF")),
            ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]
    else:
        table_style_cmds = [
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
    doc_table.setStyle(TableStyle(table_style_cmds))
    story.append(doc_table)
    story.append(Spacer(1, 0.3 * inch))

    # Visual similarity bar
    sim_color = get_similarity_color(overall_similarity)
    story.append(Paragraph("Similarity Score Visualization", heading_style))

    # Create similarity bar as a table
    bar_width = overall_similarity * 100
    bar_data = [
        ["", ""],
        ["", ""],
    ]
    bar_table = Table(
        bar_data,
        colWidths=[bar_width / 100 * 5 * inch, (100 - bar_width) / 100 * 5 * inch],
        hAlign=TA_LEFT,
    )
    bar_bg_empty = HexColor("#374151") if dark_mode else HexColor("#e5e7eb")
    bar_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), sim_color),
                ("BACKGROUND", (1, 0), (1, -1), bar_bg_empty),
                ("HEIGHT", (0, 0), (-1, -1), 20),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(bar_table)
    story.append(Paragraph(f"{overall_similarity:.1%}", normal_style))
    story.append(Spacer(1, 0.3 * inch))

    # Top suspicious paragraph pairs
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
                    fontName="Helvetica-Bold",
                    fontSize=11,
                    leading=14,
                    textColor=HexColor("#FFFFFF") if dark_mode else HexColor("#1f2937"),
                    spaceAfter=8,
                    spaceBefore=15,
                    keepWithNext=True,
                ),
            )
            story.append(pair_header)

            # Side-by-side comparison
            wrapped_a = wrap_text(chunk_a, max_chars=500)
            wrapped_b = wrap_text(chunk_b, max_chars=500)

            pair_data = [
                [f"<b>From {doc_a}:</b>", f"<b>From {doc_b}:</b>"],
                [wrapped_a, wrapped_b],
            ]

            pair_table = Table(
                pair_data, colWidths=[2.5 * inch, 2.5 * inch], hAlign=TA_LEFT
            )
            if dark_mode:
                pair_table_cmds = [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E293B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#FFFFFF")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            else:
                pair_table_cmds = [
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
            pair_table.setStyle(TableStyle(pair_table_cmds))
            story.append(pair_table)
            story.append(Spacer(1, 0.15 * inch))

            # Add page break if we're in the middle of a long report
            if rank == 3 and len(top_pairs) > 3:
                story.append(PageBreak())
    else:
        story.append(
            Paragraph(
                "No suspicious paragraph pairs found above threshold.",
                normal_style,
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
            f"Threshold used: {threshold:.1%}. Pairs with similarity below this threshold are not shown.",
            normal_style,
        )
    )

    # Build PDF
    doc.build(
        story,
        onFirstPage=_draw_header,
        onLaterPages=_draw_header,
        canvasmaker=NumberedCanvas,
    )
    return compress_pdf_buffer(buffer)


def highlight_pdf_matches(
    pdf_source: str | bytes,
    matching_chunks: List[str],
    highlight_color: Tuple[float, float, float] = (1.0, 0.85, 0.0),  # Yellow
) -> bytes:
    """
    Opens an original PDF, searches for matching plagiarized text chunks,
    applies yellow highlight annotations on exact coordinate boxes,
    and returns the modified PDF as bytes.

    Args:
        pdf_source: Path to the PDF file (str) or raw bytes (bytes)
        matching_chunks: List of text chunk strings to search and highlight
        highlight_color: RGB tuple normalized between 0.0 and 1.0

    Returns:
        bytes: Binary PDF data with highlighted matches
    """
    if not _HAS_FITZ:
        print("[pdf_report] Warning: PyMuPDF is unavailable, skipping PDF highlights.")
        if isinstance(pdf_source, bytes):
            return pdf_source
        with open(pdf_source, "rb") as f:
            return f.read()

    if isinstance(pdf_source, bytes):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        doc = fitz.open(pdf_source)

    for page in doc:
        for chunk in matching_chunks:
            chunk_clean = chunk.strip()
            # Skip very short or empty chunks to prevent accidental full-page highlights
            if len(chunk_clean) < 3:
                continue

            # Search for coordinate rectangles of the text on the page
            quad_matches = page.search_for(chunk_clean)
            for rect in quad_matches:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=highlight_color)
                annot.update()

    # Save highlighted PDF to byte stream
    output_buffer = doc.tobytes()
    doc.close()
    return output_buffer
