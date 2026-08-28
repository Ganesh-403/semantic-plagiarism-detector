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
pdf_report.py
-------------
Generates professional PDF plagiarism reports using ReportLab.
Provides side-by-side comparison of suspicious paragraph pairs with visual similarity indicators.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.core.app_config import get_pdf_footer_text
from src.utils.text_stats import compute_text_stats

try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False


_BRANDING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "branding_config.json"
)

# Resolved once; falls back to Helvetica if no bundled TTF is available.
_PDF_FONT_REGULAR = "Helvetica"
_PDF_FONT_BOLD = "Helvetica-Bold"
_PDF_FONTS_READY = False


def _ensure_pdf_fonts() -> tuple[str, str]:
    """Register a bundled Unicode TTF for ReportLab and return (regular, bold)."""
    global _PDF_FONT_REGULAR, _PDF_FONT_BOLD, _PDF_FONTS_READY
    if _PDF_FONTS_READY:
        return _PDF_FONT_REGULAR, _PDF_FONT_BOLD

    fonts_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
    for filename, font_name in (
        ("DejaVuSans.ttf", "DejaVuSans"),
        ("Roboto-Regular.ttf", "Roboto"),
    ):
        font_path = os.path.join(fonts_dir, filename)
        if not os.path.isfile(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            _PDF_FONT_REGULAR = font_name
            # Only regular faces are bundled; reuse for bold styles.
            _PDF_FONT_BOLD = font_name
            break
        except Exception:
            continue

    _PDF_FONTS_READY = True
    return _PDF_FONT_REGULAR, _PDF_FONT_BOLD


def load_branding_logo() -> bytes | None:
    """
    Reads logo_path from config/branding_config.json and returns the logo
    bytes if the file exists and is a valid image, otherwise returns None.
    """
    try:
        with open(_BRANDING_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        logo_path = cfg.get("logo_path", "").strip()
        if not logo_path:
            return None
        with open(logo_path, "rb") as img_f:
            return img_f.read()
    except Exception:
        return None


def truncate_filename(filename: str, max_len: int = 30) -> str:
    """
    Truncates a filename to max_len characters with an ellipsis if needed,
    preserving its file extension.
    Example: 'final_essay_v2_final_really_final_draft_john_smith.pdf' -> 'final_essay_v2_f...h.pdf'
    """
    if len(filename) <= max_len:
        return filename

    name, ext = os.path.splitext(filename)
    needed_len = max_len - len(ext) - 3

    if needed_len <= 2:
        return filename[: max_len - 3] + "..."

    half = needed_len // 2
    truncated_name = f"{name[:half]}...{name[-(needed_len - half):]}"
    return f"{truncated_name}{ext}"


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


def break_long_urls(text: str) -> str:
    """
    Inserts zero-width spaces (\u200b) or break opportunities after punctuation/slashes
    in long URLs so ReportLab wraps them properly without bleeding off page margins.
    """
    if not text or not isinstance(text, str):
        return text

    def _insert_zwsp(match: re.Match) -> str:
        url = match.group(0)
        # Break after slashes, dots, query parameters, dashes, underscores, and ampersands
        broken_url = re.sub(r"([/\.\?=&_\-#~:])", r"\1\u200b", url)
        return broken_url

    # Regex detecting http(s) URLs or ftp URLs
    url_pattern = re.compile(r"https?://[^\s<>\"'()]+|ftp://[^\s<>\"'()]+")
    return url_pattern.sub(_insert_zwsp, text)


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
        font_regular, _ = _ensure_pdf_fonts()
        self.setFont(font_regular, 9)
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
    top_pairs: list[tuple[str, str, float]],
    doc_a_text: str | None = None,
    doc_b_text: str | None = None,
    report_title: str = "Plagiarism Detection Report",
    logo_image: bytes | None = None,
    brand_color: str | None = None,
    incident_id: str | None = None,
    dark_mode: bool | None = None,
    language: str = "en",
) -> BytesIO:
    from src.i18n.translator import get_text

    brand_hex = brand_color or "#1e3a8a"

    """
    Generates a professional PDF plagiarism report for a document pair.

    Args:
        doc_a: Name of the first document
        doc_b: Name of the second document
        overall_similarity: Overall similarity score between documents (0-1)
        threshold: Plagiarism threshold used for detection
        top_pairs: List of (chunk_a, chunk_b, similarity) tuples for top matches
        doc_a_text: Optional raw text of document A for statistics calculation
        doc_b_text: Optional raw text of document B for statistics calculation
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

    resolved_logo_image = logo_image
    if not resolved_logo_image:
        resolved_logo_image = load_branding_logo()

    logo_height = 0
    if resolved_logo_image:
        try:
            reader = ImageReader(BytesIO(resolved_logo_image))
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
        bottomMargin=40,
    )

    # Get custom styles
    font_regular, font_bold = _ensure_pdf_fonts()

    title_style = ParagraphStyle(
        "CustomTitle",
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor=brand_clr,
        spaceAfter=30,
        alignment=TA_CENTER,
        keepWithNext=True,
        wordWrap="CJK",
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=brand_clr,
        spaceAfter=12,
        spaceBefore=20,
        keepWithNext=True,
        wordWrap="CJK",
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        fontName=font_regular,
        fontSize=10,
        leading=14,
        textColor=HexColor("#FFFFFF") if dark_mode else HexColor("#31333f"),
        wordWrap="CJK",
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
        if resolved_logo_image:
            try:
                reader = ImageReader(BytesIO(resolved_logo_image))
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

        if incident_id:
            try:
                import qrcode

                base_url = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip(
                    "/"
                )
                verify_url = f"{base_url}/verify/{incident_id}"

                qr = qrcode.QRCode(version=1, box_size=4, border=0)
                qr.add_data(verify_url)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")

                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                qr_reader = ImageReader(img_byte_arr)
                qr_w, qr_h = qr_reader.getSize()

                qr_display_w = 1.0 * inch
                qr_display_h = qr_display_w * qr_h / qr_w

                qr_x = _doc.pagesize[0] - _doc.rightMargin - qr_display_w
                qr_y = _doc.pagesize[1] - 36 - qr_display_h

                canvas_obj.drawImage(
                    qr_reader,
                    qr_x,
                    qr_y,
                    width=qr_display_w,
                    height=qr_display_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass
        footer_text = get_pdf_footer_text()
        if footer_text:
            canvas_obj.setFont(font_regular, 9)
            if dark_mode:
                canvas_obj.setFillColor(HexColor("#94a3b8"))
            else:
                canvas_obj.setFillColor(HexColor("#6b7280"))
            canvas_obj.drawCentredString(
                _doc.pagesize[0] / 2.0, 0.5 * inch, footer_text
            )

        canvas_obj.restoreState()

        if footer_text:
            canvas_obj.saveState()
            canvas_obj.setFont(font_regular, 9)
            if dark_mode:
                canvas_obj.setFillColor(HexColor("#9CA3AF"))
            else:
                canvas_obj.setFillColor(HexColor("#6B7280"))
            canvas_obj.drawCentredString(_doc.pagesize[0] / 2.0, 20, footer_text)
            canvas_obj.restoreState()

    # Build story (PDF content)

    story = []

    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", normal_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Document Comparison", heading_style))

    doc_data = [
        [get_text("pdf_document_name", language), truncate_filename(doc_a, 40)],
        [get_text("pdf_document_name", language), truncate_filename(doc_b, 40)],
        [get_text("pdf_similarity_score", language), f"{overall_similarity:.1%}"],
        [get_text("pdf_detection_threshold", language), f"{threshold:.1%}"],
    ]

    doc_table = Table(doc_data, colWidths=[2 * inch, 4 * inch], hAlign=TA_LEFT)
    if dark_mode:
        table_style_cmds = [
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#FFFFFF")),
            ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
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
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
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

    # Text statistics (if available)
    if doc_a_text is not None or doc_b_text is not None:
        story.append(Paragraph("Document Statistics", heading_style))
        story.append(Spacer(1, 0.1 * inch))

        # Compute statistics for each document
        doc_a_stats = compute_text_stats(doc_a_text) if doc_a_text else None
        doc_b_stats = compute_text_stats(doc_b_text) if doc_b_text else None

        # Create statistics table
        stats_data = [
            ["", doc_a, doc_b],
            [
                "Word Count",
                str(doc_a_stats["word_count"]) if doc_a_stats else "N/A",
                str(doc_b_stats["word_count"]) if doc_b_stats else "N/A",
            ],
            [
                "Sentence Count",
                str(doc_a_stats["sentence_count"]) if doc_a_stats else "N/A",
                str(doc_b_stats["sentence_count"]) if doc_b_stats else "N/A",
            ],
            [
                "Unique Words",
                str(doc_a_stats["unique_word_count"]) if doc_a_stats else "N/A",
                str(doc_b_stats["unique_word_count"]) if doc_b_stats else "N/A",
            ],
            [
                "Unique Word Ratio",
                f"{doc_a_stats['unique_word_ratio']:.2%}" if doc_a_stats else "N/A",
                f"{doc_b_stats['unique_word_ratio']:.2%}" if doc_b_stats else "N/A",
            ],
        ]

        # Calculate column widths - give more space to document names
        col_widths = [1.5 * inch, 2.25 * inch, 2.25 * inch]

        stats_table = Table(stats_data, colWidths=col_widths, hAlign=TA_LEFT)
        stats_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#374151")),
                    ("FONTNAME", (0, 0), (0, -1), font_bold),
                    ("FONTNAME", (1, 0), (-1, 0), font_bold),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(stats_table)
        story.append(Spacer(1, 0.3 * inch))

    # Visual similarity bar
    sim_color = get_similarity_color(overall_similarity)
    story.append(Paragraph("Similarity Score Visualization", heading_style))

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

    if top_pairs:
        story.append(Paragraph("Top Suspicious Paragraph Pairs", heading_style))
        story.append(
            Paragraph(
                f"Showing top {len(top_pairs)} most similar paragraph pairs above threshold.",
                normal_style,
            )
        )
        story.append(Spacer(1, 0.1 * inch))

        for rank, (chunk_a, chunk_b, score) in enumerate(top_pairs, 1):
            pair_color = get_similarity_color(score)
            pair_header = Paragraph(
                f"<b>Pair #{rank}</b> — Similarity: <font color='{pair_color}'>{score:.1%}</font>",
                ParagraphStyle(
                    "PairHeader",
                    fontName=font_bold,
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
            truncated_a = wrap_text(chunk_a, max_chars=500)
            truncated_b = wrap_text(chunk_b, max_chars=500)

            # Compare and highlight differences
            from src.utils.diff_highlighter import highlight_overlap

            hl_a, hl_b = highlight_overlap(truncated_a, truncated_b)

            backcolor_hex = "#FEF08A" if not dark_mode else "#854D0E"
            textcolor_hex = "#1E293B" if not dark_mode else "#FFFFFF"
            mark_start = "<mark style='background-color: rgba(250, 204, 21, 0.3); color: inherit; padding: 1px 3px; border-radius: 3px;'>"

            hl_a = hl_a.replace(
                mark_start,
                f"<font backcolor='{backcolor_hex}' color='{textcolor_hex}'>",
            ).replace("</mark>", "</font>")
            hl_b = hl_b.replace(
                mark_start,
                f"<font backcolor='{backcolor_hex}' color='{textcolor_hex}'>",
            ).replace("</mark>", "</font>")

            for char in ["*", "_", "~", "`", "#", "[", "]", "(", ")"]:
                hl_a = hl_a.replace(f"\\{char}", char)
                hl_b = hl_b.replace(f"\\{char}", char)

            hl_a = break_long_urls(hl_a)
            hl_b = break_long_urls(hl_b)

            cell_header_style = ParagraphStyle(
                f"ComparisonCellHeader_{rank}",
                fontName=font_bold,
                fontSize=9,
                leading=12,
                textColor=HexColor("#FFFFFF") if dark_mode else HexColor("#111827"),
                wordWrap="CJK",
            )
            cell_body_style = ParagraphStyle(
                f"ComparisonCellBody_{rank}",
                fontName=font_regular,
                fontSize=9,
                leading=12,
                textColor=HexColor("#FFFFFF") if dark_mode else HexColor("#31333f"),
                wordWrap="CJK",
            )

            pair_data = [
                [
                    Paragraph(f"<b>From {doc_a}:</b>", cell_header_style),
                    Paragraph(f"<b>From {doc_b}:</b>", cell_header_style),
                ],
                [
                    Paragraph(hl_a, cell_body_style),
                    Paragraph(hl_b, cell_body_style),
                ],
            ]

            pair_table = Table(
                pair_data, colWidths=[2.5 * inch, 2.5 * inch], hAlign=TA_LEFT
            )
            if dark_mode:
                pair_table_cmds = [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E293B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#FFFFFF")),
                    ("FONTNAME", (0, 0), (-1, -1), font_regular),
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
                    ("FONTNAME", (0, 0), (-1, -1), font_regular),
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

            if rank == 3 and len(top_pairs) > 3:
                story.append(PageBreak())
    else:
        story.append(
            Paragraph(
                "No suspicious paragraph pairs found above threshold.",
                normal_style,
            )
        )

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

    doc.build(story, onFirstPage=_draw_header, onLaterPages=_draw_header)
    buffer.seek(0)
    return buffer


def highlight_pdf_matches(
    pdf_source: str | bytes,
    matching_chunks: list[str],
    highlight_color: tuple[float, float, float] = (1.0, 0.85, 0.0),
) -> bytes:
    """Opens an original PDF, searches for matching plagiarized text chunks,
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
            chunk_clean = str(chunk).strip()
            if len(chunk_clean) < 3:
                continue

            quad_matches = page.search_for(chunk_clean)
            for rect in quad_matches:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=highlight_color)
                annot.update()

    output_bytes = doc.tobytes()
    doc.close()
    return output_bytes


def generate_audit_summary_html(
    metrics: dict[str, Any],
    top_flagged_pairs: list[dict[str, Any]],
    report_title: str = "Class Plagiarism Audit Summary Report",
    class_section: str = "All Classes",
) -> str:
    """Generate a clean, self-contained HTML audit summary report for instructors.

    Args:
        metrics: Dictionary containing class-wide summary statistics (documents, pairs, incidents, severity breakdown).
        top_flagged_pairs: List of flagged document pair dictionaries.
        report_title: Title header for the audit report.
        class_section: Active class or section filter name.

    Returns:
        Full HTML report document string.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_docs = metrics.get("total_documents", 0)
    eval_pairs = metrics.get("evaluated_pairs", 0)
    flagged_cnt = metrics.get("flagged_incidents", len(top_flagged_pairs))
    threshold_pct = f"{metrics.get('threshold', 0.59):.0%}"

    high_cnt = metrics.get(
        "high_severity_count",
        sum(1 for p in top_flagged_pairs if p.get("similarity", 0) >= 0.90),
    )
    med_cnt = metrics.get(
        "medium_severity_count",
        sum(1 for p in top_flagged_pairs if 0.75 <= p.get("similarity", 0) < 0.90),
    )
    low_cnt = metrics.get(
        "low_severity_count",
        sum(1 for p in top_flagged_pairs if p.get("similarity", 0) < 0.75),
    )

    table_rows_html = ""
    if not top_flagged_pairs:
        table_rows_html = '<tr><td colspan="5" style="text-align: center; color: #64748b; padding: 20px;">No flagged plagiarism pairs found for this section.</td></tr>'
    else:
        for idx, item in enumerate(top_flagged_pairs, 1):
            doc_a = item.get("doc_a", item.get("document_a", "Document A"))
            doc_b = item.get("doc_b", item.get("document_b", "Document B"))
            score = item.get("similarity", item.get("similarity_score", 0.0))
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = 0.0

            if score >= 0.90:
                sev_badge = '<span style="background-color: #fee2e2; color: #dc2626; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">High (≥90%)</span>'
            elif score >= 0.75:
                sev_badge = '<span style="background-color: #ffedd5; color: #ea580c; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">Medium (75-89%)</span>'
            else:
                sev_badge = '<span style="background-color: #dcfce7; color: #16a34a; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">Low (&lt;75%)</span>'

            table_rows_html += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center;">#{idx}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-family: monospace;">{truncate_filename(str(doc_a), 35)}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-family: monospace;">{truncate_filename(str(doc_b), 35)}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: 700; text-align: center; color: #1e293b;">{score:.1%}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center;">{sev_badge}</td>
            </tr>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 36px;
            border: 1px solid #e2e8f0;
        }}
        .header {{
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            color: #1e1b4b;
            font-size: 24px;
        }}
        .meta-line {{
            color: #64748b;
            font-size: 13px;
            margin: 4px 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid #cbd5e1;
        }}
        .card-val {{
            font-size: 22px;
            font-weight: 700;
            color: #4f46e5;
            margin-top: 4px;
        }}
        .card-lbl {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section-title {{
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 28px;
            margin-bottom: 12px;
            border-left: 4px solid #4f46e5;
            padding-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 13px;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            text-align: left;
            padding: 10px;
            border-bottom: 2px solid #cbd5e1;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Plagiarism Audit Executive Summary</h1>
            <div class="meta-line"><b>Generated:</b> {generated_at} UTC | <b>Class / Section:</b> {class_section}</div>
            <div class="meta-line"><b>Detection Threshold:</b> {threshold_pct} | <b>Algorithm:</b> Transformer Semantic Vector Index (FAISS)</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-lbl">Total Documents</div>
                <div class="card-val">{total_docs}</div>
            </div>
            <div class="card">
                <div class="card-lbl">Evaluated Pairs</div>
                <div class="card-val">{eval_pairs}</div>
            </div>
            <div class="card">
                <div class="card-lbl">Flagged Incidents</div>
                <div class="card-val" style="color: #dc2626;">{flagged_cnt}</div>
            </div>
            <div class="card">
                <div class="card-lbl">High Risk (≥90%)</div>
                <div class="card-val" style="color: #ea580c;">{high_cnt}</div>
            </div>
        </div>

        <div class="section-title">📊 Severity Distribution Breakdown</div>
        <div style="display: flex; gap: 12px; margin-bottom: 20px;">
            <div style="flex: 1; background: #fef2f2; border: 1px solid #fca5a5; padding: 12px; border-radius: 6px; text-align: center;">
                <span style="font-weight: 700; color: #dc2626; font-size: 18px;">{high_cnt}</span><br>
                <span style="font-size: 12px; color: #991b1b;">High Severity (≥90%)</span>
            </div>
            <div style="flex: 1; background: #fff7ed; border: 1px solid #fdba74; padding: 12px; border-radius: 6px; text-align: center;">
                <span style="font-weight: 700; color: #ea580c; font-size: 18px;">{med_cnt}</span><br>
                <span style="font-size: 12px; color: #9a3412;">Medium Severity (75-89%)</span>
            </div>
            <div style="flex: 1; background: #f0fdf4; border: 1px solid #86efac; padding: 12px; border-radius: 6px; text-align: center;">
                <span style="font-weight: 700; color: #16a34a; font-size: 18px;">{low_cnt}</span><br>
                <span style="font-size: 12px; color: #166534;">Low Severity (&lt;75%)</span>
            </div>
        </div>

        <div class="section-title">🚨 Top Flagged Document Pairs</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 40px; text-align: center;">#</th>
                    <th>Document A</th>
                    <th>Document B</th>
                    <th style="text-align: center;">Similarity</th>
                    <th style="text-align: center;">Severity Level</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>

        <div class="footer">
            Semantic Plagiarism Detection System Audit Report · Automated Compliance Summary
        </div>
    </div>
</body>
</html>"""
    return html_content


def generate_audit_summary_pdf(
    metrics: dict[str, Any],
    top_flagged_pairs: list[dict[str, Any]],
    report_title: str = "Class Plagiarism Audit Summary Report",
    class_section: str = "All Classes",
) -> BytesIO:
    """Generate a clean, multi-page ReportLab PDF audit summary report.

    Args:
        metrics: Class summary statistics.
        top_flagged_pairs: Flagged pairs list.
        report_title: Document title.
        class_section: Section filter title.

    Returns:
        BytesIO containing generated PDF bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    font_regular, font_bold = _ensure_pdf_fonts()
    title_style = ParagraphStyle(
        "AuditTitle",
        parent=styles["Heading1"],
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor=HexColor("#1e1b4b"),
        spaceAfter=15,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    heading_style = ParagraphStyle(
        "AuditHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=13,
        leading=16,
        textColor=HexColor("#4f46e5"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "AuditBody",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=9,
        leading=12,
        textColor=HexColor("#334155"),
        wordWrap="CJK",
    )

    story = []
    story.append(Paragraph(report_title, title_style))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threshold_str = f"{metrics.get('threshold', 0.59):.0%}"
    meta_text = f"<b>Generated:</b> {timestamp} UTC &nbsp;|&nbsp; <b>Section:</b> {class_section} &nbsp;|&nbsp; <b>Threshold:</b> {threshold_str}"
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 0.15 * inch))

    total_docs = metrics.get("total_documents", 0)
    eval_pairs = metrics.get("evaluated_pairs", 0)
    flagged_cnt = metrics.get("flagged_incidents", len(top_flagged_pairs))
    high_cnt = metrics.get(
        "high_severity_count",
        sum(1 for p in top_flagged_pairs if p.get("similarity", 0) >= 0.90),
    )

    summary_data = [
        ["Total Documents", str(total_docs), "Evaluated Pairs", str(eval_pairs)],
        ["Flagged Incidents", str(flagged_cnt), "High Severity (≥90%)", str(high_cnt)],
    ]
    summary_table = Table(
        summary_data, colWidths=[1.8 * inch, 1.2 * inch, 1.8 * inch, 1.2 * inch]
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (-1, -1), font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#4F46E5")),
                ("TEXTCOLOR", (3, 0), (3, -1), HexColor("#DC2626")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Top Flagged Document Pairs", heading_style))

    table_data = [["#", "Document A", "Document B", "Similarity", "Severity"]]
    for idx, item in enumerate(top_flagged_pairs[:20], 1):
        doc_a = item.get("doc_a", item.get("document_a", "Doc A"))
        doc_b = item.get("doc_b", item.get("document_b", "Doc B"))
        score = item.get("similarity", item.get("similarity_score", 0.0))
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0

        if score >= 0.90:
            sev_str = "High"
        elif score >= 0.75:
            sev_str = "Medium"
        else:
            sev_str = "Low"

        table_data.append(
            [
                str(idx),
                truncate_filename(str(doc_a), 28),
                truncate_filename(str(doc_b), 28),
                f"{score:.1%}",
                sev_str,
            ]
        )

    if len(table_data) == 1:
        table_data.append(["-", "No flagged incidents found", "-", "-", "-"])

    pairs_table = Table(
        table_data,
        colWidths=[0.4 * inch, 2.3 * inch, 2.3 * inch, 1.0 * inch, 1.0 * inch],
    )
    pairs_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(pairs_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Audit Notes & Compliance", heading_style))
    notes_text = (
        "This audit report compiles class-wide similarity inspection metrics generated by the "
        "Semantic Plagiarism Detection System. Flagged document pairs reflect cosine similarity "
        "measurements computed over dense vector embeddings. Final academic integrity determination "
        "requires instructor evaluation."
    )
    story.append(Paragraph(notes_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_batch_plagiarism_report(
    incidents: list[dict[str, Any]],
    *,
    report_title: str = "Batch Plagiarism Investigation Report",
) -> BytesIO:
    """Generate one consolidated PDF containing all flagged incidents."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 12))

    total_incidents = len(incidents)

    severity_counts: dict[str, int] = {}
    for incident in incidents:
        severity = str(incident.get("severity_rank", "Unknown"))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    story.append(Paragraph("Summary Statistics", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Total flagged incidents: {total_incidents}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))

    severity_rows = [["Severity", "Count"]]
    severity_rows.extend(
        [severity, str(count)] for severity, count in severity_counts.items()
    )

    severity_table = Table(severity_rows)
    severity_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(severity_table)
    story.append(PageBreak())

    story.append(Paragraph("Flagged Plagiarism Cases", styles["Heading2"]))
    story.append(Spacer(1, 10))

    for index, incident in enumerate(incidents, start=1):
        document_a = str(incident.get("document_a", "Unknown"))
        document_b = str(incident.get("document_b", "Unknown"))
        severity = str(incident.get("severity_rank", "Unknown"))

        similarity = incident.get("similarity_score", 0)
        try:
            similarity_text = f"{float(similarity):.1%}"
        except (TypeError, ValueError):
            similarity_text = str(similarity)

        story.append(
            Paragraph(
                f"Case {index}: {document_a} ↔ {document_b}",
                styles["Heading3"],
            )
        )

        case_rows = [
            ["Field", "Value"],
            ["Document A", document_a],
            ["Document B", document_b],
            ["Similarity", similarity_text],
            ["Severity", severity],
        ]

        case_table = Table(
            case_rows,
            colWidths=[1.5 * inch, 4.5 * inch],
        )
        case_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(case_table)
        story.append(Spacer(1, 15))

        if index < len(incidents):
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_audit_summary_report(
    metrics: dict[str, Any],
    top_flagged_pairs: list[dict[str, Any]],
    output_format: str = "pdf",
    class_section: str = "All Classes",
) -> bytes | str:
    """Consolidated helper to export audit summary report in PDF or HTML format.

    Args:
        metrics: Summary metrics dictionary.
        top_flagged_pairs: List of top flagged document pairs.
        output_format: Output format type ('pdf' or 'html').
        class_section: Selected class section label.

    Returns:
        Bytes for PDF export or str for HTML export.
    """
    if str(output_format).lower() == "html":
        return generate_audit_summary_html(
            metrics=metrics,
            top_flagged_pairs=top_flagged_pairs,
            class_section=class_section,
        )
    else:
        pdf_buf = generate_audit_summary_pdf(
            metrics=metrics,
            top_flagged_pairs=top_flagged_pairs,
            class_section=class_section,
        )
        return pdf_buf.getvalue()
