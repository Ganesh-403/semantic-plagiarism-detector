"""
badge_generator.py
------------------
Generates "Originality Verified" badges for students with 0% similarity results.
Supports both PNG and PDF output formats for gamification and academic integrity encouragement.
"""

import html
import re
from datetime import datetime
from io import BytesIO
from typing import Optional
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HEX_COLOR_PATTERN = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
DEFAULT_BADGE_COLOR = "#4f46e5"


def validate_hex_color(color: Optional[str]) -> str:
    """Validate a hex color string, auto-prepending '#' if missing, and
    falling back to a safe default if the value is invalid."""
    if not color:
        return DEFAULT_BADGE_COLOR
    candidate = color if color.startswith("#") else f"#{color}"
    if HEX_COLOR_PATTERN.match(candidate):
        return candidate
    return DEFAULT_BADGE_COLOR


def generate_badge_svg(
    student_name: str = "Student",
    date: Optional[str] = None,
    accent_color: Optional[str] = None,
) -> str:
    """
    Generates a simple SVG "Originality Verified" badge.

    The accent_color is validated (and defaulted if invalid) before being
    inserted into the SVG markup, preventing malformed or unescaped color
    values from producing invalid SVG.

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        accent_color: Optional hex color string for the badge accent

    Returns:
        A string containing the SVG markup for the badge.
    """
    safe_color = validate_hex_color(accent_color)
    if date is None:
        date = datetime.now().strftime("%B %d, %Y")

    safe_name = html.escape(student_name)
    safe_date = html.escape(date)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120" viewBox="0 0 400 120">
  <rect width="400" height="120" rx="12" fill="{safe_color}" />
  <text x="20" y="45" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#ffffff">Originality Verified</text>
  <text x="20" y="75" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#e0e7ff">Awarded to: {safe_name}</text>
  <text x="20" y="100" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#e0e7ff">Date: {safe_date}</text>
</svg>"""


def generate_badge_png(    student_name: str = "Student",    date: Optional[str] = None,
    text_preview: str = "",
) -> BytesIO:
    """
    Generates a visually appealing PNG badge for plagiarism-free work.

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        text_preview: Preview of the verified text (optional)

    Returns:
        BytesIO buffer containing the PNG badge
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for PNG badge generation")

    # Badge dimensions
    width, height = 800, 600

    # Create image with gradient background
    img = Image.new("RGB", (width, height), color="#1e3a8a")
    draw = ImageDraw.Draw(img)

    # Create gradient effect
    for y in range(height):
        # Interpolate between dark blue and lighter blue
        r = int(30 + (59 - 30) * y / height)
        g = int(58 + (130 - 58) * y / height)
        b = int(138 + (246 - 138) * y / height)
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

    # Add decorative border
    border_color = "#fbbf24"
    border_width = 8
    draw.rectangle(
        [border_width, border_width, width - border_width, height - border_width],
        outline=border_color,
        width=border_width,
    )

    # Inner border
    draw.rectangle(
        [
            border_width + 4,
            border_width + 4,
            width - border_width - 4,
            height - border_width - 4,
        ],
        outline="#ffffff",
        width=2,
    )

    # Try to load fonts, fallback to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        subtitle_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    title_text = "ORIGINALITY VERIFIED"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 60), title_text, fill="#ffffff", font=title_font)

    # Subtitle
    subtitle_text = "Plagiarism-Free Certificate"
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (width - subtitle_width) // 2
    draw.text((subtitle_x, 120), subtitle_text, fill="#fbbf24", font=subtitle_font)

    # Checkmark icon (simple drawing)
    check_x, check_y = width // 2, 220
    check_size = 80
    # Draw circle
    draw.ellipse(
        [
            check_x - check_size,
            check_y - check_size,
            check_x + check_size,
            check_y + check_size,
        ],
        fill="#22c55e",
        outline="#ffffff",
        width=4,
    )
    # Draw checkmark
    check_points = [
        (check_x - 25, check_y + 5),
        (check_x - 5, check_y + 35),
        (check_x + 35, check_y - 25),
    ]
    draw.line(check_points, fill="#ffffff", width=8)

    # Student name
    name_text = f"Awarded to: {student_name}"
    name_bbox = draw.textbbox((0, 0), name_text, font=body_font)
    name_width = name_bbox[2] - name_bbox[0]
    name_x = (width - name_width) // 2
    draw.text((name_x, 340), name_text, fill="#ffffff", font=body_font)

    # Date
    if date is None:
        date = datetime.now().strftime("%B %d, %Y")
    date_text = f"Date: {date}"
    date_bbox = draw.textbbox((0, 0), date_text, font=body_font)
    date_width = date_bbox[2] - date_bbox[0]
    date_x = (width - date_width) // 2
    draw.text((date_x, 380), date_text, fill="#e0e7ff", font=body_font)

    # Text preview (truncated if too long)
    if text_preview:
        preview_text = (
            f"Verified: {text_preview[:80]}..."
            if len(text_preview) > 80
            else f"Verified: {text_preview}"
        )
        preview_bbox = draw.textbbox((0, 0), preview_text, font=small_font)
        preview_width = preview_bbox[2] - preview_bbox[0]
        preview_x = (width - preview_width) // 2
        draw.text((preview_x, 430), preview_text, fill="#cbd5e1", font=small_font)

    # Footer
    footer_text = "Semantic Plagiarism Detection System"
    footer_bbox = draw.textbbox((0, 0), footer_text, font=small_font)
    footer_width = footer_bbox[2] - footer_bbox[0]
    footer_x = (width - footer_width) // 2
    draw.text((footer_x, 540), footer_text, fill="#94a3b8", font=small_font)

    # Save to buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG", quality=95)
    buffer.seek(0)
    return buffer


def generate_badge_pdf(
    student_name: str = "Student",
    date: Optional[str] = None,
    text_preview: str = "",
    brand_color: Optional[str] = None,
) -> BytesIO:
    """
    Generates a professional PDF certificate for plagiarism-free work.

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        text_preview: Preview of the verified text (optional)
        brand_color: Optional hex color string for branding

    Returns:
        BytesIO buffer containing the PDF certificate
    """
    brand_hex = validate_hex_color(brand_color)
    brand_clr = HexColor(brand_hex)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Get custom styles
    styles = getSampleStyleSheet()

    # Custom title style
    title_style = ParagraphStyle(
        "BadgeTitle",
        parent=styles["Heading1"],
        fontSize=28,
        textColor=brand_clr,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    # Custom heading style
    heading_style = ParagraphStyle(
        "BadgeHeading",
        parent=styles["Heading2"],
        fontSize=18,
        textColor=HexColor("#f59e0b"),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    # Normal style
    normal_style = ParagraphStyle(
        "BadgeNormal",
        parent=styles["Normal"],
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    # Small style
    small_style = ParagraphStyle(
        "BadgeSmall",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=HexColor("#64748b"),
    )

    # Build story (PDF content)
    story = []

    # Decorative border
    border_data = [
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
    ]
    border_table = Table(
        border_data,
        colWidths=[1 * inch, 4.5 * inch, 1 * inch],
        rowHeights=[0.5 * inch, 6 * inch, 0.5 * inch],
        hAlign=TA_CENTER,
    )
    border_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (2, 2), HexColor("#f8fafc")),
                ("GRID", (0, 0), (2, 2), 3, brand_clr),
                ("VALIGN", (0, 0), (2, 2), "MIDDLE"),
            ]
        )
    )
    story.append(border_table)
    story.append(Spacer(1, 0.3 * inch))

    # Title
    story.append(Paragraph("ORIGINALITY VERIFIED", title_style))
    story.append(Spacer(1, 0.1 * inch))

    # Subtitle
    story.append(Paragraph("Plagiarism-Free Certificate", heading_style))
    story.append(Spacer(1, 0.5 * inch))

    # Green checkmark indicator
    checkmark_style = ParagraphStyle(
        "Checkmark",
        parent=styles["Normal"],
        fontSize=48,
        textColor=HexColor("#22c55e"),
        alignment=TA_CENTER,
    )
    story.append(Paragraph("✓", checkmark_style))
    story.append(Spacer(1, 0.3 * inch))

    # Awarded to
    story.append(Paragraph("<b>This certificate is awarded to:</b>", normal_style))
    story.append(Spacer(1, 0.1 * inch))

    # Student name
    name_style = ParagraphStyle(
        "StudentName",
        parent=styles["Heading2"],
        fontSize=22,
        textColor=HexColor("#1e293b"),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph(student_name, name_style))
    story.append(Spacer(1, 0.4 * inch))

    # Date
    if date is None:
        date = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"<b>Date:</b> {date}", normal_style))
    story.append(Spacer(1, 0.3 * inch))

    # Text preview
    if text_preview:
        preview = (
            text_preview[:150] + "..." if len(text_preview) > 150 else text_preview
        )
        story.append(Paragraph("<b>Verified Text Preview:</b>", normal_style))
        story.append(Paragraph(f"<i>{preview}</i>", small_style))
        story.append(Spacer(1, 0.4 * inch))

    # Achievement description
    story.append(
        Paragraph(
            "This certifies that the submitted work has been verified as "
            "original with 0% similarity to any indexed documents.",
            normal_style,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Divider
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("─" * 50, small_style))
    story.append(Spacer(1, 0.3 * inch))

    # Footer
    story.append(
        Paragraph(
            "Generated by Semantic Plagiarism Detection System",
            small_style,
        )
    )

    # Build PDF
    doc.build(story)
    from src.utils.pdf_report import compress_pdf_buffer

    return compress_pdf_buffer(buffer)
