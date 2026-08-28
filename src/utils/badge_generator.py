"""
badge_generator.py
------------------
Generates "Originality Verified" badges for students with 0% similarity results.
Supports both PNG and PDF output formats for gamification and academic integrity encouragement.
Also provides utilities for generating dynamic SVG and PNG badges for documentation,
dashboards, and API responses.

This module handles color validation, text measurement, and SVG template
rendering. It supports both standard hex color codes and common CSS
named colors, providing a robust fallback mechanism for theme configurations.

Issue #2898: Fallback behavior for named colors in Badge Generator.
"""

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PngImagePlugin = None

def has_pillow() -> bool:
    """Check if PIL/Pillow is installed."""
    return Image is not None

try:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    HexColor = None
    TA_CENTER = None
    A4 = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    inch = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None

logger = logging.getLogger(__name__)

# Comprehensive lookup dictionary for common CSS named colors.
# This allows theme configurations to use human-readable color names
# (e.g., "red", "transparent", "darkslategray") instead of forcing
# strict hex code validation.
CSS_NAMED_COLORS: dict[str, str] = {
    "transparent": "#00000000",
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "silver": "#c0c0c0",
    "gray": "#808080",
    "grey": "#808080",
    "maroon": "#800000",
    "olive": "#808000",
    "lime": "#00ff00",
    "aqua": "#00ffff",
    "teal": "#008080",
    "navy": "#000080",
    "fuchsia": "#ff00ff",
    "purple": "#800080",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "brown": "#a52a2a",
    "coral": "#ff7f50",
    "crimson": "#dc143c",
    "darkblue": "#00008b",
    "darkgreen": "#006400",
    "darkred": "#8b0000",
    "gold": "#ffd700",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lightblue": "#add8e6",
    "lightgreen": "#90ee90",
    "lightyellow": "#ffffe0",
    "moccasin": "#ffe4b5",
    "orangered": "#ff4500",
    "plum": "#dda0dd",
    "salmon": "#fa8072",
    "sienna": "#a0522d",
    "skyblue": "#87ceeb",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "tan": "#d2b48c",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "whitesmoke": "#f5f5f5",
    "yellowgreen": "#9acd32",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
}

# Regex pattern for validating standard hex color codes.
# Supports 3-digit (#RGB), 4-digit (#RGBA), 6-digit (#RRGGBB), and 8-digit (#RRGGBBAA) formats.
_HEX_COLOR_PATTERN = re.compile(
    r"^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)

DEFAULT_BADGE_COLOR = "#4f46e5"


def has_reportlab() -> bool:
    """Return True when reportlab is installed and PDF badge generation is available."""
    return SimpleDocTemplate is not None


def validate_hex_color(
    color: Optional[str], default_color: str = DEFAULT_BADGE_COLOR
) -> str:
    """Validate and normalize a color string into a standard hex format.

    This function accepts both standard hex color codes (e.g., "#ff0000")
    and common CSS named colors (e.g., "red", "transparent"). If a named
    color is provided, it is converted to its hex equivalent using the
    internal CSS_NAMED_COLORS dictionary.

    If the input is neither a valid hex code nor a recognized named color,
    the function logs a warning and returns the specified default color.

    Args:
        color: The input color string to validate. Can be a hex code or
               a CSS named color.
        default_color: The fallback hex color to return if validation fails.
                      Defaults to DEFAULT_BADGE_COLOR.

    Returns:
        A validated hex color string (e.g., "#ff0000").

    Examples:
        >>> validate_hex_color("#ff0000")
        '#ff0000'
        >>> validate_hex_color("red")
        '#ff0000'
        >>> validate_hex_color("transparent")
        '#00000000'
        >>> validate_hex_color("invalid_color_name")
        '#4f46e5'
    """
    if not color or not isinstance(color, str):
        logger.warning(
            "Invalid color input (empty or non-string). Falling back to default: %s",
            default_color,
        )
        return default_color

    # Strip whitespace and convert to lowercase for case-insensitive matching
    cleaned_color = color.strip().lower()

    # 1. Check if it's a valid standard hex code
    if _HEX_COLOR_PATTERN.match(cleaned_color):
        return cleaned_color

    # 2. Check if it's a recognized CSS named color (Issue #2898)
    if cleaned_color in CSS_NAMED_COLORS:
        hex_equivalent = CSS_NAMED_COLORS[cleaned_color]
        logger.debug(
            "Resolved CSS named color '%s' to hex equivalent '%s'.",
            color,
            hex_equivalent,
        )
        return hex_equivalent

    # 3. Fallback to default if neither hex nor named color matched
    logger.warning(
        "Unrecognized color format '%s'. Expected hex code or CSS named color. "
        "Falling back to default: %s",
        color,
        default_color,
    )
    return default_color


def generate_badge_svg(
    student_name: str = "Student",
    date: Optional[str] = None,
    accent_color: Optional[str] = None,
    font_family: str = "Verdana, Geneva, sans-serif",
    font_size: int = 11,
) -> str:
    """
    Generates a simple SVG "Originality Verified" badge.

    Builds the SVG with ElementTree so text/attributes are XML-escaped
    automatically instead of relying on manual string interpolation.

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        accent_color: Optional hex color string for the badge accent
        font_family: Font family to use for SVG text elements

    Returns:
        A string containing the SVG markup for the badge.
    """
    safe_color = validate_hex_color(accent_color, DEFAULT_BADGE_COLOR)
    if date is None:
        date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    safe_name = html.escape(student_name)
    safe_date = html.escape(date)
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": "400",
            "height": "120",
            "viewBox": "0 0 400 120",
        },
    )
    ET.SubElement(
        root,
        "rect",
        {
            "width": "400",
            "height": "120",
            "rx": "12",
            "fill": safe_color,
        },
    )

    title = ET.SubElement(
        root,
        "text",
        {
            "x": "20",
            "y": "45",
            "font-family": font_family,
            "font-size": str(font_size),
            "fill": "#ffffff",
        },
    )
    title.text = "Originality Verified"

    awarded = ET.SubElement(
        root,
        "text",
        {
            "x": "20",
            "y": "75",
            "font-family": font_family,
            "font-size": str(font_size),
            "fill": "#e0e7ff",
        },
    )
    awarded.text = f"Awarded to: {student_name}"

    dated = ET.SubElement(
        root,
        "text",
        {
            "x": "20",
            "y": "100",
            "font-family": font_family,
            "font-size": str(font_size),
            "fill": "#e0e7ff",
        },
    )
    dated.text = f"Date: {date}"

    return ET.tostring(root, encoding="unicode")


def generate_badge_png(
    student_name: str = "Student",
    date: Optional[str] = None,
    text_preview: str = "",
    student_id: Optional[str] = None,
) -> BytesIO:
    """
    Generates a visually appealing PNG badge for plagiarism-free work.
    Caches generated badges in Redis for 24 hours based on student ID/name and date (Issue #2941).

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        text_preview: Preview of the verified text (optional)
        student_id: Optional unique ID of the student for caching

    Returns:
        BytesIO buffer containing the PNG badge
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for PNG badge generation")

    target_date = date if date is not None else datetime.now().strftime("%B %d, %Y")
    ident = str(student_id if student_id is not None else student_name)

    # Check Redis cache
    from src.utils.redis_cache import BADGE_TTL, CacheNamespace, RedisCache

    cache = RedisCache.get_instance()
    cache_key = CacheNamespace.BADGES.build_key("png", ident, target_date)
    cached_bytes = cache.get(cache_key)
    if cached_bytes is not None and isinstance(cached_bytes, bytes):
        return BytesIO(cached_bytes)

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

    # Load bundled TTF font (Roboto-Regular or DejaVuSans), with system and default fallbacks
    fonts_dir = Path(__file__).parent.parent / "assets" / "fonts"
    bundled_fonts = [
        fonts_dir / "Roboto-Regular.ttf",
        fonts_dir / "DejaVuSans.ttf",
    ]

    def _load_badge_font(size: int):
        for font_path in bundled_fonts:
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except (IOError, OSError):
                    pass
        for system_font in ["arial.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(system_font, size)
            except (IOError, OSError):
                pass
        return ImageFont.load_default()

    title_font = _load_badge_font(48)
    subtitle_font = _load_badge_font(32)
    body_font = _load_badge_font(24)
    small_font = _load_badge_font(18)

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
        date = datetime.now(timezone.utc).strftime("%B %d, %Y")
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

    # Create PngInfo for accessibility alt-text metadata
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Description", f"Originality Verified Certificate for {student_name}")

    # Save to buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG", pnginfo=pnginfo, quality=95)
    png_bytes = buffer.getvalue()
    try:
        cache.set(cache_key, png_bytes, ttl=BADGE_TTL)
    except Exception:
        pass
    buffer.seek(0)
    return buffer


def generate_badge_pdf(
    student_name: str = "Student",
    date: Optional[str] = None,
    text_preview: str = "",
    brand_color: Optional[str] = None,
    student_id: Optional[str] = None,
) -> BytesIO:
    """
    Generates a professional PDF certificate for plagiarism-free work.
    Caches generated certificates in Redis for 24 hours based on student ID/name and date (Issue #2941).

    Args:
        student_name: Name of the student (optional, defaults to "Student")
        date: Date string (optional, defaults to current date)
        text_preview: Preview of the verified text (optional)
        brand_color: Optional hex color string for branding
        student_id: Optional unique ID of the student for caching

    Returns:
        BytesIO buffer containing the PDF certificate
    """
    if not has_reportlab():
        raise ImportError("reportlab is required for PDF badge generation")

    target_date = date if date is not None else datetime.now().strftime("%B %d, %Y")
    ident = str(student_id if student_id is not None else student_name)

    # Check Redis cache
    from src.utils.redis_cache import BADGE_TTL, CacheNamespace, RedisCache

    cache = RedisCache.get_instance()
    cache_key = CacheNamespace.BADGES.build_key("pdf", ident, target_date)
    cached_bytes = cache.get(cache_key)
    if cached_bytes is not None and isinstance(cached_bytes, bytes):
        return BytesIO(cached_bytes)

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

    result_buffer = compress_pdf_buffer(buffer)
    pdf_bytes = result_buffer.getvalue()
    try:
        cache.set(cache_key, pdf_bytes, ttl=BADGE_TTL)
    except Exception:
        pass
    result_buffer.seek(0)
    return result_buffer


def generate_svg_badge(
    label: str,
    message: str,
    label_color: str = "#555",
    message_color: str = "#007ec6",
    icon: Optional[str] = None,
) -> str:
    """Generate a shields.io-style SVG badge.

    Args:
        label: The left-side text (e.g., "build", "coverage").
        message: The right-side text (e.g., "passing", "95%").
        label_color: Background color for the label side.
        message_color: Background color for the message side.
        icon: Optional base64 encoded SVG icon.

    Returns:
        A complete SVG XML string.
    """
    # Validate colors using the robust fallback mechanism
    valid_label_color = validate_hex_color(label_color, "#555555")
    valid_message_color = validate_hex_color(message_color, "#007ec6")

    # Calculate approximate widths (simplified for this example)
    label_width = max(30, len(label) * 7 + 10)
    message_width = max(30, len(message) * 7 + 10)
    total_width = label_width + message_width

    svg_template = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
      <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
      </linearGradient>
      <mask id="a">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
      </mask>
      <g mask="url(#a)">
        <path fill="{valid_label_color}" d="M0 0h{label_width}v20H0z"/>
        <path fill="{valid_message_color}" d="M{label_width} 0h{message_width}v20H{label_width}z"/>
        <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
      </g>
      <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
        <text x="{label_width/2}" y="14">{label}</text>
        <text x="{label_width + message_width/2}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
        <text x="{label_width + message_width/2}" y="14">{message}</text>
      </g>
    </svg>"""

    return svg_template


def get_badge_cache_key(label: str, message: str, color: str) -> str:
    """Generate a deterministic cache key for a badge configuration."""
    raw_key = f"{label}|{message}|{color}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()
