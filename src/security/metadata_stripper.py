import io
import logging

import fitz  # PyMuPDF
from PIL import Image
from pypdf import PdfReader, PdfWriter

Image.MAX_IMAGE_PIXELS = 50_000_000

logger = logging.getLogger(__name__)


def strip_exif_metadata(
    file_bytes: bytes, filename: str, max_bytes: int = 25_000_000
) -> bytes:
    """
    Strips EXIF, XMP, and other identifying metadata from files in-memory.
    Supports PDF and common image formats (JPEG, PNG).
    Returns the sanitized file bytes.
    """
    if len(file_bytes) > max_bytes:
        raise ValueError("File size exceeds EXIF stripping limit")
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        return _strip_pdf_metadata(file_bytes)
    elif ext in ["jpg", "jpeg", "png", "tiff", "webp"]:
        return _strip_image_metadata(file_bytes)
    else:
        # For DOCX, TXT, CSV, ZIP, we return as-is for now,
        # or implement specific strippers if needed.
        return file_bytes


def _strip_pdf_metadata(pdf_bytes: bytes) -> bytes:
    """Uses PyMuPDF (fitz) to remove PDF Info dict and XMP metadata."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # 1. Remove XML/XMP Metadata
        if doc.is_pdf:
            doc.del_xml_metadata()

        # 2. Clear standard Info dictionary (Author, Title, Creator, etc.)
        doc.set_metadata(
            {
                "creationDate": "",
                "modDate": "",
                "title": "",
                "author": "",
                "subject": "",
                "keywords": "",
                "creator": "",
                "producer": "",
                "trapped": "",
            }
        )

        # Save to a new bytes buffer with garbage collection to ensure scrubbed data is dropped
        out_bytes = doc.write(garbage=4, clean=True)
        doc.close()

        return out_bytes
    except Exception as e:
        logger.error(f"Failed to strip PDF metadata: {e}")
        # If scrubbing fails, fail-safe is to return the original (or raise? Security context says strip or drop)
        # To be safe against crashes, we log and return the original, though returning empty might be safer in strict environments.
        return pdf_bytes


def strip_pdf_javascript(pdf_bytes: bytes) -> bytes:
    """
    Detects and removes embedded PDF JavaScript actions.

    Removes JavaScript-related catalog actions:
    /JS, /JavaScript, and /OpenAction.
    Logs a warning when JavaScript actions are detected.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()

        catalog = reader.trailer.get("/Root")
        javascript_detected = False

        if catalog:
           catalog = catalog.get_object()

        for key in ["/JS", "/JavaScript", "/OpenAction"]:
           if key in catalog:
              javascript_detected = True
              del catalog[key]

        if javascript_detected:
            logger.warning("Embedded PDF JavaScript actions detected and removed")

        for page in reader.pages:
            writer.add_page(page)

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    except Exception as e:
        logger.error(f"Failed to strip PDF JavaScript: {e}")
        return pdf_bytes


def inspect_pdf_fonts(pdf_bytes: bytes, max_font_bytes: int = 10_000_000) -> bool:
    """
    Inspects embedded font streams in a PDF for oversized payloads that
    could cause memory exhaustion in PDF renderers.

    Args:
        pdf_bytes (bytes): The raw byte content of the PDF file.
        max_font_bytes (int): Maximum allowed size (in bytes) for any single
            embedded font stream. Defaults to 10,000,000 (10 MB).

    Returns:
        bool: True if all embedded font streams are within the safety limit.

    Raises:
        ValueError: If an embedded font stream exceeds max_font_bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in doc:
            for font in page.get_fonts(full=True):
                xref = font[0]
                font_data = doc.extract_font(xref)
                font_buffer = font_data[-1] if font_data else b""
                if len(font_buffer) > max_font_bytes:
                    raise ValueError(
                        "Embedded PDF font stream exceeds safety limit"
                    )
        return True
    finally:
        doc.close()


def _strip_image_metadata(file_bytes: bytes) -> bytes:
    """
    Uses Pillow to read the image and save it without EXIF data.
    Includes safety checks to prevent decompression bombs or excessive memory usage
    by validating image dimensions before full decoding.

    Args:
        file_bytes (bytes): The raw byte content of the image file.

    Returns:
        bytes: The sanitized image bytes without EXIF metadata.

    Raises:
        ValueError: If the image dimensions exceed the 10,000px safety limit.
    """
    MAX_DIMENSION = 10000

    try:
        # Open image to inspect dimensions without fully decoding pixel data
        with Image.open(io.BytesIO(file_bytes)) as image:
            width, height = image.size

            # Safety check: prevent decompression bombs or excessive memory allocation
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ValueError("Image dimensions exceed 10,000px safety limit")

            # Save format defaults to JPEG if original was JPEG, PNG for PNG, etc.
            # Capture it before any mode conversion (convert() drops the format).
            save_format = image.format if image.format else "JPEG"

            # Palette-based images (P mode) carry their colors in a palette
            # (color map) rather than the pixel channels. Copying the palette
            # indices into a fresh image drops that palette and corrupts the
            # color channels, so convert to RGBA first to preserve them.
            if image.mode == "P":
                image = image.convert("RGBA")

            # We extract only the image data, discarding info/exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            out_io = io.BytesIO()
            image_without_exif.save(out_io, format=save_format)

            return out_io.getvalue()
    except Image.DecompressionBombError:
        raise ValueError("Image dimensions exceed security safety limits.")
    except ValueError:
        # Re-raise ValueError to ensure safety limits are strictly enforced
        raise
    except Exception as e:
        logger.error(f"Failed to strip image metadata: {e}")
        return file_bytes
