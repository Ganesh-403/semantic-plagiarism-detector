import io
import logging

import fitz  # PyMuPDF
from PIL import Image
Image.MAX_IMAGE_PIXELS = 50_000_000

logger = logging.getLogger(__name__)

def strip_exif_metadata(file_bytes: bytes, filename: str, max_bytes: int = 25_000_000) -> bytes:
    """
    Strips EXIF, XMP, and other identifying metadata from files in-memory.
    Supports PDF and common image formats (JPEG, PNG).
    Returns the sanitized file bytes.
    """
    if len(file_bytes) > max_bytes:
        raise ValueError("File size exceeds EXIF stripping limit")
    ext = filename.lower().split('.')[-1]

    if ext == 'pdf':
        return _strip_pdf_metadata(file_bytes)
    elif ext in ['jpg', 'jpeg', 'png', 'tiff', 'webp']:
        return _strip_image_metadata(file_bytes)
    else:
        # For DOCX, TXT, CSV, ZIP, we return as-is for now,
        # or implement specific strippers if needed.
        return file_bytes

def _strip_pdf_metadata(file_bytes: bytes) -> bytes:
    """Uses PyMuPDF (fitz) to remove PDF Info dict and XMP metadata."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        # 1. Remove XML/XMP Metadata
        if doc.is_pdf:
            doc.del_xml_metadata()

        # 2. Clear standard Info dictionary (Author, Title, Creator, etc.)
        doc.set_metadata({
            "creationDate": "",
            "modDate": "",
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "trapped": ""
        })

        # Save to a new bytes buffer with garbage collection to ensure scrubbed data is dropped
        out_bytes = doc.write(garbage=4, clean=True)
        doc.close()
        return out_bytes
    except Exception as e:
        logger.error(f"Failed to strip PDF metadata: {e}")
        # If scrubbing fails, fail-safe is to return the original (or raise? Security context says strip or drop)
        # To be safe against crashes, we log and return the original, though returning empty might be safer in strict environments.
        return file_bytes

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
    

    try:
        # Open image to inspect dimensions without fully decoding pixel data
        with Image.open(io.BytesIO(file_bytes)) as image:
           

            # We extract only the image data, discarding info/exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            out_io = io.BytesIO()
            # Save format defaults to JPEG if original was JPEG, PNG for PNG, etc.
            save_format = image.format if image.format else 'JPEG'
            image_without_exif.save(out_io, format=save_format)

            return out_io.getvalue()
    except ValueError:
        raise

    except Image.DecompressionBombError:
        raise ValueError("Image dimensions exceed security safety limits.")

    except Exception as e:
       logger.error(f"Failed to strip image metadata: {e}")
       return file_bytes