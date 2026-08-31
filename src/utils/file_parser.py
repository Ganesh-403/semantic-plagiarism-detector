"""
src/utils/file_parser.py
------------------------
Utility functions for parsing and inspecting raw file data.

Provides helpers to determine file types based on magic byte signatures,
ensuring secure and accurate file routing before heavy extraction logic
is executed.

Supports decrypted and password-protected PDF parsing using PyMuPDF (fitz),
along with file categorization and validation helpers.
"""


import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, List, Optional, Tuple, Union

import fitz
logger = logging.getLogger(__name__)


# ── String & Name Formatting ─────────────────────────────────────────────────

def truncate_filename(name: str, max_len: int = 35) -> str:
    """Truncate filename with ellipsis if it exceeds max_len."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


# ── Magic Byte Signatures (Issue #1570) ──────────────────────────────────────

# Magic byte signatures for common document and image formats.
# Each tuple contains (byte_signature, mime_type, description).
_MAGIC_SIGNATURES = [
    # PDF: %PDF-
    (b"%PDF", "application/pdf", "Portable Document Format"),
    # ZIP Archive (also used for DOCX, XLSX, PPTX, ODT, EPUB)
    (b"PK\x03\x04", "application/zip", "ZIP Archive / Office Open XML"),
    (b"PK\x05\x06", "application/zip", "ZIP Empty Archive"),
    (b"PK\x07\x08", "application/zip", "ZIP Spanned Archive"),
    # Microsoft Compound File Binary (DOC, XLS, PPT, MSG)
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/msword", "MS Compound Document"),
    # Rich Text Format
    (b"{\\rtf", "application/rtf", "Rich Text Format"),
    # Images
    (b"\xff\xd8\xff", "image/jpeg", "JPEG Image"),
    (b"\x89PNG\r\n\x1a\n", "image/png", "PNG Image"),
    (b"GIF87a", "image/gif", "GIF Image (87a)"),
    (b"GIF89a", "image/gif", "GIF Image (89a)"),
    (b"BM", "image/bmp", "BMP Image"),
    (b"RIFF", "image/webp", "WebP Image (RIFF header)"),
    # Plain Text / Markdown (Heuristic: starts with printable ASCII)
    # Handled as fallback below.
]

# Maximum number of bytes to read for signature inspection
_MAX_INSPECTION_BYTES = 16


# ── Custom Exceptions ─────────────────────────────────────────────────────────



# Issue #2931: Maximum seconds allowed for extracting text from a single
# PDF page before aborting. Certain corrupted or maliciously crafted PDFs
# (e.g. containing millions of tiny vector lines) can cause PyMuPDF's
# get_text() to hang indefinitely on a single page.
PAGE_EXTRACTION_TIMEOUT_SECONDS = 10


class PageExtractionTimeoutError(Exception):
    """Raised when extracting text from a single PDF page exceeds the timeout."""


def _get_page_text_with_timeout(
    page: "fitz.Page",
    timeout: float = PAGE_EXTRACTION_TIMEOUT_SECONDS,
) -> str:
    """Run ``page.get_text()`` with a hard timeout to guard against hangs.

    Uses a worker thread so it works regardless of which thread this is
    called from (``signal.alarm`` only works on the main thread).

    Raises:
        PageExtractionTimeoutError: If extraction exceeds ``timeout`` seconds.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(page.get_text)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            raise PageExtractionTimeoutError(
                f"Extracting text from page {page.number} exceeded "
                f"{timeout} seconds."
            )


class EncryptedPDFError(Exception):
    """Custom exception raised when a PDF requires a password to be read."""

    pass


# ── MIME Type Detection from Bytes (Issue #1570) ─────────────────────────────


def get_file_mime_type_from_bytes(
    file_bytes: bytes | bytearray | memoryview,
) -> str:
    """Inspect raw byte headers to determine the file's MIME type.

    This function analyzes the magic bytes (file signature) at the beginning
    of the byte stream to identify the file format. This is critical for
    security validation to ensure a file's actual content matches its
    claimed extension, preventing malicious payload execution.

    Args:
        file_bytes: The raw bytes of the file to inspect. Can be bytes,
                    bytearray, or memoryview.

    Returns:
        A standard MIME type string (e.g., 'application/pdf').
        Returns 'text/plain' if the content appears to be valid ASCII/UTF-8 text.
        Returns 'application/octet-stream' if the MIME type cannot be determined.

    Examples:
        >>> get_file_mime_type_from_bytes(b'%PDF-1.4\\n...')
        'application/pdf'

        >>> get_file_mime_type_from_bytes(b'PK\\x03\\x04...')
        'application/zip'
    """
    if not file_bytes:
        logger.debug("get_file_mime_type_from_bytes: Empty byte stream provided.")
        return "application/octet-stream"

    # Extract the header bytes for inspection
    try:
        header = bytes(file_bytes[:_MAX_INSPECTION_BYTES])
    except Exception as exc:
        logger.warning(
            "get_file_mime_type_from_bytes: Failed to read header bytes: %s", exc
        )
        return "application/octet-stream"

    # Check against known magic signatures
    for signature, mime_type, description in _MAGIC_SIGNATURES:
        if header.startswith(signature):
            logger.debug(
                "get_file_mime_type_from_bytes: Matched signature for %s (%s)",
                description,
                mime_type,
            )
            return mime_type

    # Fallback 1: Check if it's likely plain text / markdown
    # If the first 1024 bytes are mostly printable ASCII/UTF-8, treat as text
    try:
        sample = bytes(file_bytes[:1024])
        # Decode to UTF-8 to verify it's valid text
        decoded = sample.decode("utf-8", errors="strict")

        # Count printable characters vs control characters
        printable_count = sum(1 for c in decoded if c.isprintable() or c in "\n\r\t")
        ratio = printable_count / len(decoded) if decoded else 0

        if ratio > 0.90:
            logger.debug("get_file_mime_type_from_bytes: Detected as plain text/UTF-8.")
            return "text/plain"
    except (UnicodeDecodeError, ValueError):
        # Not valid UTF-8 text
        pass

    # Fallback 2: Unknown binary format
    logger.debug(
        "get_file_mime_type_from_bytes: No matching signature found. "
        "Returning application/octet-stream."
    )
    return "application/octet-stream"


def is_office_open_xml(file_bytes: bytes | bytearray) -> bool:
    """Check if a ZIP file is specifically an Office Open XML document (DOCX, XLSX).

    Office documents are ZIP archives containing a specific [Content_Types].xml
    file at the root. This helper inspects the ZIP central directory or local
    file headers to verify its presence.

    Args:
        file_bytes: Raw bytes of a ZIP file.

    Returns:
        True if the ZIP contains OOXML markers, False otherwise.
    """
    if not file_bytes:
        return False

    # Quick string search in the first 4KB for the OOXML content types marker
    # This is a heuristic but highly reliable for standard Office files
    try:
        header_sample = bytes(file_bytes[:4096])
        return (
            b"[Content_Types].xml" in header_sample or b"_rels/.rels" in header_sample
        )
    except Exception:
        return False


# ── File Size Formatting ─────────────────────────────────────────────────────


def get_file_size_formatted(num_bytes: int) -> str:
    """
    Convert a file size in bytes to a human-readable string.

    Args:
        num_bytes (int): File size in bytes.

    Returns:
        str: Human-readable file size using B, KB, MB, or GB.
    """
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} {units[-1]}"




def validate_pdf_page_count(
    file_bytes: bytes,
    max_pages: int = 500,
) -> int:
    """Validate that a PDF does not exceed the configured page limit.

    The document is opened only to inspect its page count; no page text is
    extracted. The PyMuPDF document is always closed before returning or
    raising.

    Args:
        file_bytes: Raw PDF bytes.
        max_pages: Maximum allowed number of pages. Defaults to 500.

    Returns:
        The PDF page count when it is within the configured limit.

    Raises:
        TypeError: If ``file_bytes`` is not bytes-like or ``max_pages`` is not
            an integer.
        ValueError: If ``max_pages`` is less than one or the PDF exceeds the
            configured page limit.
        fitz.FileDataError: If the supplied bytes are not a valid PDF.
    """
    if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("file_bytes must be bytes-like.")

    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise TypeError("max_pages must be an integer.")

    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    doc = fitz.open(
        stream=bytes(file_bytes),
        filetype="pdf",
    )
    try:
        page_count = doc.page_count
    finally:
        doc.close()

    if page_count > max_pages:
        raise ValueError(
            "PDF exceeds maximum allowed page limit "
            f"({max_pages} pages)"
        )

    return page_count


def get_file_size_formatted_short(num_bytes: int) -> str:
    """
    Convert a file size in bytes to a compact human-readable string.

    Args:
        num_bytes (int): File size in bytes.

    Returns:
        str: Compact file size using B, KB, MB, or GB with no spaces
            and no trailing zeros (e.g. "1MB", "500KB", "12B").
    """
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            rounded = round(size, 2)
            if rounded == int(rounded):
                return f"{int(rounded)}{unit}"
            return f"{rounded:g}{unit}"
        size /= 1024

    return f"{size:g}{units[-1]}"


# ── PDF Extraction & Metadata ────────────────────────────────────────────────


def get_pdf_page_count(file_bytes: bytes) -> int:
    """Return the total page count of a PDF file from its bytes.

    Returns 0 if the bytes are empty, invalid, or corrupted.
    """
    if not file_bytes:
        return 0
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return 0


def extract_text_from_pdf(
    file_bytes: bytes, password: Optional[str] = None
) -> tuple[str, bool]:
    """
    Extracts text from PDF bytes.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.
        password (str, optional): Password to decrypt the PDF if protected.

    Returns:
        Tuple[str, bool]: Extracted text, and a boolean flag indicating if the PDF was password-protected.

    Raises:
        EncryptedPDFError: If the PDF is encrypted and no password (or an incorrect password) is provided.
        ValueError: If the PDF exceeds the configured page limit.
    """
    validate_pdf_page_count(file_bytes)

    doc = fitz.open(stream=file_bytes, filetype="pdf")

    try:
        is_protected = bool(doc.is_encrypted or doc.needs_pass)

        if is_protected:
            if not password:
                raise EncryptedPDFError("PDF is password-protected. Password required.")

            # doc.authenticate returns > 0 on success
            auth_success = doc.authenticate(password)
            if not auth_success:
                raise EncryptedPDFError("Incorrect password for PDF.")

        text_content = []
        for page in doc:
            try:
                text_content.append(_get_page_text_with_timeout(page))
            except PageExtractionTimeoutError:
                logger.error(
                    "Skipping page %s: text extraction exceeded %s seconds.",
                    page.number,
                    PAGE_EXTRACTION_TIMEOUT_SECONDS,
                )
                text_content.append("")
    finally:
        # Release the handle on every path, including the EncryptedPDFError
        # raises above, which previously leaked the open document.
        doc.close()

    return "\n".join(text_content), is_protected


def extract_pdf_metadata(file_bytes: bytes) -> dict[str, Any]:
    """
    Extract document metadata from PDF bytes.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.

    Returns:
        dict[str, Any]: Dictionary with keys 'title', 'author', 'creation_date',
            'mod_date', and 'page_count'. Missing or empty fields default to None.

    Raises:
        EncryptedPDFError: If the PDF is encrypted and requires a password.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise EncryptedPDFError("PDF is password-protected. Password required.")

    metadata = doc.metadata or {}
    page_count = doc.page_count
    doc.close()

    return {
        "title": metadata.get("title") or None,
        "author": metadata.get("author") or None,
        "creation_date": metadata.get("creationDate") or None,
        "mod_date": metadata.get("modDate") or None,
        "page_count": page_count,
    }


# ── File Categorization Helpers ──────────────────────────────────────────────


def get_file_mime_category(filename: str) -> str:
    """
    Categorize an uploaded file into a high-level MIME group based on its extension.

    This helper simplifies routing and validation logic by grouping specific
    file extensions into broader, semantic categories.

    Args:
        filename: The name of the file (e.g., "document.pdf", "script.PY").

    Returns:
        str: The MIME category. One of: 'pdf', 'word_document', 'text', 'code', 'archive', 'unknown'.
    """
    if not filename or not isinstance(filename, str):
        return "unknown"

    ext = filename.split(".")[-1].lower() if "." in filename else ""

    mime_mapping = {
        "pdf": "pdf",
        "doc": "word_document",
        "docx": "word_document",
        "txt": "text",
        "md": "text",
        "markdown": "text",
        "mdown": "text",
        "csv": "text",
        "rtf": "text",
        "py": "code",
        "js": "code",
        "java": "code",
        "cpp": "code",
        "c": "code",
        "html": "code",
        "css": "code",
        "zip": "archive",
        "rar": "archive",
        "tar": "archive",
        "gz": "archive",
        "7z": "archive",
    }

    return mime_mapping.get(ext, "unknown")


def get_supported_mime_categories() -> list[str]:
    """
    Retrieve a list of all supported high-level MIME categories.

    Returns:
        List[str]: A list of unique category names.
    """
    return ["pdf", "word_document", "text", "code", "archive", "unknown"]


def is_extension_supported(
    filename: str, allowed_categories: Optional[list[str]] = None
) -> bool:
    """
    Check if a file's extension belongs to an allowed list of MIME categories.

    Args:
        filename: The name of the file to check.
        allowed_categories: List of allowed categories. Defaults to all known categories except 'unknown'.

    Returns:
        bool: True if the file's category is in the allowed list, False otherwise.
    """
    if allowed_categories is None:
        allowed_categories = ["pdf", "word_document", "text", "code", "archive"]

    category = get_file_mime_category(filename)
    return category in allowed_categories
