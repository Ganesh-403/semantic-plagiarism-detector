"""Utility for parsing files and extracting metadata."""

import fitz


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
"""
src/utils/file_parser.py
------------------------
Utility functions for parsing PDF, DOCX, TXT, and Markdown files.
Supports decrypted and password-protected PDF parsing using PyMuPDF (fitz),
along with file categorization, validation helpers, and PDF metadata extraction.
"""

from typing import Any, List, Optional, Tuple

import fitz  # PyMuPDF


class EncryptedPDFError(Exception):
    """Custom exception raised when a PDF requires a password to be read."""
    pass


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


def extract_text_from_pdf(file_bytes: bytes, password: Optional[str] = None) -> Tuple[str, bool]:
    """
    Extracts text from PDF bytes.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.
        password (str, optional): Password to decrypt the PDF if protected.

    Returns:
        Tuple[str, bool]: Extracted text, and a boolean flag indicating if the PDF was password-protected.

    Raises:
        EncryptedPDFError: If the PDF is encrypted and no password (or an incorrect password) is provided.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    is_protected = doc.is_encrypted or doc.needs_pass

    if is_protected:
        if not password:
            raise EncryptedPDFError("PDF is password-protected. Password required.")

        # doc.authenticate returns > 0 on success
        auth_success = doc.authenticate(password)
        if not auth_success:
            raise EncryptedPDFError("Incorrect password for PDF.")

    text_content = []
    for page in doc:
        text_content.append(page.get_text())

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

    ext = filename.split('.')[-1].lower() if '.' in filename else ""

    mime_mapping = {
        'pdf': 'pdf',
        'doc': 'word_document',
        'docx': 'word_document',
        'txt': 'text',
        'md': 'text',
        'markdown': 'text',
        'mdown': 'text',
        'csv': 'text',
        'rtf': 'text',
        'py': 'code',
        'js': 'code',
        'java': 'code',
        'cpp': 'code',
        'c': 'code',
        'html': 'code',
        'css': 'code',
        'zip': 'archive',
        'rar': 'archive',
        'tar': 'archive',
        'gz': 'archive',
        '7z': 'archive',
    }

    return mime_mapping.get(ext, 'unknown')


def get_supported_mime_categories() -> List[str]:
    """
    Retrieve a list of all supported high-level MIME categories.

    Returns:
        List[str]: A list of unique category names.
    """
    # We use a dummy filename to trigger the mapping logic, then extract unique values
    # Alternatively, we can just return the static list derived from the mapping
    return ['pdf', 'word_document', 'text', 'code', 'archive', 'unknown']


def is_extension_supported(filename: str, allowed_categories: Optional[List[str]] = None) -> bool:
    """
    Check if a file's extension belongs to an allowed list of MIME categories.

    Args:
        filename: The name of the file to check.
        allowed_categories: List of allowed categories. Defaults to all known categories except 'unknown'.

    Returns:
        bool: True if the file's category is in the allowed list, False otherwise.
    """
    if allowed_categories is None:
        allowed_categories = ['pdf', 'word_document', 'text', 'code', 'archive']

    category = get_file_mime_category(filename)
    return category in allowed_categories
