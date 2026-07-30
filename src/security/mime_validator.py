import logging

logger = logging.getLogger(__name__)

# Strict mapping of file extension to allowed MIME types/signatures
# Note: For zip-based files like docx and epub, their magic byte type from the OS might occasionally
# be detected as generic application/zip, which is also allowed.
ALLOWED_MIME_TYPES = {
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    "doc": {
        "application/msword",
        "application/vnd.ms-office",
        "application/octet-stream",
    },
    "zip": {"application/zip", "application/x-zip-compressed", "application/octet-stream"},
    "txt": {"text/plain", "text/x-python", "text/markdown"},
    "csv": {"text/csv", "text/plain", "application/csv"},
    "md": {"text/markdown", "text/plain", "application/octet-stream"},
    "rtf": {"application/rtf", "text/rtf", "text/plain"},
    "epub": {"application/epub+zip", "application/zip", "application/octet-stream"},
}

# Fallback headers checking if python-magic is unavailable or has issues
ALLOWED_MAGIC_HEADERS = {
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],
    "zip": [b"PK\x03\x04"],
    "epub": [b"PK\x03\x04"],
    "doc": [b"\xd0\xcf\x11\xe0"],
    "rtf": [b"{\\rtf"],
}

def validate_mime_type(file_bytes: bytes, filename: str) -> bool:
    """Validate the uploaded file bytes against a whitelist of allowed MIME signatures based on file extension.

    Returns True if valid, False otherwise.
    """
    if not file_bytes:
        return False

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not extension or extension not in ALLOWED_MIME_TYPES:
        logger.warning(f"[mime_validator] Security check: Unsupported file extension '{extension}' for file '{filename}'.")
        return False

    # 1. Try python-magic validation
    try:
        import magic
        # Use python-magic to detect the actual MIME type of the buffer
        mime_type = magic.from_buffer(file_bytes, mime=True)
        if mime_type:
            # Clean/normalize mime type (sometimes it has parameters)
            mime_type_clean = mime_type.split(";")[0].strip().lower()
            allowed = ALLOWED_MIME_TYPES[extension]
            
            # Check if clean mime type is allowed
            if mime_type_clean in allowed:
                return True
                
            # If magic says text/plain or similar, allow it for text formats
            if mime_type_clean.startswith("text/") and extension in {"txt", "csv", "md", "rtf"}:
                return True

            logger.warning(
                f"[mime_validator] Security warning: MIME type mismatch for '{filename}'. "
                f"Expected one of {allowed}, got '{mime_type_clean}'."
            )
    except Exception as e:
        logger.debug(f"[mime_validator] python-magic failed, falling back to header validation: {e}")

    # 2. Fallback: Magic Byte Header Check
    # If the extension has a known binary header signature, check it
    if extension in ALLOWED_MAGIC_HEADERS:
        headers = ALLOWED_MAGIC_HEADERS[extension]
        for header in headers:
            if file_bytes.lstrip().startswith(header):
                return True
        logger.warning(f"[mime_validator] Security warning: Fallback magic bytes check failed for '{filename}'.")
        return False

    # For text files, if magic failed, we can verify it contains mostly printable text
    if extension in {"txt", "csv", "md"}:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                file_bytes.decode(encoding, errors="strict")
                return True
            except UnicodeDecodeError:
                continue
        logger.warning(f"[mime_validator] Security warning: Text validation check failed for '{filename}' (not valid UTF-8/UTF-16/Latin-1).")
        return False

    return False
