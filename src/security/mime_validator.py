"""Secure MIME validation using magic bytes and container inspection."""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Optional
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Strict mapping of file extension to allowed MIME types/signatures.
ALLOWED_MIME_TYPES: dict[str, list[str]] = {
    "pdf": ["application/pdf"],
    "docx": [
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ],
    "xlsx": [
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet",
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ],
    "doc": [
        "application/msword",
        "application/vnd.ms-office",
        "application/octet-stream",
    ],
    "zip": [
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ],
    "txt": ["text/plain", "text/x-python", "text/markdown"],
    "csv": ["text/csv", "text/plain", "application/csv"],
    "md": [
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    ],
    "rtf": ["application/rtf", "text/rtf", "text/plain"],
    "epub": [
        "application/epub+zip",
        "application/zip",
        "application/octet-stream",
    ],
    "odt": [
        "application/vnd.oasis.opendocument.text",
        "application/zip",
        "application/octet-stream",
    ],
    "png": ["image/png"],
    "jpg": ["image/jpeg"],
    "jpeg": ["image/jpeg"],
}

ALLOWED_MAGIC_HEADERS = {
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04"],
    "epub": [b"PK\x03\x04"],
    "odt": [b"PK\x03\x04"],
    "doc": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],
    "rtf": [b"{\\rtf"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
}

OOXML_EXTENSIONS = {"docx", "xlsx"}
OOXML_REQUIRED_PARTS = {
    "docx": {"[Content_Types].xml", "word/document.xml"},
    "xlsx": {"[Content_Types].xml", "xl/workbook.xml"},
}
OOXML_MAIN_CONTENT_TYPES = {
    "docx": {
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document.main+xml",
        "application/vnd.ms-word.document.macroEnabled.main+xml",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument." "spreadsheetml.sheet.main+xml",
        "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
    },
}

# Conservative limits for metadata-only archive inspection.
MAX_OOXML_ARCHIVE_ENTRIES = 10_000
MAX_OOXML_TOTAL_UNCOMPRESSED_SIZE = 250 * 1024 * 1024
MAX_CONTENT_TYPES_XML_SIZE = 2 * 1024 * 1024

BLOCKED_EXECUTABLE_EXTENSIONS = {
    "exe",
    "sh",
    "bat",
    "js",
    "vbs",
}


def _normalized_zip_name(name: str) -> str:
    """Normalize ZIP member names for case-insensitive comparisons."""
    return name.replace("\\", "/").lstrip("/").casefold()


def _validate_ooxml_archive(
    file_bytes: bytes,
    extension: str,
    filename: str,
) -> bool:
    """Verify that a ZIP payload is the requested OOXML package type."""
    if extension not in OOXML_EXTENSIONS:
        raise ValueError(f"Unsupported OOXML extension: {extension}")

    if not file_bytes.startswith(b"PK"):
        logger.warning(
            "[mime_validator] Invalid ZIP signature for OOXML file " "'%s'.",
            filename,
        )
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            members = archive.infolist()
            if len(members) > MAX_OOXML_ARCHIVE_ENTRIES:
                logger.warning(
                    "[mime_validator] OOXML archive '%s' contains " "too many entries.",
                    filename,
                )
                return False

            total_uncompressed_size = sum(member.file_size for member in members)
            if total_uncompressed_size > MAX_OOXML_TOTAL_UNCOMPRESSED_SIZE:
                logger.warning(
                    "[mime_validator] OOXML archive '%s' exceeds "
                    "the uncompressed-size safety limit.",
                    filename,
                )
                return False

            normalized_names = {
                _normalized_zip_name(member.filename): member for member in members
            }
            required = {name.casefold() for name in OOXML_REQUIRED_PARTS[extension]}

            if not required.issubset(normalized_names):
                logger.warning(
                    "[mime_validator] '%s' is missing required %s "
                    "OOXML package parts.",
                    filename,
                    extension.upper(),
                )
                return False

            content_types_member = normalized_names["[content_types].xml".casefold()]
            if content_types_member.file_size > MAX_CONTENT_TYPES_XML_SIZE:
                logger.warning(
                    "[mime_validator] [Content_Types].xml in '%s' "
                    "exceeds the safety limit.",
                    filename,
                )
                return False

            with archive.open(
                content_types_member,
                "r",
            ) as content_types_file:
                content_types_xml = content_types_file.read(
                    MAX_CONTENT_TYPES_XML_SIZE + 1
                )

            if len(content_types_xml) > MAX_CONTENT_TYPES_XML_SIZE:
                logger.warning(
                    "[mime_validator] [Content_Types].xml in '%s' " "is too large.",
                    filename,
                )
                return False

            root = ElementTree.fromstring(content_types_xml)
            declared_content_types = {
                element.attrib.get("ContentType", "")
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] == "Override"
            }

            if not (declared_content_types & OOXML_MAIN_CONTENT_TYPES[extension]):
                logger.warning(
                    "[mime_validator] '%s' does not declare a valid "
                    "%s main content type.",
                    filename,
                    extension.upper(),
                )
                return False

            bad_member = archive.testzip()
            if bad_member is not None:
                logger.warning(
                    "[mime_validator] Corrupt OOXML member '%s' in " "'%s'.",
                    bad_member,
                    filename,
                )
                return False

            return True

    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ElementTree.ParseError,
        KeyError,
        OSError,
        RuntimeError,
    ) as exception:
        logger.warning(
            "[mime_validator] Invalid OOXML archive '%s': %s",
            filename,
            exception,
        )
        return False


def _check_magic_bytes(
    file_bytes: bytes,
    extension: str,
    filename: str,
) -> Optional[bool]:
    """Attempt MIME validation using python-magic.

    Returns:
        True when detected MIME is allowed.
        False when a definite mismatch is detected.
        None when python-magic is unavailable or fails.
    """
    try:
        import magic

        mime_type = magic.from_buffer(file_bytes, mime=True)
        if mime_type:
            detected = mime_type.split(";")[0].strip().lower()
            allowed = ALLOWED_MIME_TYPES[extension]

            if detected in allowed:
                return True

            if detected.startswith("text/") and extension in {
                "txt",
                "csv",
                "md",
                "rtf",
            }:
                return True

            logger.warning(
                "[mime_validator] MIME type mismatch for '%s'. "
                "Expected one of %s, got '%s'.",
                filename,
                allowed,
                detected,
            )
            return False

    except (ImportError, ModuleNotFoundError) as exception:
        logger.debug(
            "[mime_validator] python-magic unavailable; using "
            "fallback validation: %s",
            exception,
        )
    except Exception as exception:
        logger.debug(
            "[mime_validator] python-magic failed; using fallback " "validation: %s",
            exception,
        )

    return None


def _check_extension_fallback(
    file_bytes: bytes,
    extension: str,
    filename: str,
) -> bool:
    """Validate binary headers or text encoding without python-magic."""
    if extension in ALLOWED_MAGIC_HEADERS:
        if any(
            file_bytes.lstrip().startswith(header)
            for header in ALLOWED_MAGIC_HEADERS[extension]
        ):
            return True

        logger.warning(
            "[mime_validator] Fallback magic-byte check failed for " "'%s'.",
            filename,
        )
        return False

    if extension in {"txt", "csv", "md"}:
        if b"\x00" in file_bytes:
            logger.warning(
                "[mime_validator] Text validation failed for '%s': "
                "binary null byte detected.",
                filename,
            )
            return False

        for encoding in ("utf-8", "utf-16"):
            try:
                file_bytes.decode(encoding, errors="strict")
                return True
            except UnicodeDecodeError:
                continue

        logger.warning(
            "[mime_validator] Text validation failed for '%s': "
            "not valid UTF-8 or UTF-16.",
            filename,
        )
        return False

    return False


def validate_single_extension(filename: str) -> bool:
    """Reject filenames with executable double extensions."""
    parts = filename.lower().split(".")

    if len(parts) < 3:
        return True

    return parts[-1] not in BLOCKED_EXECUTABLE_EXTENSIONS


def validate_mime_type(file_bytes: bytes, filename: str) -> bool:
    """Validate uploaded bytes against the declared file extension.

    OOXML documents are always inspected internally before MIME-library
    results are trusted because DOCX, XLSX, and ordinary ZIP files
    share the same leading magic bytes.
    """
    if not file_bytes:
        return False
        
    if not validate_single_extension(filename):
        logger.warning(
        "[mime_validator] Blocked executable double extension: '%s'.",
        filename,
        )
        return False

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not extension or extension not in ALLOWED_MIME_TYPES:
        logger.warning(
            "[mime_validator] Unsupported extension '%s' for '%s'.",
            extension,
            filename,
        )
        return False

    if extension in OOXML_EXTENSIONS:
        return _validate_ooxml_archive(
            file_bytes,
            extension,
            filename,
        )

    # Legacy Microsoft Word .doc files use the OLE Compound File signature.
    # Validate the complete eight-byte header before trusting MIME detection.
    if extension == "doc":
        ole_header = ALLOWED_MAGIC_HEADERS["doc"][0]
        if not file_bytes.startswith(ole_header):
            logger.warning(
                "[mime_validator] Invalid OLE Compound File header for '%s'.",
                filename,
            )
            return False

    # PDF validation is intentionally strict even when libmagic is
    # permissive or unavailable.
    if extension == "pdf" and not file_bytes.startswith(b"%PDF-"):
        logger.warning(
            "[mime_validator] Invalid PDF magic header for '%s'.",
            filename,
        )
        return False

    magic_result = _check_magic_bytes(
        file_bytes,
        extension,
        filename,
    )
    if magic_result is not None:
        return magic_result

    return _check_extension_fallback(
        file_bytes,
        extension,
        filename,
    )
