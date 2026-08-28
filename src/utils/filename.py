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

"""Security helpers for untrusted document filenames."""

from __future__ import annotations

import hashlib
import html
import os
import re
import unicodedata
from collections.abc import Collection, Mapping
from pathlib import PurePath
from typing import IO, TypeVar

DEFAULT_FILENAME = "document"

# Upper bound on a sanitized filename, in characters.
#
# 128 leaves headroom under the 255-byte limit most filesystems impose, which
# matters because a single character can occupy up to four bytes once encoded,
# and under the shorter limits that apply inside archives and on the export
# paths a name is later joined onto.
MAX_FILENAME_LENGTH = 128

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_SEPARATOR_RE = re.compile(r"[\s_-]+")
_DOT_RE = re.compile(r"\.{2,}")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}

T = TypeVar("T")


def _basename(filename: str) -> str:
    """Return the last component for both POSIX and Windows paths."""
    normalized = filename.replace("\\", "/")
    return PurePath(normalized).name


def _safe_extension(filename: str) -> str:
    """Return a normalized, conservative extension."""
    # ``os.path.splitext(".pdf")`` treats the whole value as a stem. After
    # removing an HTML tag, however, an extension-only value should retain the
    # extension and receive the fallback stem.
    if re.fullmatch(r"\.[A-Za-z0-9]{1,15}", filename):
        return filename.lower()

    _stem, extension = os.path.splitext(filename)
    extension = extension.lower()

    if not extension:
        return ""

    cleaned = re.sub(r"[^a-z0-9.]", "", extension)
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    if cleaned == "." or len(cleaned) > 16:
        return ""
    return cleaned


def get_file_sha256_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest for file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def compute_file_hash_stream(
    file_stream: IO[bytes],
    chunk_size: int = 65536,
) -> str:
    """Return the SHA-256 hex digest for a file-like object.

    The stream is read incrementally in fixed-size chunks to avoid loading
    the entire file into memory.
    """
    hasher = hashlib.sha256()

    while chunk := file_stream.read(chunk_size):
        hasher.update(chunk)

    return hasher.hexdigest()


_SHA256_HEX_RE = re.compile(r"[0-9a-fA-F]{64}")


def normalize_sha256_hash(hash_str: str) -> str:
    """Validate a SHA-256 hex digest and return it in lower-case form.

    Args:
        hash_str: A 64-character hexadecimal SHA-256 digest, in any case.

    Returns:
        str: The digest normalized to lower-case.

    Raises:
        ValueError: If the input is not a 64-character hexadecimal string.
    """
    if not isinstance(hash_str, str) or not _SHA256_HEX_RE.fullmatch(hash_str):
        raise ValueError("Invalid SHA-256 hash: expected a 64-character hex string.")

    return hash_str.lower()


def sanitize_filename(
    filename: object,
    *,
    fallback: str = DEFAULT_FILENAME,
    max_length: int = MAX_FILENAME_LENGTH,
) -> str:
    """Return a filesystem- and HTML-safe document filename.

    The function treats filenames as untrusted input. It removes directory
    components, HTML tags, control characters, and shell/HTML punctuation,
    while retaining a conservative extension and a readable ASCII stem.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer.")
    if max_length < 8:
        raise ValueError("max_length must be at least 8.")

    raw = html.unescape(str(filename or ""))
    raw = unicodedata.normalize("NFKC", raw)
    raw = _CONTROL_RE.sub("", raw)

    # Strip markup before selecting the basename. Closing tags contain "/"
    # and must never be interpreted as path separators.
    raw = _HTML_TAG_RE.sub("", raw)
    raw = _basename(raw)

    extension = _safe_extension(raw)
    stem = raw[: -len(extension)] if extension else raw
    stem = _UNSAFE_RE.sub("_", stem)
    stem = _SEPARATOR_RE.sub("_", stem)
    stem = _DOT_RE.sub(".", stem)
    stem = stem.strip(" ._-")

    safe_fallback = _UNSAFE_RE.sub("_", str(fallback or DEFAULT_FILENAME))
    safe_fallback = _SEPARATOR_RE.sub("_", safe_fallback).strip(" ._-")
    if not safe_fallback:
        safe_fallback = DEFAULT_FILENAME

    if not stem:
        stem = safe_fallback

    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"

    maximum_stem_length = max_length - len(extension)
    if len(stem) > maximum_stem_length:
        hash_prefix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:4]
        hash_suffix = f"_{hash_prefix}"
        allowed_stem = maximum_stem_length - len(hash_suffix)
        if allowed_stem > 0:
            truncated_stem = stem[:allowed_stem].rstrip(" ._-")
            if not truncated_stem:
                truncated_stem = (
                    safe_fallback[:allowed_stem] or DEFAULT_FILENAME[:allowed_stem]
                )
            stem = f"{truncated_stem}{hash_suffix}"
        else:
            stem = hash_prefix[:maximum_stem_length]
    else:
        stem = stem[:maximum_stem_length].rstrip(" ._-")
        if not stem:
            stem = safe_fallback[:maximum_stem_length] or DEFAULT_FILENAME

    return f"{stem}{extension}"


def unique_filename(
    filename: object,
    existing_names: Collection[str],
    *,
    fallback: str = DEFAULT_FILENAME,
    max_length: int = MAX_FILENAME_LENGTH,
) -> str:
    """Return a sanitized filename that does not collide with existing names.

    The result never exceeds ``max_length``, whatever the collision count.

    Raises:
        ValueError: If ``max_length`` cannot hold the extension alongside the
            counter needed to disambiguate. This requires an unusually small
            limit together with a very large number of colliding names.
    """
    safe_name = sanitize_filename(
        filename,
        fallback=fallback,
        max_length=max_length,
    )
    existing = {str(name).casefold() for name in existing_names}

    if safe_name.casefold() not in existing:
        return safe_name

    stem, extension = os.path.splitext(safe_name)
    counter = 1

    while True:
        suffix = f"_{counter}"
        allowed_stem = max_length - len(extension) - len(suffix)

        # A negative budget used to be handed straight to the slice, where
        # stem[:-2] trims from the *end* instead of clamping to nothing. The
        # candidate then came out longer than max_length — exactly the limit
        # this function exists to enforce.
        if allowed_stem < 0:
            raise ValueError(
                f"max_length={max_length} is too small to disambiguate "
                f"{safe_name!r}: the extension {extension!r} and the counter "
                f"suffix {suffix!r} need {len(extension) + len(suffix)} "
                "characters between them."
            )

        candidate_stem = stem[:allowed_stem].rstrip(" ._-")
        candidate = f"{candidate_stem}{suffix}{extension}"

        if candidate.casefold() not in existing:
            return candidate

        counter += 1


def sanitize_filename_mapping(
    files: Mapping[object, T],
    *,
    fallback: str = DEFAULT_FILENAME,
    max_length: int = MAX_FILENAME_LENGTH,
) -> dict[str, T]:
    """Sanitize mapping keys and preserve all entries using unique names."""
    sanitized: dict[str, T] = {}

    for original_name, value in files.items():
        safe_name = unique_filename(
            original_name,
            sanitized,
            fallback=fallback,
            max_length=max_length,
        )
        sanitized[safe_name] = value

    return sanitized


DEFAULT_ALLOWED_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".csv",
        ".docx",
        ".epub",
        ".pdf",
        ".rtf",
        ".txt",
        ".zip",
    }
)

DANGEROUS_EXECUTABLE_EXTENSIONS = frozenset(
    {
        ".app",
        ".bat",
        ".bin",
        ".cmd",
        ".com",
        ".cpl",
        ".dll",
        ".dmg",
        ".exe",
        ".gadget",
        ".hta",
        ".inf",
        ".ins",
        ".iso",
        ".jar",
        ".js",
        ".jse",
        ".lnk",
        ".msc",
        ".msi",
        ".msp",
        ".mst",
        ".pif",
        ".ps1",
        ".ps1xml",
        ".ps2",
        ".ps2xml",
        ".psc1",
        ".psc2",
        ".reg",
        ".scr",
        ".sh",
        ".sys",
        ".vb",
        ".vbe",
        ".vbs",
        ".vhd",
        ".vhdx",
        ".vmdk",
        ".ws",
        ".wsc",
        ".wsf",
        ".wsh",
    }
)


class InvalidFileExtensionError(ValueError):
    """Raised when an uploaded filename has a disallowed final extension."""


def get_final_extension(filename: object) -> str:
    """Return the lowercase absolute last extension for an untrusted name.

    The filename is normalized to its final path component first, then split
    at the final dot only. This prevents an earlier safe-looking extension
    from hiding an executable payload, such as ``document.pdf.exe``.
    """
    raw = html.unescape(str(filename or ""))
    raw = unicodedata.normalize("NFKC", raw)
    raw = _CONTROL_RE.sub("", raw)
    raw = _HTML_TAG_RE.sub("", raw)
    basename = _basename(raw).strip()
    stem, extension = os.path.splitext(basename)
    return extension.casefold()


def get_file_extension_sanitized(filename: str) -> str:
    """Return the lower-case file extension, starting with a dot.

    Returns an empty string if the filename has no extension.
    """
    basename = _basename(str(filename or ""))
    _stem, extension = os.path.splitext(basename)
    return extension.lower()


_EXTENSION_BADGES: dict[str, str] = {
    ".pdf": "📄 PDF",
    ".docx": "📝 DOCX",
    ".doc": "📝 DOC",
    ".txt": "📑 TXT",
    ".csv": "📊 CSV",
    ".epub": "📚 EPUB",
    ".rtf": "📃 RTF",
    ".zip": "📦 ZIP",
}
_DEFAULT_EXTENSION_BADGE = "📁 FILE"


def format_extension_badge(filename: str) -> str:
    """Return a color-coded emoji badge describing a filename's format.

    Used in document list views so filenames aren't shown as plain text
    with no visual indication of file type, e.g. ``"📄 PDF"`` for a
    ``.pdf`` file. Falls back to a generic file badge for unrecognized or
    missing extensions.

    Examples
    --------
    >>> format_extension_badge("report.pdf")
    '📄 PDF'
    >>> format_extension_badge("notes.CSV")
    '📊 CSV'
    >>> format_extension_badge("no_extension")
    '📁 FILE'
    """
    extension = get_final_extension(filename)
    return _EXTENSION_BADGES.get(extension, _DEFAULT_EXTENSION_BADGE)


def validate_document_extension(
    filename: object,
    *,
    allowed_extensions: Collection[str] = DEFAULT_ALLOWED_DOCUMENT_EXTENSIONS,
    require_extension: bool = True,
) -> str:
    """Validate the final suffix and return its normalized lowercase value.

    Only the absolute last extension determines whether a file is acceptable.
    The validation is case-insensitive and rejects executable/script suffixes
    even when they follow a safe document extension.

    Args:
        filename: Untrusted filename supplied by a user or archive.
        allowed_extensions: Explicit final extensions accepted by the caller.
        require_extension: Whether extensionless filenames should be rejected.

    Raises:
        InvalidFileExtensionError: When the final suffix is missing, dangerous,
            or not present in the allowed extension set.
    """
    normalized_allowed = {
        (
            extension.casefold()
            if str(extension).startswith(".")
            else f".{str(extension).casefold()}"
        )
        for extension in allowed_extensions
    }
    final_extension = get_final_extension(filename)

    if not final_extension:
        if require_extension:
            raise InvalidFileExtensionError(
                "The uploaded file must have a supported extension."
            )
        return ""

    if final_extension in DANGEROUS_EXECUTABLE_EXTENSIONS:
        raise InvalidFileExtensionError(
            f"Executable or script file extensions are not allowed: {final_extension}"
        )

    if final_extension not in normalized_allowed:
        allowed_text = ", ".join(sorted(normalized_allowed))
        raise InvalidFileExtensionError(
            f"Unsupported final file extension {final_extension!r}. "
            f"Allowed extensions: {allowed_text}"
        )

    return final_extension


def sanitize_and_validate_filename(
    filename: object,
    *,
    allowed_extensions: Collection[str] = DEFAULT_ALLOWED_DOCUMENT_EXTENSIONS,
    fallback: str = DEFAULT_FILENAME,
    max_length: int = MAX_FILENAME_LENGTH,
) -> str:
    """Validate an untrusted final suffix, then return a sanitized filename."""
    validate_document_extension(
        filename,
        allowed_extensions=allowed_extensions,
    )
    return sanitize_filename(
        filename,
        fallback=fallback,
        max_length=max_length,
    )
