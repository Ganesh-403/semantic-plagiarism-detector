"""Security helpers for untrusted document filenames."""

from __future__ import annotations

import html
import os
import re
import unicodedata
from collections.abc import Collection, Mapping
from pathlib import PurePath
from typing import TypeVar

DEFAULT_FILENAME = "document"
MAX_FILENAME_LENGTH = 255

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
    """Return a sanitized filename that does not collide with existing names."""
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
