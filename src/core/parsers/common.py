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

"""src/core/parsers/common.py - Common types, constants, and validation helpers."""

import io
import logging
from pathlib import Path
from typing import BinaryIO, Optional, Union

from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

PDFInput: TypeAlias = str | Path | bytes | BinaryIO | io.BytesIO

DEFAULT_OCR_DPI = 250
DEFAULT_OCR_LANGUAGE = "eng"

from src.core.app_config import SUPPORTED_OCR_LANGUAGES

# In-memory session scan counters for rate limiting
_session_scan_counters: dict[str, int] = {}
MAX_BATCH_FILES = 100


def check_batch_rate_limit(file_count: int, session_id: Optional[str] = None) -> None:
    """Enforce batch rate limit on document parsing requests.

    Parameters
    ----------
    file_count : int
        Number of files in current batch.
    session_id : str, optional
        Unique session identifier for tracking cumulative uploads.

    Raises
    ------
    ValueError
        If batch size exceeds single-batch limit or cumulative session limit.
    """
    if file_count > MAX_BATCH_FILES:
        raise ValueError(
            f"Batch size of {file_count} files exceeds maximum threshold of {MAX_BATCH_FILES} files."
        )

    if session_id:
        current_count = _session_scan_counters.get(session_id, 0)
        if current_count + file_count > MAX_BATCH_FILES * 5:
            raise ValueError(
                "Cumulative session limit exceeded. Please try again later."
            )
        _session_scan_counters[session_id] = current_count + file_count


def validate_ocr_dpi(value: int) -> int:
    """Validate and clamp OCR DPI resolution within reasonable boundaries (150 - 400)."""
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid ocr_dpi value %r. Defaulting to %d.",
                value,
                DEFAULT_OCR_DPI,
            )
            return DEFAULT_OCR_DPI

    if value < 150 or value > 400:
        clamped = max(150, min(400, value))
        logger.warning(
            "ocr_dpi %d is outside valid range [150, 400]. Clamped to %d.",
            value,
            clamped,
        )
        return clamped
    return value


def validate_ocr_language(value: str) -> str:
    """Validate OCR language string against supported language codes."""
    if not isinstance(value, str) or not value.strip():
        logger.warning(
            "Invalid ocr_language %r. Defaulting to '%s'.",
            value,
            DEFAULT_OCR_LANGUAGE,
        )
        return DEFAULT_OCR_LANGUAGE

    parts = [p.strip().lower() for p in value.split("+")]
    if not parts or any(not p for p in parts):
        logger.warning(
            "Invalid ocr_language %r. Defaulting to '%s'.",
            value,
            DEFAULT_OCR_LANGUAGE,
        )
        return DEFAULT_OCR_LANGUAGE

    for part in parts:
        if part not in SUPPORTED_OCR_LANGUAGES:
            logger.warning(
                "Unsupported OCR language '%s'. Defaulting to '%s'. "
                "Supported languages: %s",
                part,
                DEFAULT_OCR_LANGUAGE,
                ", ".join(sorted(SUPPORTED_OCR_LANGUAGES.keys())),
            )
            return DEFAULT_OCR_LANGUAGE
    return "+".join(dict.fromkeys(parts))


def normalize_ocr_settings(
    ocr_language: Optional[str] = None,
    ocr_dpi: Optional[int] = None,
) -> tuple[str, int]:
    """Validate and normalize OCR language and DPI resolution parameters."""
    lang = (
        validate_ocr_language(ocr_language)
        if ocr_language is not None
        else DEFAULT_OCR_LANGUAGE
    )
    dpi = validate_ocr_dpi(ocr_dpi) if ocr_dpi is not None else DEFAULT_OCR_DPI
    return lang, dpi
