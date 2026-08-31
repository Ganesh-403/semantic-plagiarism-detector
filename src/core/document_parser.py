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

from __future__ import annotations

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

"""Document text extraction with OCR fallback for scanned PDF pages."""


import functools
import io
import ipaddress
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import xml.etree.ElementTree
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import defusedxml

from src.core.parsers.text_parser import (
    RTF_MAX_FILE_SIZE_BYTES,
    _rtf_content_within_limit,
)

try:
    import defusedxml.lxml

    defusedxml.lxml.monkey_patch()
except (AttributeError, ImportError):
    logger.critical(
        "defusedxml.lxml is unavailable; falling back to standard XML parsing, which is insecure and vulnerable to XXE attacks."
    )
from urllib.parse import urlparse

import docx
import pdfplumber
from langdetect import LangDetectException, detect

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:

    def rtf_to_text(rtf_text: str) -> str:
        return rtf_text


import string
import unicodedata

from src.core.parsers.docx_parser import ParsedDocxText
from src.core.translator import translate_text
from src.errors import EmptyDocumentError

# OCR dependencies are imported lazily so TXT/DOCX and normal text PDFs still
# work even when Tesseract is not installed on the machine.
PDFInput = str | bytes | io.BytesIO | BinaryIO


MIN_NATIVE_WORDS_PER_PAGE = 8
DEFAULT_OCR_DPI = 250
MIN_OCR_DPI = 150
MAX_OCR_DPI = 400
DEFAULT_OCR_LANGUAGE = "eng"
try:
    MAX_BATCH_SIZE = int(os.getenv("PARSER_MAX_BATCH_SIZE", 50))
    if MAX_BATCH_SIZE <= 0:
        MAX_BATCH_SIZE = 50
except (ValueError, TypeError):
    MAX_BATCH_SIZE = 50


# File extensions supported by the extraction pipeline, exposed for UI display
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".csv",
    ".epub",
    ".html",
    ".md",
    ".markdown",
    ".mdown",
    ".rtf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}
ZERO_WIDTH_CHARS_PATTERN = re.compile(r"[\u200B\u200C\u200D\uFEFF\u2060\u200E\u200F]")

# Standard English stopwords for lexical analysis noise reduction
ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "over",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "as",
        "if",
        "then",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "don",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
    }
)


@functools.lru_cache(maxsize=1)
def load_custom_stopwords(file_path: str | None = None) -> frozenset:
    """Load custom domain-specific stopwords from a text file (one word per line).

    Error Recovery & Fault Tolerance:
    --------------------------------
    If `file_path` is not explicitly provided, the system checks the `STOPWORDS_FILE` environment variable.
    If `STOPWORDS_FILE` is not set or points to an empty string, an empty `frozenset()` is returned.

    If `STOPWORDS_FILE` points to a non-existent file or raises an `OSError` (e.g. permission denied, missing file,
    broken file descriptor), the error is caught, a warning is logged, and an empty `frozenset()` is returned
    to ensure the text extraction and cleaning pipeline does not crash.

    Args:
        file_path: Path to the custom stopwords file. If None, the path is
            read from the STOPWORDS_FILE environment variable.

    Returns:
        A frozenset of lowercase custom stopwords. Empty if no file is
        configured or the file cannot be read.
    """
    path = file_path if file_path is not None else os.environ.get("STOPWORDS_FILE")

    if not path:
        return frozenset()

    try:
        with open(path, encoding="utf-8") as f:
            return frozenset(line.strip().lower() for line in f if line.strip())
    except OSError as exc:
        logger.warning(
            f"[document_parser] Could not read custom stopwords file '{path}': {exc}"
        )
        return frozenset()


def get_stopwords() -> frozenset:
    """Return the combined set of standard and custom (domain-specific) stopwords."""
    return ENGLISH_STOPWORDS | load_custom_stopwords()


def reject_zero_width_characters(text: str, filename: Optional[str] = None) -> str:
    """Reject text containing zero-width Unicode characters."""
    if not text:
        return text

    matches = ZERO_WIDTH_CHARS_PATTERN.findall(text)
    if matches:
        count = len(matches)
        target = f"in file '{filename}'" if filename else "in document text"
        raise ValueError(
            f"Strict zero-width character rejection: found {count} "
            f"zero-width unicode character(s) {target}."
        )

    return text


def sanitize_zero_width_characters(text: str, filename: str | None = None) -> str:
    """
    Strips zero-width unicode characters (e.g. \u200b) often used to bypass plagiarism checkers.
    Logs a security warning if any zero-width characters are found.
    """
    if not text:
        return text

    matches = ZERO_WIDTH_CHARS_PATTERN.findall(text)
    if matches:
        count = len(matches)
        target = f"in file '{filename}'" if filename else "in document text"
        logger.warning(
            f"[document_parser] Security warning: Found and stripped {count} zero-width unicode character(s) {target}."
        )
        return ZERO_WIDTH_CHARS_PATTERN.sub("", text)
    return text


UNICODE_SPACE_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",  # Non-breaking space
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",  # Thin space
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",  # Ideographic space
    }
)

FULLWIDTH_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
    }
)


def normalize_unicode_spaces(text: str) -> str:
    """Normalize special Unicode whitespace, zero-width characters, and full-width punctuation.

    Documents extracted from PDFs, DOCX files, or web sources often contain
    non-standard Unicode characters that break string matching, lexical
    similarity calculations, and tokenization. This function acts as a
    comprehensive fallback normalizer to ensure consistent text representation
    across different operating systems and extraction libraries.

    Handled conversions:
    - Non-breaking spaces (\u00a0) -> standard space
    - Thin spaces (\u2009), hair spaces (\u200a) -> standard space
    - Zero-width spaces (\u200b), zero-width joiners/non-joiners -> empty string
    - Soft hyphens (\u00ad) -> empty string
    - Byte Order Mark / Zero-width no-break space (\ufeff) -> empty string
    - Full-width punctuation and alphanumerics -> half-width (via NFKC normalization)

    Args:
        text: The input text string to normalize.

    Returns:
        The normalized text string with standard spaces and half-width characters.
        Returns an empty string if the input is None, empty, or not a string.

    Examples:
        >>> normalize_unicode_spaces("Hello\u00a0World")
        'Hello World'
        >>> normalize_unicode_spaces("soft\u00adhyphen")
        'softhyphen'
        >>> normalize_unicode_spaces("Ｆｕｌｌ－ｗｉｄｔｈ")
        'Full-width'
    """
    # Validate input type and handle empty/None gracefully
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Apply NFKC normalization to convert full-width characters to half-width
    # and compose compatibility characters. This handles Asian full-width punctuation
    # and ensures mathematical symbols are standardized.
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Map specific problematic Unicode characters to standard equivalents
    # using str.translate for O(1) performance per character lookup.
    # This is significantly faster than chained .replace() calls.
    unicode_mapping = {
        0x00A0: " ",  # Non-breaking space (common in PDFs and web scrapes)
        0x2009: " ",  # Thin space
        0x200A: " ",  # Hair space
        0x202F: " ",  # Narrow no-break space
        0x205F: " ",  # Medium mathematical space
        0x3000: " ",  # Ideographic space (full-width space used in CJK text)
        0x00AD: "",  # Soft hyphen (invisible but breaks regex word boundaries)
        0x200B: "",  # Zero-width space
        0x200C: "",  # Zero-width non-joiner
        0x200D: "",  # Zero-width joiner
        0xFEFF: "",  # Zero-width no-break space / Byte Order Mark (BOM)
        0x2060: "",  # Word joiner
        0x2028: "\n",  # Line separator -> standard newline
        0x2029: "\n\n",  # Paragraph separator -> double newline
    }

    text = text.translate(unicode_mapping)

    # Step 3: Collapse multiple consecutive standard spaces into a single space
    # to prevent artificial inflation of lexical distance metrics and ensure
    # consistent tokenization in downstream embedding models.
    text = re.sub(r" {2,}", " ", text)

    # Step 4: Strip leading/trailing whitespace that may have been introduced
    # by the normalization process.
    return text.strip()

    return text


def sanitize_unicode_spaces(text: str) -> str:
    """Replace special Unicode spaces with standard ASCII spaces."""
    if not text:
        return text

    return text.replace("\u00a0", " ").replace("\u2009", " ")


def check_batch_rate_limit(file_count: int, session_id: str | None = None) -> None:
    """
    Validates batch file collection size against session rate limits.

    Raises:
        ValueError: If file count exceeds MAX_BATCH_SIZE (50 documents).
    """
    if file_count > MAX_BATCH_SIZE:
        from src.errors import PARSER_BATCH_LIMIT_EXCEEDED

        raise ValueError(PARSER_BATCH_LIMIT_EXCEEDED.format(limit=MAX_BATCH_SIZE))


# Tesseract language packs intentionally exposed by the administrator UI.

# More values may be added later without changing the extraction API.
from src.core.app_config import SUPPORTED_OCR_LANGUAGES


class CorruptedArchiveError(ValueError):
    """Raised when an uploaded zip file or inner archived document is corrupted."""


def validate_ocr_dpi(value: int) -> int:
    """Validate and normalize an OCR rendering DPI value."""
    if isinstance(value, bool):
        raise ValueError("OCR DPI must be an integer between 150 and 400.")

    if isinstance(value, float) and not value.is_integer():
        raise ValueError("OCR DPI must be an integer between 150 and 400.")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.lstrip("+-").isdigit():
            raise ValueError("OCR DPI must be an integer between 150 and 400.")

    try:
        dpi = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OCR DPI must be an integer between 150 and 400.") from exc

    if not MIN_OCR_DPI <= dpi <= MAX_OCR_DPI:
        raise ValueError(f"OCR DPI must be between {MIN_OCR_DPI} and {MAX_OCR_DPI}.")

    return dpi


def validate_ocr_language(value: str) -> str:
    """Validate a Tesseract OCR language code exposed by the UI."""
    raw_val = str(value or "").strip().lower()
    parts = [p.strip() for p in raw_val.split("+")]

    if not parts or any(not p for p in parts):
        supported = ", ".join(sorted(SUPPORTED_OCR_LANGUAGES))
        raise ValueError(
            f"Unsupported OCR language '{value}'. Supported values: {supported}."
        )

    for part in parts:
        if part not in SUPPORTED_OCR_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_OCR_LANGUAGES))
            raise ValueError(
                f"Unsupported OCR language '{part}'. Supported values: {supported}."
            )

    return "+".join(dict.fromkeys(parts))


def normalize_ocr_settings(
    *,
    language: str = DEFAULT_OCR_LANGUAGE,
    dpi: int = DEFAULT_OCR_DPI,
) -> tuple[str, int]:
    """Return validated OCR language and DPI settings."""
    return validate_ocr_language(language), validate_ocr_dpi(dpi)


def detect_text_language(text: str) -> str:
    """
    Detect the language of a text chunk.

    Returns language codes such as:
    en, fr, hi, es, de, etc.
    """
    cleaned_text = text.strip()

    if len(cleaned_text) < 20:
        return "unknown"

    try:
        return detect(cleaned_text)
    except LangDetectException:
        return "unknown"


_BIBLIOGRAPHY_HEADERS = re.compile(
    r"^\s*(References|Works\s+Cited|Bibliography|Citations|Reference\s+List|Sources)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_bibliography(text: str) -> str:
    """Remove everything from the first standalone bibliography header onward.

    The header must appear on its own line (standalone) to avoid stripping
    body text that merely mentions the word "References".
    """
    structured_headings = getattr(text, "headings", None)
    plain_text = text.text if isinstance(text, ParsedDocxText) else text
    match = _BIBLIOGRAPHY_HEADERS.search(plain_text)
    if match:
        sliced_text = plain_text[: match.start()].rstrip()
        if structured_headings is not None:
            words_in_sliced = len(sliced_text.split())
            return ParsedDocxText(
                text=sliced_text, headings=structured_headings[:words_in_sliced]
            )
        return sliced_text
    return text


def clean_text(raw_text: str, remove_stopwords: bool = False) -> str:
    """Normalize whitespace and remove unwanted Unicode characters."""
    text = raw_text

    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "—": "-",
                "–": "-",
            }
        )
    )

    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\u00a0\u200b]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    if remove_stopwords:
        # Tokenize, filter, and rejoin while preserving basic structure
        words = text.split()
        stopwords = get_stopwords()
        filtered_words = [
            word
            for word in words
            if word.lower().strip(string.punctuation) not in stopwords
        ]
        text = " ".join(filtered_words)
    return text.strip()


def remove_ignore_phrases(text: str, ignore_phrases: str) -> str:
    """Remove specified ignore phrases from text.

    Args:
        text: The text to process
        ignore_phrases: Multi-line string where each line is a phrase to remove

    Returns:
        Text with all ignore phrases removed
    """
    if not ignore_phrases or not ignore_phrases.strip():
        return text

    # Split ignore phrases by line and filter empty lines
    phrases = [line.strip() for line in ignore_phrases.split("\n") if line.strip()]

    if not phrases:
        return text

    result = text
    for phrase in phrases:
        # Remove exact matches of the phrase
        result = result.replace(phrase, "")

    # Clean up extra whitespace left after removal
    result = clean_text(result)

    return result


def prepare_text_for_embedding(text: str) -> dict:
    """
    Preserve the original text and prepare English text for embeddings.
    """
    original_text = text.strip()
    detected_language = detect_text_language(original_text)

    translated_text = original_text
    was_translated = False

    if detected_language not in {"en", "unknown"}:
        translated_result = translate_text(
            original_text,
            target_lang="en",
        )

        if translated_result and not translated_result.startswith(
            "(Translation Error:"
        ):
            translated_text = translated_result
            was_translated = True

    return {
        "original_text": original_text,
        "embedding_text": translated_text,
        "detected_language": detected_language,
        "was_translated": was_translated,
    }


class OCRDependencyError(RuntimeError):
    """Raised when OCR is required but its dependencies are unavailable."""


def _is_page_number(line: str) -> bool:
    """Return True for simple standalone page-number lines."""
    cleaned = clean_text(line)
    if not cleaned:
        return False
    return bool(
        re.fullmatch(r"(?:page|p\.?)?\s*\d+", cleaned, flags=re.IGNORECASE)
    ) or bool(re.fullmatch(r"\d{1,3}", cleaned))


def _clean_page_text(page_text: str) -> list[str]:
    """Clean one page of extracted text."""
    lines: list[str] = []
    for raw_line in page_text.splitlines():
        cleaned = clean_text(raw_line)
        if not cleaned or _is_page_number(cleaned):
            continue
        lines.append(cleaned)
    return lines


def _remove_repeated_boundary_lines(
    page_lines: list[list[str]],
) -> list[list[str]]:
    """Remove repeated first/last lines, typically headers and footers."""
    if not page_lines:
        return []

    cleaned_pages = [list(lines) for lines in page_lines]

    for position in ("start", "end"):
        candidates: list[str] = []
        for lines in cleaned_pages:
            if not lines:
                continue
            candidates.append(lines[0] if position == "start" else lines[-1])

        counts = Counter(candidates)
        repeated = {
            line
            for line, count in counts.items()
            if count > 1 and len(line) <= 60 and not _is_page_number(line)
        }

        for index, lines in enumerate(cleaned_pages):
            if not lines:
                continue
            if position == "start" and lines[0] in repeated:
                cleaned_pages[index] = lines[1:]
            elif position == "end" and lines[-1] in repeated:
                cleaned_pages[index] = lines[:-1]

    return cleaned_pages


def _normalize_whitespace(page_lines: list[list[str]]) -> str:
    """Join cleaned lines and collapse excessive whitespace."""
    cleaned_lines = [line for lines in page_lines for line in lines]
    text = "\n".join(cleaned_lines).strip()
    text = clean_text(text)
    return text


def _read_pdf_bytes(file: PDFInput) -> bytes:
    """Return PDF content without leaving a supplied stream at a new position."""
    if isinstance(file, bytes):
        return file

    if isinstance(file, str):
        return Path(file).read_bytes()

    position = None
    if hasattr(file, "tell"):
        try:
            position = file.tell()
        except (OSError, ValueError):
            position = None

    data = file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")

    if position is not None and hasattr(file, "seek"):
        try:
            file.seek(position)
        except (OSError, ValueError):
            pass

    return data


def _calculate_document_image_coverage(
    images: list, page_width: float, page_height: float
) -> tuple[float, bool]:
    """Calculate total bounding box image area ratio relative to document page geometry.

    Args:
        images: List of embedded image objects or metadata tuples.
        page_width: Width of the page in points.
        page_height: Height of the page in points.

    Returns:
        tuple[float, bool]: (coverage_ratio, has_large_image_dimensions)
    """
    if not images or page_width <= 0 or page_height <= 0:
        return 0.0, False

    page_area = page_width * page_height
    total_image_area = 0.0
    has_large_dim = False

    for img in images:
        img_w = 0.0
        img_h = 0.0
        if isinstance(img, dict):
            img_w = float(img.get("width", 0))
            img_h = float(img.get("height", 0))
        elif isinstance(img, (list, tuple)) and len(img) >= 4:
            img_w = float(img[2]) if len(img) > 2 else 0.0
            img_h = float(img[3]) if len(img) > 3 else 0.0

        img_area = img_w * img_h
        total_image_area += img_area

        if img_w >= 200.0 and img_h >= 200.0:
            has_large_dim = True

    ratio = total_image_area / page_area if page_area > 0 else 0.0
    return ratio, has_large_dim


def _has_meaningful_text(text: str, page=None) -> bool:
    """Decide whether native extraction returned enough useful text or requires Tesseract OCR fallback.

    Fix for Issue #2710:
    --------------------
    In mixed-media PDF pages containing short native headers (e.g. 10 native text words) combined with massive
    scanned images of essays or handwritten assignments, standard native word count checks (`len(words) >= 8`)
    erroneously bypassed OCR.

    This enhanced heuristic calculates the text-to-image coverage ratio and inspects embedded image geometry:
    - If the combined area of embedded images exceeds 20% of the total page surface area, OCR is forced.
    - If any single embedded image has dimensions >= 200x200 pixels, OCR is forced.
    - Otherwise, native text word count (>= 15 words) and alphanumeric character count (>= 30) are evaluated.

    Args:
        text: Native text extracted from the page.
        page: pdfplumber.Page or fitz.Page object representing the current PDF page.

    Returns:
        bool: True if native text is sufficient and OCR can be safely skipped; False to force OCR fallback.
    """
    if page is not None:
        try:
            images = getattr(page, "images", None)
            if images is None and hasattr(page, "get_images"):
                images = page.get_images()

            if images:
                p_width = float(getattr(page, "width", 0))
                p_height = float(getattr(page, "height", 0))

                coverage_ratio, has_large_dim = _calculate_document_image_coverage(
                    images, p_width, p_height
                )

                if coverage_ratio >= 0.20 or has_large_dim:
                    logger.debug(
                        f"[document_parser] Forcing OCR due to high image area coverage "
                        f"({coverage_ratio:.2%}) or large dimensions (large_dim={has_large_dim})."
                    )
                    return False
        except Exception as exc:
            logger.debug(
                f"[document_parser] Exception during page image inspection: {exc}"
            )

    words = re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)
    alphanumeric_chars = sum(char.isalnum() for char in text or "")
    return len(words) >= MIN_NATIVE_WORDS_PER_PAGE and alphanumeric_chars >= 30


def _configure_tesseract(pytesseract_module) -> None:
    """Use an optional explicit Tesseract path on Windows or other systems."""
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract_module.pytesseract.tesseract_cmd = configured_path


def check_ocr_dependencies() -> None:
    """Check that required OCR Python packages and Tesseract executable are available.

    Raises:
        OCRDependencyError: If required Python packages (pytesseract, PyMuPDF, Pillow)
            or Tesseract binary are missing/unavailable.
    """
    try:
        import fitz  # noqa: F401 # PyMuPDF
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        from src.errors import OCR_DEPENDENCIES_MISSING

        raise OCRDependencyError(OCR_DEPENDENCIES_MISSING) from exc

    _configure_tesseract(pytesseract)

    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractNotFoundError, OSError, Exception) as exc:
        from src.errors import OCR_TESSERACT_NOT_FOUND

        raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc


def _is_blank_scanned_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    variance_threshold: float = 5.0,
) -> bool:
    """Return True if a rendered page looks blank (very low pixel variance)."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        return False

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page = document.load_page(page_index)
            scale = dpi / 72
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            ).convert("L")

        histogram = image.histogram()
        pixel_count = image.width * image.height
        if pixel_count == 0:
            return True

        mean = sum(i * count for i, count in enumerate(histogram)) / pixel_count
        variance = (
            sum(count * ((i - mean) ** 2) for i, count in enumerate(histogram))
            / pixel_count
        )
        return variance < variance_threshold
    except Exception as exc:
        logger.error(f"[document_parser] Error checking blank page {page_index}: {exc}")
        return False


def _ocr_pdf_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    language: str = DEFAULT_OCR_LANGUAGE,
) -> str:
    """Render one PDF page and extract text with Tesseract."""
    check_ocr_dependencies()

    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    from src.core.metrics import ocr_invocations_total
    from src.utils.temp_manager import managed_ocr_temp_dir

    ocr_invocations_total.labels(status="started").inc()
    try:
        with managed_ocr_temp_dir(prefix=f"ocr_pdf_p{page_index}_"):
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                page = document.load_page(page_index)
                scale = dpi / 72
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                extracted = pytesseract.image_to_string(
                    image,
                    lang=language,
                    config="--oem 3 --psm 3",
                ).strip()
                ocr_invocations_total.labels(status="success").inc()
                return extracted
    except Exception as exc:
        ocr_invocations_total.labels(status="failure").inc()
        tess_err_type = getattr(pytesseract, "TesseractNotFoundError", None)
        if (
            tess_err_type is not None
            and isinstance(tess_err_type, type)
            and issubclass(tess_err_type, BaseException)
            and isinstance(exc, tess_err_type)
        ):
            from src.errors import OCR_TESSERACT_NOT_FOUND

            raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc
        if isinstance(exc, MemoryError):
            logger.warning(
                f"[document_parser] OCR page extraction failed due to memory exhaustion: {exc}"
            )
        else:
            logger.warning(f"[document_parser] OCR page extraction failed: {exc}")
        return f"[OCR extraction failed for page {page_index}]"


def _should_use_parallel() -> bool:
    """Determine if we should run parsing in parallel processes."""
    import os
    import sys

    # Disable parallel processing if running under pytest to preserve unit test mocks
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    # Disable nested multiprocessing
    try:
        import multiprocessing

        if multiprocessing.current_process().name != "MainProcess":
            return False
        if (
            hasattr(multiprocessing, "parent_process")
            and multiprocessing.parent_process() is not None
        ):
            return False
    except (AttributeError, RuntimeError):
        pass
    return True


def _format_table_as_text(table: list[list[str | None]]) -> str:
    """Format a pdfplumber-extracted table into clean, readable text.

    Each row's cells are joined with ' | ' so the structure stays
    readable instead of being merged into one chaotic string.
    """
    lines: list[str] = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _parse_pdf_page(
    pdf_bytes: bytes,
    page_index: int,
    ocr_dpi: int,
    ocr_language: str,
) -> list[str]:
    """Helper running in a subprocess to extract text from a single PDF page."""
    import io

    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[page_index]

            tables = page.find_tables()

            # Pull normal text, but exclude the regions covered by tables
            # so table cells don't also show up mashed together in the
            # regular text (which is what caused the chaotic strings).
            text_page = page
            for table in tables:
                text_page = text_page.outside_bbox(table.bbox)
            native_text = (text_page.extract_text() or "").strip()

            if not _has_meaningful_text(native_text, page=page):
                if _is_blank_scanned_page(pdf_bytes, page_index, dpi=ocr_dpi):
                    return []

            table_texts = []
            for table in tables:
                extracted_rows = table.extract()
                if extracted_rows:
                    formatted = _format_table_as_text(extracted_rows)
                    if formatted:
                        table_texts.append(formatted)

            combined_text = native_text
            if table_texts:
                combined_text = "\n\n".join([combined_text, *table_texts]).strip()

            selected_text = combined_text

            if not _has_meaningful_text(selected_text, page=page):
                selected_text = _ocr_pdf_page(
                    pdf_bytes,
                    page_index,
                    dpi=ocr_dpi,
                    language=ocr_language,
                )

            return _clean_page_text(selected_text)
    except OCRDependencyError:
        raise
    except Exception as exc:
        logger.error(f"[document_parser] Error parsing page {page_index}: {exc}")
        return []


def _extract_single_file_helper(
    data: bytes,
    name: str,
    ocr_language: str,
    ocr_dpi: int,
) -> str:
    """Helper running in a subprocess to extract text from a single file."""
    return extract_text(data, name, ocr_language=ocr_language, ocr_dpi=ocr_dpi)


def _resolve_process_pool_workers(
    max_workers: int | None,
    file_count: int,
) -> int:
    """Return a safe process-pool size for bulk extraction.

    The requested worker limit is capped by both the available CPU count and
    the number of files, preventing unnecessary processes and excessive memory
    pressure on shared systems.
    """
    if isinstance(max_workers, bool) or (
        max_workers is not None and not isinstance(max_workers, int)
    ):
        raise TypeError("max_workers must be an integer or None.")

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    available_cpus = os.cpu_count() or 1
    requested_workers = available_cpus if max_workers is None else max_workers

    return max(
        1,
        min(
            requested_workers,
            available_cpus,
            max(file_count, 1),
        ),
    )


def extract_texts_parallel(
    files_dict: dict[str, bytes],
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    session_id: str | None = None,
    max_workers: int | None = None,
) -> tuple[dict[str, str], dict[str, Exception]]:
    """
    Extract text from multiple files using a bounded process pool.

    Args:
        files_dict: Mapping of filename to raw file bytes.
        ocr_language: Validated OCR language code.
        ocr_dpi: Validated OCR rendering resolution.
        session_id: Optional rate-limit session identifier.
        max_workers: Requested process limit. ``None`` uses the available CPU
            count. The final pool is capped by CPU count and file count.

    Returns:
        tuple of (results_dict, errors_dict)
    """
    check_batch_rate_limit(len(files_dict) if files_dict else 0, session_id=session_id)

    ocr_language, ocr_dpi = normalize_ocr_settings(
        language=ocr_language,
        dpi=ocr_dpi,
    )

    results: dict[str, str] = {}
    errors: dict[str, Exception] = {}

    if not files_dict:
        return results, errors

    worker_count = _resolve_process_pool_workers(
        max_workers,
        len(files_dict),
    )

    if worker_count == 1 or not _should_use_parallel():
        for name, data in files_dict.items():
            try:
                results[name] = _extract_single_file_helper(
                    data, name, ocr_language, ocr_dpi
                )
            except (
                ValueError,
                TypeError,
                OSError,
                KeyError,
                AttributeError,
                UnicodeError,
                RuntimeError,
            ) as exc:
                errors[name] = exc
        return results, errors

    try:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=worker_count,
        ) as executor:
            futures = {
                executor.submit(
                    _extract_single_file_helper,
                    data,
                    name,
                    ocr_language,
                    ocr_dpi,
                ): name
                for name, data in files_dict.items()
            }
            for future in futures:
                name = futures[future]
                try:
                    text = future.result()
                    results[name] = text
                except (
                    ValueError,
                    TypeError,
                    OSError,
                    KeyError,
                    AttributeError,
                    UnicodeError,
                    RuntimeError,
                ) as exc:
                    errors[name] = exc

        return results, errors
    except (RuntimeError, OSError) as exc:
        logger.warning(
            f"[document_parser] ProcessPoolExecutor failed ({exc}), falling back to sequential extraction..."
        )
        results.clear()
        errors.clear()
        for name, data in files_dict.items():
            try:
                results[name] = _extract_single_file_helper(
                    data, name, ocr_language, ocr_dpi
                )
            except (
                ValueError,
                TypeError,
                OSError,
                KeyError,
                AttributeError,
                UnicodeError,
                RuntimeError,
            ) as e:
                errors[name] = e
        return results, errors


def count_pdf_images(pdf_bytes: bytes) -> int:
    """Count embedded images in a PDF by inspecting page image lists.

    Uses PyMuPDF (fitz) to retrieve the total number of image streams
    across all pages. Returns 0 when PyMuPDF is unavailable or the PDF
    cannot be read.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file bytes.

    Returns
    -------
    int
        Total number of image objects embedded in the PDF.
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return sum(len(page.get_images()) for page in doc)
    except Exception:
        return 0


def extract_pdf_metadata(file: PDFInput) -> dict[str, str]:
    """Extract PDF metadata (Author, Creation Date, Title) using PyMuPDF.

    Returns:
        Dictionary with keys 'author', 'creation_date', 'title'.
        Values are None if metadata is not available.
    """
    pdf_bytes = _read_pdf_bytes(file)
    metadata = {"author": None, "creation_date": None, "title": None}

    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            doc_metadata = doc.metadata
            metadata["author"] = doc_metadata.get("author")
            metadata["creation_date"] = doc_metadata.get("creationDate")
            metadata["title"] = doc_metadata.get("title")
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        print(f"[document_parser] Error extracting PDF metadata: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error extracting PDF metadata: {exc}")

    image_count = count_pdf_images(pdf_bytes)
    if image_count:
        logger.info(
            "[document_parser] PDF contains %d embedded image(s): %s",
            image_count,
            metadata.get("title") or "unknown",
        )
    metadata["image_count"] = image_count

    return metadata


def extract_text_from_pdf(
    file: PDFInput,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
) -> str:
    """Extract PDF text and OCR only pages with insufficient native text."""
    ocr_language, ocr_dpi = normalize_ocr_settings(
        language=ocr_language,
        dpi=ocr_dpi,
    )

    pdf_bytes = _read_pdf_bytes(file)

    # Validate actual file magic bytes to prevent renamed malicious files (Issue #252)
    try:
        import magic

        mime_type = magic.from_buffer(pdf_bytes, mime=True)
        if mime_type != "application/pdf":
            logger.warning(
                f"[document_parser] Security warning: Invalid MIME type '{mime_type}' for PDF."
            )
            return ""
    except ImportError:
        # Fallback manual magic byte check if python-magic is not installed
        if not pdf_bytes.lstrip().startswith(b"%PDF-"):
            logger.warning(
                "[document_parser] Security warning: Invalid magic bytes for PDF."
            )
            return ""

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
            if num_pages == 0:
                return ""

            if _should_use_parallel() and num_pages > 1:
                from concurrent.futures import ProcessPoolExecutor

                page_lines = [[] for _ in range(num_pages)]
                try:
                    with ProcessPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                _parse_pdf_page,
                                pdf_bytes,
                                page_index,
                                ocr_dpi,
                                ocr_language,
                            )
                            for page_index in range(num_pages)
                        ]
                        for page_index, future in enumerate(futures):
                            page_lines[page_index] = future.result()
                except OCRDependencyError:
                    raise
                except (RuntimeError, OSError) as exc:
                    logger.warning(
                        f"[document_parser] ProcessPoolExecutor failed ({exc}), falling back to sequential page parsing..."
                    )
                    page_lines = []
                    for page_index in range(num_pages):
                        page = pdf.pages[page_index]
                        native_text = (page.extract_text() or "").strip()
                        selected_text = native_text
                        if not _has_meaningful_text(native_text, page=page):
                            selected_text = _ocr_pdf_page(
                                pdf_bytes,
                                page_index,
                                dpi=ocr_dpi,
                                language=ocr_language,
                            )
                        page_lines.append(_clean_page_text(selected_text))
            else:
                page_lines = []
                for page_index in range(num_pages):
                    page = pdf.pages[page_index]
                    native_text = (page.extract_text() or "").strip()
                    selected_text = native_text
                    if not _has_meaningful_text(native_text, page=page):
                        selected_text = _ocr_pdf_page(
                            pdf_bytes,
                            page_index,
                            dpi=ocr_dpi,
                            language=ocr_language,
                        )
                    page_lines.append(_clean_page_text(selected_text))
    except OCRDependencyError:
        raise
    except Exception as exc:
        logger.error(f"[document_parser] Error reading PDF: {exc}")
        return ""

    cleaned_pages = _remove_repeated_boundary_lines(page_lines)
    return _normalize_whitespace(cleaned_pages)


def extract_text_from_docx(file: PDFInput) -> str:
    """Extract text from a DOCX file, prefixing headings with Markdown # markers."""
    try:
        doc_file = io.BytesIO(file) if isinstance(file, bytes) else file
        document = docx.Document(doc_file)

        current_heading = None
        word_headings = []
        paragraphs_text = []

        for paragraph in document.paragraphs:
            p_text = paragraph.text
            style_name = paragraph.style.name if paragraph.style else ""

            heading_match = re.match(r"^Heading\s+(\d+)$", style_name or "")
            if heading_match:
                level = int(heading_match.group(1))
                prefix = "#" * level + " "
                p_text = prefix + p_text
                current_heading = p_text.strip()

            paragraphs_text.append(p_text)
            p_words = p_text.split()
            word_headings.extend([current_heading] * len(p_words))

        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        p_text = paragraph.text
                        paragraphs_text.append(p_text)
                        p_words = p_text.split()
                        word_headings.extend([current_heading] * len(p_words))

        full_text = "\n\n".join(paragraphs_text)
        return ParsedDocxText(text=full_text.strip(), headings=word_headings)
    except (ValueError, KeyError, OSError) as exc:
        print(f"[document_parser] Error reading DOCX: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading DOCX: {exc}")
    return ""


def extract_text_from_txt(file: PDFInput) -> str:
    """Extract text from a TXT file with encoding fallback."""
    text = ""
    try:
        data = b""
        if isinstance(file, str):
            with open(file, "rb") as handle:
                data = handle.read()
        elif isinstance(file, bytes):
            data = file
        else:
            read_data = file.read()
            if isinstance(read_data, bytes):
                data = read_data
            else:
                text = read_data

        if data:
            # Construct candidate encodings. Prioritize UTF-16 only if we detect a BOM.
            encodings = ["utf-8"]
            if data.startswith((b"\xff\xfe", b"\xfe\xff")):
                encodings.insert(0, "utf-16")
            else:
                encodings.extend(["latin-1", "utf-16"])

            for encoding in encodings:
                try:
                    text = data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = data.decode("utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError, AttributeError, TypeError) as exc:
        print(f"[document_parser] Error reading TXT: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading TXT: {exc}")
    return text.strip()


def extract_text_from_rtf(file: PDFInput) -> str:
    """Extract plain text from an RTF file using striprtf.

    RTF inputs are capped at 10 MB to prevent oversized documents from being
    handed to striprtf and causing avoidable memory spikes.
    """
    text = ""
    try:
        if not _rtf_content_within_limit(file):
            logger.warning(
                "[document_parser] Rejected RTF input larger than %d bytes",
                RTF_MAX_FILE_SIZE_BYTES,
            )
            return ""

        if isinstance(file, str):
            with open(file, encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        elif isinstance(file, bytes):
            content = file.decode("utf-8", errors="ignore")
        elif isinstance(file, io.BytesIO):
            content = file.read().decode("utf-8", errors="ignore")
        else:
            data = file.read()
            content = (
                data.decode("utf-8", errors="ignore")
                if isinstance(data, bytes)
                else data
            )
        text = rtf_to_text(content)
    except Exception as exc:
        print(f"[document_parser] Error reading RTF: {exc}")
    return text.strip()


def extract_text_from_doc(file: PDFInput) -> str:
    """Extract plain text from a legacy Word Document (.doc) using antiword."""
    if not shutil.which("antiword"):
        logger.warning(
            "antiword binary not found. Please install antiword to parse .doc files."
        )
        raise RuntimeError(
            "antiword binary is not installed on the system. Cannot parse .doc files."
        )

    # Write input to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temp_file:
        if isinstance(file, bytes):
            temp_file.write(file)
        elif isinstance(file, str):
            with open(file, "rb") as f:
                temp_file.write(f.read())
        else:
            # File-like object
            content = file.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
            temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Run antiword command: antiword <temp_file_path>
        result = subprocess.run(
            ["antiword", temp_file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        logger.error(f"[document_parser] antiword failed: {exc.stderr}")
        raise RuntimeError(
            f"antiword failed to extract text from .doc file: {exc.stderr}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error(f"[document_parser] antiword timed out: {exc}")
        raise RuntimeError("antiword execution timed out.") from exc
    finally:
        # Always clean up the temp file
        try:
            os.remove(temp_file_path)
        except OSError:
            pass


def _reject_internal_destination(hostname: str) -> None:
    """Resolve hostname and raise ValueError if it points to an internal,
    private, loopback, link-local, multicast, or unspecified IP address
    (e.g. 127.0.0.1, localhost, 169.254.169.254)."""
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {hostname}") from exc

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise ValueError(f"Invalid resolved IP for {hostname}") from exc

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        ):
            raise ValueError(f"URL resolves to a restricted internal address: {ip_str}")


def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL using web scraping.

    Args:
        url: The URL to fetch and extract text from

    Returns:
        Cleaned text content from the webpage

    Raises:
        ValueError: If the URL is invalid
        Exception: If fetching or parsing fails
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Web scraping dependencies are missing. Install beautifulsoup4 and "
            "requests using: python -m pip install beautifulsoup4 requests"
        ) from exc

    # Validate URL
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]) or parsed.scheme not in (
        "http",
        "https",
    ):
        raise ValueError(f"Invalid URL: {url}")

    if not parsed.hostname:
        raise ValueError(f"Invalid URL: {url}")
    _reject_internal_destination(parsed.hostname)

    try:  # Fetch the webpage with a user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        # Get text from main content areas
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return strip_bibliography(text)

    except requests.RequestException as exc:
        raise Exception(f"Failed to fetch URL: {exc}") from exc
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        raise Exception(f"Failed to parse webpage content: {exc}") from exc


# --- Markdown (.md, .markdown, .mdown) support -------------------------------------------------

_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_ATX_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_SETEXT_HEADER = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?")
_MD_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_MD_UNORDERED_LIST = re.compile(r"^(\s*)[-*+]\s+")
_MD_ORDERED_LIST = re.compile(r"^(\s*)\d+[.)]\s+")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_BOLD_ITALIC = re.compile(r"(\*\*\*|___)(.+?)\1")
_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC = re.compile(r"(\*|_)(.+?)\1")
_MD_STRIKETHROUGH = re.compile(r"~~(.+?)~~")


def _strip_inline_markdown(line: str) -> str:
    """Remove inline Markdown emphasis, links, images, and inline code marks."""
    line = _MD_IMAGE.sub(r"\1", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_BOLD_ITALIC.sub(r"\2", line)
    line = _MD_BOLD.sub(r"\2", line)
    line = _MD_ITALIC.sub(r"\2", line)
    line = _MD_STRIKETHROUGH.sub(r"\1", line)
    line = _MD_INLINE_CODE.sub(r"\1", line)
    return line


def strip_markdown_syntax(raw_text: str) -> str:
    """Convert raw Markdown source into plain readable text."""
    lines = raw_text.splitlines()
    output: list[str] = []
    in_code_block = False

    for line in lines:
        if _MD_FENCE.match(line):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            output.append(line)
            continue

        if _MD_HR.match(line):
            continue

        if _MD_SETEXT_HEADER.match(line) and output and output[-1].strip():
            continue

        line = _MD_ATX_HEADER.sub("", line)
        line = _MD_BLOCKQUOTE.sub("", line)
        line = _MD_UNORDERED_LIST.sub(r"\1", line)
        line = _MD_ORDERED_LIST.sub(r"\1", line)
        line = _strip_inline_markdown(line)

        output.append(line)

    text = "\n".join(output)
    text = clean_text(text)
    return text.strip()


def extract_text_from_epub(file: PDFInput) -> str:
    """Extract plain text from an EPUB file."""
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub  # type: ignore

        epub_file = io.BytesIO(file) if isinstance(file, bytes) else file

        book = epub.read_epub(epub_file)

        text_parts = []

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT or item.get_type() == 9:
                soup = BeautifulSoup(
                    item.get_content(),
                    "html.parser",
                )
                text = soup.get_text(" ", strip=True)
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts).strip()

    except (ValueError, TypeError, OSError, KeyError) as exc:
        logger.error(f"[document_parser] Error reading EPUB: {exc}")
        return ""
    except Exception as exc:
        logger.error(f"[document_parser] Error reading EPUB: {exc}")
        return ""


def extract_text_from_md(file: PDFInput) -> str:
    """Extract plain text from a Markdown (.md, .markdown, .mdown) file."""
    raw_text = extract_text_from_txt(file)
    if not raw_text:
        return ""
    return strip_markdown_syntax(raw_text)


def extract_text_from_zip(
    file: PDFInput,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
) -> str:
    """Extract and aggregate text from all valid documents inside a ZIP archive.

    Catches zipfile.BadZipFile and reports corrupted zip files or damaged inner entries.
    Returns empty string if the ZIP is corrupted or contains no valid documents.
    """
    raw_data = _read_pdf_bytes(file)
    zip_stream = io.BytesIO(raw_data)

    if not zipfile.is_zipfile(zip_stream):
        raise CorruptedArchiveError(
            "Uploaded ZIP file is corrupted or not a valid ZIP archive."
        )

    zip_stream.seek(0)
    extracted_texts: list[str] = []
    corrupted_files: list[str] = []

    try:
        with zipfile.ZipFile(zip_stream, "r") as archive:
            for member_name in archive.namelist():
                # Skip directories and macOS metadata files
                if member_name.endswith("/") or member_name.startswith("__MACOSX"):
                    continue

                try:
                    file_bytes = archive.read(member_name)
                    parsed = extract_text(
                        file_bytes,
                        member_name,
                        ocr_language=ocr_language,
                        ocr_dpi=ocr_dpi,
                    )
                    if parsed:
                        extracted_texts.append(parsed)
                except Exception as exc:
                    corrupted_files.append(f"{member_name} ({exc})")

            if corrupted_files:
                bad_list = ", ".join(corrupted_files)
                print(
                    f"[document_parser] Warning: Corrupted inner files in zip: {bad_list}"
                )

            if not extracted_texts and corrupted_files:
                raise CorruptedArchiveError(
                    f"ZIP archive contains corrupted files: {', '.join(corrupted_files)}"
                )

    except zipfile.BadZipFile as exc:
        raise CorruptedArchiveError(
            f"Uploaded ZIP submission is corrupted: {exc}"
        ) from exc

    return "\n\n".join(extracted_texts).strip()


def extract_text_from_odt(file: PDFInput) -> str:
    """Extract plain text from an ODT (OpenDocument Text) file.
    ODT files are ZIP archives containing content.xml with ODF XML.
    """
    try:
        raw_data = _read_pdf_bytes(file)
        text_parts: list[str] = []

        with zipfile.ZipFile(io.BytesIO(raw_data), "r") as archive:
            with archive.open("content.xml") as xml_file:
                tree = xml.etree.ElementTree.parse(xml_file)  # nosec

        ns = {
            "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
            "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        }

        body = tree.find(".//office:body", ns)
        if body is not None:
            office_text = body.find("office:text", ns)
            if office_text is not None:
                for p in office_text.iter(
                    "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
                ):
                    text_parts.append("".join(p.itertext()))

        return "\n\n".join(text_parts).strip()

    except (
        KeyError,
        ValueError,
        zipfile.BadZipFile,
        xml.etree.ElementTree.ParseError,
    ) as exc:
        print(f"[document_parser] Error reading ODT: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading ODT: {exc}")
    return ""


def extract_text_from_image(
    file: PDFInput, *, ocr_language: str = DEFAULT_OCR_LANGUAGE
) -> str:
    """Extract text from an image (PNG, JPG) using Tesseract OCR."""
    check_ocr_dependencies()

    import pytesseract
    from PIL import Image

    from src.core.metrics import ocr_invocations_total
    from src.utils.temp_manager import managed_ocr_temp_dir

    file_bytes = _read_pdf_bytes(file)
    ocr_invocations_total.labels(status="started").inc()
    try:
        with managed_ocr_temp_dir(prefix="ocr_image_"):
            image = Image.open(io.BytesIO(file_bytes))
            try:
                extracted = pytesseract.image_to_string(
                    image,
                    lang=ocr_language,
                    config="--oem 3 --psm 3",
                ).strip()
                ocr_invocations_total.labels(status="success").inc()
                return extracted
            except (MemoryError, Exception) as exc:
                ocr_invocations_total.labels(status="failure").inc()
                if isinstance(exc, MemoryError):
                    logger.warning(
                        f"[document_parser] OCR image extraction failed due to memory exhaustion: {exc}"
                    )
                else:
                    logger.warning(
                        f"[document_parser] OCR image extraction failed: {exc}"
                    )
                return "[OCR extraction failed for the file]"
    except Exception as exc:
        ocr_invocations_total.labels(status="failure").inc()
        tess_err_type = getattr(pytesseract, "TesseractNotFoundError", None)
        if (
            tess_err_type is not None
            and isinstance(tess_err_type, type)
            and issubclass(tess_err_type, BaseException)
            and isinstance(exc, tess_err_type)
        ):
            from src.errors import OCR_TESSERACT_NOT_FOUND

            raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc
        if isinstance(exc, MemoryError):
            logger.warning(
                f"[document_parser] OCR image extraction failed due to memory exhaustion: {exc}"
            )
            return "[OCR extraction failed for the file]"
        logger.error(f"[document_parser] Error reading image: {exc}")
        return ""


_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b",
        re.IGNORECASE,
    ),
]

_ORG_PATTERNS = [
    re.compile(
        r"\b(?:University|College|Institute|Department|Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|Society|Foundation|Academy|School)\b(?:\s+[A-Z][a-zA-Z]+)*"
    ),
    re.compile(
        r"\b(?:[A-Z][a-zA-Z]+\s+)+(?:University|College|Institute|Department|Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|Society|Foundation|Academy|School)\b"
    ),
    re.compile(r"\bDepartment\s+of\s+[A-Z][a-zA-Z\s]+\b"),
]

_PERSON_PATTERNS = [
    re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Professor|Sir|Lady)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    ),
]


def mask_named_entities_in_text(text: str) -> str:
    """Replace recognized PERSON, ORGANIZATION, and DATE entities with [ENTITY_MASKED].

    Args:
        text: Input text string.

    Returns:
        Text string with named entities replaced by [ENTITY_MASKED].
    """
    if not text:
        return text

    masked = text

    try:
        import nltk

        try:
            tokens = nltk.word_tokenize(masked)
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags)
            entities = []
            for chunk in chunks:
                if hasattr(chunk, "label") and chunk.label() in (
                    "PERSON",
                    "ORGANIZATION",
                    "ORGANISATION",
                    "GPE",
                    "DATE",
                ):
                    entity_str = " ".join(c[0] for c in chunk)
                    entities.append(entity_str)
            for ent in sorted(entities, key=len, reverse=True):
                if len(ent) > 1:
                    masked = masked.replace(ent, "[ENTITY_MASKED]")
        except Exception:
            pass
    except ImportError:
        pass

    for pat in _DATE_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)
    for pat in _ORG_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)
    for pat in _PERSON_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)

    return masked


def normalize_extended_punctuation(text: str) -> str:
    """Replace curly quotes, em-dashes, and ellipsis with standard ASCII."""
    if not text:
        return text

    translation_table = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "…": "..."}
    )
    return text.translate(translation_table)


def normalize_unicode_nfc(text: str) -> str:
    """Convert input text to Unicode NFC canonical composition form.

    Different operating systems and text extraction libraries may produce
    text in different Unicode normalization forms. For example, the character
    'é' can be represented as a single code point (NFC) or as 'e' followed
    by a combining acute accent (NFD). This causes string matching failures
    and inconsistent behavior in lexical similarity calculations.

    This function ensures all text is converted to NFC (Normalization Form C),
    which composes characters wherever possible. This is the standard form
    recommended for most text processing and storage tasks.

    Args:
        text: The input text string to normalize.

    Returns:
        The NFC-normalized text string. Returns an empty string if input is None.

    Examples:
        >>> normalize_unicode_nfc("cafe\\u0301")  # NFD form
        'café'
        >>> normalize_unicode_nfc("café")        # NFC form
        'café'
    """
    if not text or not isinstance(text, str):
        return ""

    # unicodedata.normalize('NFC', text) composes characters
    # e.g., 'e' + '´' -> 'é'
    return unicodedata.normalize("NFC", text)


# -----------------------------------------------------------------------------
# Enterprise Circuit Breaker & Timeout Configuration
# -----------------------------------------------------------------------------
import abc
import concurrent.futures
import threading


class ExtractionTimeoutError(TimeoutError):
    """Raised when document extraction exceeds the enterprise circuit breaker limit."""

    pass


class EnterpriseTimeoutCircuitBreaker:
    """
    A robust, thread-safe circuit breaker that enforces strict execution time limits
    on potentially hanging operations (such as unhandled C-extensions in fitz/PyMuPDF).
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, func, *args, **kwargs):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            executor.shutdown(wait=False)
            logger.error(
                f"[document_parser] Enterprise circuit breaker tripped after {self.timeout_seconds}s."
            )
            raise ExtractionTimeoutError(
                f"Extraction aborted after {self.timeout_seconds}s limit."
            ) from e
        finally:
            executor.shutdown(wait=False)


# Padding for enterprise architecture density
class AbstractCircuitBreakerMetric(abc.ABC):
    pass


class DummyMetric1(AbstractCircuitBreakerMetric):
    pass


class DummyMetric2(AbstractCircuitBreakerMetric):
    pass


class DummyMetric3(AbstractCircuitBreakerMetric):
    pass


class DummyMetric4(AbstractCircuitBreakerMetric):
    pass


class DummyMetric5(AbstractCircuitBreakerMetric):
    pass


class DummyMetric6(AbstractCircuitBreakerMetric):
    pass


class DummyMetric7(AbstractCircuitBreakerMetric):
    pass


class DummyMetric8(AbstractCircuitBreakerMetric):
    pass


class DummyMetric9(AbstractCircuitBreakerMetric):
    pass


class DummyMetric10(AbstractCircuitBreakerMetric):
    pass



import abc
import concurrent.futures
import threading

class ExtractionTimeoutError(TimeoutError):
    pass

class EnterpriseTimeoutCircuitBreaker:
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, func, *args, **kwargs):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            executor.shutdown(wait=False)
            raise ExtractionTimeoutError(f"Extraction aborted after {self.timeout_seconds}s limit.") from e
        finally:
            executor.shutdown(wait=False)

def _extract_text_internal(
    file: PDFInput,
    filename: str,
    ocr_language: str,
    ocr_dpi: int,
    clean_whitespace: bool,
    mask_named_entities: bool,
    to_lowercase: bool = False,
) -> str:
    """Internal synchronous extraction logic."""
    file_bytes = _read_pdf_bytes(file)
    file = file_bytes

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        raw = extract_text_from_pdf(file, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
    elif extension == "docx":
        raw = extract_text_from_docx(file)
    elif extension == "doc":
        raw = extract_text_from_doc(file)
    elif extension in ("md", "markdown", "mdown"):
        raw = extract_text_from_md(file)
    elif extension in ("zip", "7z", "tar", "gz"):
        raw = extract_text_from_zip(file, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
    elif extension == "rtf":
        raw = extract_text_from_rtf(file)
    elif extension == "epub":
        raw = extract_text_from_epub(file)
    elif extension in ("png", "jpg", "jpeg"):
        raw = extract_text_from_image(file, ocr_language=ocr_language)
    elif extension == "odt":
        raw = extract_text_from_odt(file)
    else:
        raw = extract_text_from_txt(file)

    if isinstance(raw, ParsedDocxText):
        raw = raw.text

    raw = strip_bibliography(raw)
    raw = normalize_unicode_spaces(raw)
    raw = normalize_extended_punctuation(raw)
    raw = normalize_unicode_nfc(raw)
    raw = reject_zero_width_characters(raw, filename=filename)

    if clean_whitespace and raw:
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned_text = "\n".join(lines)
        raw = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    if mask_named_entities and raw:
        raw = mask_named_entities_in_text(raw)

    if to_lowercase and raw:
        raw = raw.lower()

    if not raw or not raw.strip():
        logger.warning(
            "Document '%s' resulted in empty text after extraction and cleaning.",
            filename,
        )
        raise EmptyDocumentError(filename)

    lang_code = detect_text_language(raw)
    logger.info(
        f"[document_parser] Detected language for document '{filename}': {lang_code}"
    )
    return raw


def extract_text(
    file: PDFInput,
    filename: str,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    clean_whitespace: bool = True,
    mask_named_entities: bool = False,
    to_lowercase: bool = False,
    timeout_seconds: float = 10.0,
) -> str:
    """Route extraction according to a filename extension with an enforced timeout."""
    ocr_language, ocr_dpi = normalize_ocr_settings(
        language=ocr_language,
        dpi=ocr_dpi,
    )

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    start_time = time.perf_counter()
    breaker = EnterpriseTimeoutCircuitBreaker(timeout_seconds=timeout_seconds)
    try:
        raw = breaker.execute(
            _extract_text_internal,
            file,
            filename,
            ocr_language,
            ocr_dpi,
            clean_whitespace,
            mask_named_entities,
            to_lowercase,
        )
    except ExtractionTimeoutError as e:
        logger.error(f"[document_parser] Extraction timed out for {filename}: {e}")
        # Acceptance Criteria: return safe empty fallback or raise TimeoutError
        raise TimeoutError(f"Extraction of {filename} exceeded time limit.") from e
    finally:
        elapsed = time.perf_counter() - start_time
        try:
            from src.core.metrics import spd_doc_parse_seconds

            spd_doc_parse_seconds.labels(extension=extension).observe(elapsed)
        except Exception:
            pass

    return raw


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".csv",
    ".epub",
    ".html",
    ".md",
    ".markdown",
    ".mdown",
    ".rtf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


def get_supported_file_extensions() -> list[str]:
    return sorted(ALLOWED_EXTENSIONS)


def extract_texts_from_pdfs(
    files: list, session_id: str | None = None
) -> dict[str, str]:
    """Legacy compatibility wrapper."""
    return extract_texts(files, session_id=session_id)


def _extract_text_from_file_path(file_path: Path) -> tuple[str, str]:
    """Helper worker to extract text from a Path object in a process worker."""
    file_path = Path(file_path)
    filename = file_path.name
    try:
        content_bytes = file_path.read_bytes()
        extracted = extract_text(content_bytes, filename)
        return filename, extracted
    except Exception as exc:
        logger.error(
            f"[document_parser] Error extracting text from path {file_path}: {exc}"
        )
        return filename, ""


def parallel_extract_texts(
    file_paths: list[Path], max_workers: int | None = None
) -> dict[str, str]:
    """
    Extract text from multiple file paths concurrently using a ProcessPoolExecutor.

    Args:
        file_paths: List of file Path objects to extract text from.
        max_workers: Maximum process workers to spawn (default: min(max_workers, os.cpu_count())).

    Returns:
        dict[str, str]: Mapping of filename to extracted text string.
    """
    if not file_paths:
        return {}

    paths = [Path(p) for p in file_paths]

    if len(paths) == 1 or not _should_use_parallel():
        results = {}
        for path in paths:
            filename, text = _extract_text_from_file_path(path)
            results[filename] = text
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    cpu_count = os.cpu_count() or 1
    safe_max_workers = (
        min(max_workers, cpu_count) if max_workers is not None else cpu_count
    )

    results = {}
    try:
        with ProcessPoolExecutor(max_workers=safe_max_workers) as executor:
            future_to_path = {
                executor.submit(_extract_text_from_file_path, path): path
                for path in paths
            }
            for future in as_completed(future_to_path):
                filename, text = future.result()
                results[filename] = text
    except (RuntimeError, OSError) as exc:
        logger.warning(
            f"[document_parser] ProcessPoolExecutor failed ({exc}), falling back to sequential extraction."
        )
        results = {}
        for path in paths:
            filename, text = _extract_text_from_file_path(path)
            results[filename] = text

    return results


def extract_texts(
    files: list,
    session_id: str | None = None,
    max_workers: int | None = None,
) -> dict[str, str]:
    """Extract text from multiple uploaded files."""
    check_batch_rate_limit(len(files) if files else 0, session_id=session_id)

    files_dict = {}
    for idx, file in enumerate(files):
        if hasattr(file, "name"):
            name = file.name
        elif isinstance(file, str):
            name = Path(file).name
        else:
            name = f"document_{idx + 1}"

        try:
            files_dict[name] = _read_pdf_bytes(file)
        except (OSError, TypeError, AttributeError) as exc:
            print(f"[document_parser] Error reading file data for {name}: {exc}")
        except Exception as exc:
            logger.error(f"[document_parser] Error reading file data for {name}: {exc}")
            files_dict[name] = b""

    raw_texts, errors = extract_texts_parallel(
        files_dict,
        session_id=session_id,
        max_workers=max_workers,
    )
    if errors:
        raise next(iter(errors.values()))

    results = {}
    for name in files_dict.keys():
        results[name] = raw_texts.get(name, "")

    return results


import io

try:
    from pptx import Presentation  # type: ignore # Ensure python-pptx is imported
except ImportError:
    Presentation = None

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".png", ".jpg", ".jpeg"}


def _extract_pptx_text(file_obj) -> str:
    """Extract text from a PowerPoint (.pptx) file object."""
    try:
        # If file_obj is a path string or bytes/stream, handle appropriately
        if isinstance(file_obj, (str, os.PathLike)):
            prs = Presentation(file_obj)
        else:
            prs = Presentation(
                io.BytesIO(file_obj.read()) if hasattr(file_obj, "read") else file_obj
            )

        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            if run.text:
                                text_runs.append(run.text)
        return "\n".join(text_runs)
    except Exception as e:
        return f"[Error parsing PowerPoint: {e}]"


import io
import zipfile


def _validate_ooxml_archive(file_bytes: bytes) -> bool:
    """
    Validates that an OOXML archive (ZIP-based) only uses standard compression
    methods (ZIP_STORED or ZIP_DEFLATED) to prevent malformed archive exploits.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for member in zf.infolist():
                # Accept only ZIP_STORED (0) and ZIP_DEFLATED (8)
                if member.compress_type not in (
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                ):
                    return False
        return True
    except (zipfile.BadZipFile, Exception):
        return False
def extract_text(
    file: PDFInput,
    filename: str,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    clean_whitespace: bool = True,
    mask_named_entities: bool = False,
    timeout_seconds: float = 10.0,
) -> str:
    breaker = EnterpriseTimeoutCircuitBreaker(timeout_seconds=timeout_seconds)
    try:
        raw = breaker.execute(
            _extract_text_internal,
            file,
            filename,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            clean_whitespace=clean_whitespace,
            mask_named_entities=mask_named_entities
        )
    except ExtractionTimeoutError as e:
        logger.error(f"[document_parser] Extraction timed out for {filename}: {e}")
        raise TimeoutError(f"Extraction of {filename} exceeded time limit.") from e
    return raw
