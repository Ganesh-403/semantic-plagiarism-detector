"""src/core/parsers/dispatch.py - Main text extraction routing and parallel dispatchers."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from src.core.parsers.cleaners import (
    detect_text_language,
    normalize_extended_punctuation,
    normalize_unicode_nfc,
    normalize_unicode_spaces,
    sanitize_zero_width_characters,
    strip_bibliography,
)
from src.core.parsers.common import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    PDFInput,
    check_batch_rate_limit,
    normalize_ocr_settings,
)
from src.core.parsers.docx_parser import (
    ParsedDocxText,
    extract_text_from_doc,
    extract_text_from_docx,
)
from src.core.parsers.ocr_parser import extract_text_from_image
from src.core.parsers.pdf_parser import (
    _read_pdf_bytes,
    _should_use_parallel,
    extract_text_from_pdf,
    extract_texts_parallel,
)
from src.core.parsers.text_parser import (
    extract_text_from_epub,
    extract_text_from_md,
    extract_text_from_odt,
    extract_text_from_rtf,
    extract_text_from_txt,
    extract_text_from_zip,
)

logger = logging.getLogger(__name__)

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
    """Return sorted list of supported file extensions."""
    return sorted(ALLOWED_EXTENSIONS)


def extract_text(
    file: PDFInput,
    filename: str,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    to_lowercase: bool = False,
) -> str:
    """Route extraction according to a filename extension."""
    ocr_language, ocr_dpi = normalize_ocr_settings(
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
    )

    file_bytes = _read_pdf_bytes(file)
    from src.security.mime_validator import validate_mime_type

    if not validate_mime_type(file_bytes, filename):
        logger.warning(
            f"[document_parser] Security warning: Rejected file '{filename}' "
            f"because its MIME type / magic bytes do not match its file extension."
        )
        return ""
    file = file_bytes

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    start_time = time.perf_counter()
    try:
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

        # ParsedDocxText is an internal structured result; the public extraction
        # API continues to return plain text.
        if isinstance(raw, ParsedDocxText):
            raw = raw.text

        raw = strip_bibliography(raw)
        raw = normalize_unicode_spaces(raw)
        raw = normalize_extended_punctuation(raw)
        raw = normalize_unicode_nfc(raw)
        raw = sanitize_zero_width_characters(raw, filename=filename)
        lang_code = detect_text_language(raw)

        if to_lowercase:
            raw = raw.lower()

        logger.info(
            f"[document_parser] Detected language for document '{filename}': {lang_code}"
        )
        return raw
    finally:
        elapsed = time.perf_counter() - start_time
        try:
            from src.core.metrics import spd_doc_parse_seconds

            spd_doc_parse_seconds.labels(extension=extension).observe(elapsed)
        except Exception:
            pass


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
    """Extract text from multiple file paths concurrently using a ProcessPoolExecutor."""
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
    session_id: Optional[str] = None,
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
        except Exception as exc:
            logger.error(f"[document_parser] Error reading file data for {name}: {exc}")
            files_dict[name] = b""

    raw_texts, errors = extract_texts_parallel(
        files_dict,
        session_id=session_id,
        max_workers=max_workers,
    )
    results = {}
    for name in files_dict.keys():
        results[name] = raw_texts.get(name, "")

    return results
