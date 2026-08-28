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

"""src/core/parsers/pdf_parser.py - PDF document text extraction, page parsing, table extraction, and metadata strategies."""

import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

from src.core.parsers.common import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    PDFInput,
    check_batch_rate_limit,
    normalize_ocr_settings,
)
from src.core.parsers.ocr_parser import (
    OCRDependencyError,
    _is_blank_scanned_page,
    _ocr_pdf_page,
)

logger = logging.getLogger(__name__)

MIN_NATIVE_WORDS_PER_PAGE = 15


def _read_pdf_bytes(file: PDFInput) -> bytes:
    """Normalize PDFInput types into raw bytes."""
    if isinstance(file, bytes):
        return file
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return f.read()
    if hasattr(file, "read"):
        content = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        return content
    raise TypeError(f"Unsupported input type for PDF extraction: {type(file)}")


def _is_page_number(line: str) -> bool:
    """Return True if a line consists only of page number patterns."""
    line = line.strip().lower()
    if not line:
        return False
    if line.isdigit():
        return True
    return bool(
        re.match(
            r"^(?:page\s*\d+(?:\s*of\s*\d+)?|\d+\s*/\s*\d+|-?\s*\d+\s*-?)$",
            line,
        )
    )


def _clean_page_text(page_text: str) -> list[str]:
    """Split page text into clean lines while removing standalone page numbers."""
    if not page_text:
        return []
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    return [line for line in lines if not _is_page_number(line)]


def _remove_repeated_boundary_lines(
    page_lines: list[list[str]],
) -> list[list[str]]:
    """Identify and remove repeated header/footer lines occurring across pages."""
    if len(page_lines) < 2:
        return page_lines

    header_candidates: dict[str, int] = {}
    footer_candidates: dict[str, int] = {}

    for lines in page_lines:
        if not lines:
            continue
        first_line = lines[0]
        header_candidates[first_line] = header_candidates.get(first_line, 0) + 1
        last_line = lines[-1]
        footer_candidates[last_line] = footer_candidates.get(last_line, 0) + 1

    total_pages = len(page_lines)
    min_occurrence = max(2, int(total_pages * 0.6))

    repeated_headers = {
        line for line, count in header_candidates.items() if count >= min_occurrence
    }
    repeated_footers = {
        line for line, count in footer_candidates.items() if count >= min_occurrence
    }

    cleaned_pages: list[list[str]] = []
    for lines in page_lines:
        if not lines:
            cleaned_pages.append([])
            continue
        cleaned = list(lines)
        if cleaned and cleaned[0] in repeated_headers:
            cleaned.pop(0)
        if cleaned and cleaned[-1] in repeated_footers:
            cleaned.pop(-1)
        cleaned_pages.append(cleaned)

    return cleaned_pages


def _normalize_whitespace(page_lines: list[list[str]]) -> str:
    """Rejoin cleaned page lines into a standardized document string.

    Performs normalization on page boundary newlines, tab characters, multiple spaces,
    and trailing line breaks to ensure consistent text format prior to plagiarism detection.

    Args:
        page_lines: Nested list of strings representing extracted lines grouped by page.

    Returns:
        str: Normalized combined text document string.
    """
    page_strings = ["\n".join(lines) for lines in page_lines if lines]
    raw_document = "\n\n".join(page_strings)
    raw_document = re.sub(r"[ \t]+", " ", raw_document)
    raw_document = re.sub(r"\n{3,}", "\n\n", raw_document)
    return raw_document.strip()


def _calculate_image_area_coverage(
    images: list, page_width: float, page_height: float
) -> tuple[float, bool]:
    """Calculate total bounding box image area ratio relative to page geometry.

    Args:
        images: List of embedded image objects or metadata tuples.
        page_width: Width of the PDF page in points.
        page_height: Height of the PDF page in points.

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
    """Evaluate whether native text extraction on a PDF page is sufficient or requires Tesseract OCR fallback.

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

                coverage_ratio, has_large_dim = _calculate_image_area_coverage(
                    images, p_width, p_height
                )

                if coverage_ratio >= 0.20 or has_large_dim:
                    logger.debug(
                        f"[pdf_parser] Forcing OCR due to high image area coverage "
                        f"({coverage_ratio:.2%}) or large dimensions (large_dim={has_large_dim})."
                    )
                    return False
        except Exception as exc:
            logger.debug(f"[pdf_parser] Exception during page image inspection: {exc}")

    if not text:
        return False

    words = text.split()
    alphanumeric_chars = sum(1 for c in text if c.isalnum())
    return len(words) >= MIN_NATIVE_WORDS_PER_PAGE and alphanumeric_chars >= 30


def _should_use_parallel() -> bool:
    """Check if multiprocessing should be enabled."""
    return os.cpu_count() is not None and os.cpu_count() > 1


def _format_table_as_text(table: list[list[Optional[str]]]) -> str:
    """Format a pdfplumber-extracted table into clean, readable text."""
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
    """Extract text from a single PDF page."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[page_index]
            tables = page.find_tables()
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
    from src.core.parsers.dispatch import extract_text

    return extract_text(data, name, ocr_language=ocr_language, ocr_dpi=ocr_dpi)


def _resolve_process_pool_workers(
    max_workers: int | None,
    file_count: int,
) -> int:
    """Return a safe process-pool size for bulk extraction."""
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
    session_id: Optional[str] = None,
    max_workers: int | None = None,
) -> tuple[dict[str, str], dict[str, Exception]]:
    """Extract text from multiple files using a bounded process pool."""
    check_batch_rate_limit(len(files_dict) if files_dict else 0, session_id=session_id)

    ocr_language, ocr_dpi = normalize_ocr_settings(
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
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
    """Count embedded images in a PDF by inspecting page image lists."""
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return sum(len(page.get_images()) for page in doc)
    except Exception:
        return 0


def extract_pdf_metadata(file: PDFInput) -> dict[str, str]:
    """Extract PDF metadata (Author, Creation Date, Title) using PyMuPDF."""
    pdf_bytes = _read_pdf_bytes(file)
    metadata = {"author": None, "creation_date": None, "title": None}

    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            doc_metadata = doc.metadata
            metadata["author"] = doc_metadata.get("author")
            metadata["creation_date"] = doc_metadata.get("creationDate")
            metadata["title"] = doc_metadata.get("title")
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
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
    )

    pdf_bytes = _read_pdf_bytes(file)

    try:
        import magic

        mime_type = magic.from_buffer(pdf_bytes, mime=True)
        if mime_type != "application/pdf":
            logger.warning(
                f"[document_parser] Security warning: Invalid MIME type '{mime_type}' for PDF."
            )
            return ""
    except ImportError:
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


def extract_texts_from_pdfs(
    files: list,
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Extract text from multiple PDF files."""
    check_batch_rate_limit(len(files) if files else 0, session_id=session_id)
    results: dict[str, str] = {}
    for item in files:
        if isinstance(item, tuple):
            name, content = item
        else:
            name, content = str(item), item
        results[name] = extract_text_from_pdf(content)
    return results
