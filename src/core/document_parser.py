"""Document text extraction with OCR fallback for scanned PDF pages."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

from urllib.parse import urlparse

import docx

import pdfplumber
from langdetect import LangDetectException, detect
from striprtf.striprtf import rtf_to_text

logger = logging.getLogger(__name__)
from src.core.translator import translate_text

# OCR dependencies are imported lazily so TXT/DOCX and normal text PDFs still
# work even when Tesseract is not installed on the machine.
PDFInput = Union[str, bytes, io.BytesIO, BinaryIO]

MIN_NATIVE_WORDS_PER_PAGE = 8
DEFAULT_OCR_DPI = 250
MIN_OCR_DPI = 150
MAX_OCR_DPI = 400
DEFAULT_OCR_LANGUAGE = "eng"
MAX_BATCH_SIZE = 50


def check_batch_rate_limit(file_count: int, session_id: Optional[str] = None) -> None:
    """
    Validates batch file collection size against session rate limits.

    Raises:
        ValueError: If file count exceeds MAX_BATCH_SIZE (50 documents).
    """
    if file_count > MAX_BATCH_SIZE:
        from src.errors import PARSER_BATCH_LIMIT_EXCEEDED

        raise ValueError(
            PARSER_BATCH_LIMIT_EXCEEDED.format(limit=MAX_BATCH_SIZE)
        )


# Tesseract language packs intentionally exposed by the administrator UI.

# More values may be added later without changing the extraction API.
SUPPORTED_OCR_LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
}


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
    language = str(value or "").strip().lower()

    if language not in SUPPORTED_OCR_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_OCR_LANGUAGES))
        raise ValueError(
            f"Unsupported OCR language '{language or value}'. "
            f"Supported values: {supported}."
        )

    return language


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
    """Remove everything from the first bibliography header onward.

    The header must appear on its own line (standalone) to avoid stripping
    body text that merely mentions the word "References".
    """
    match = _BIBLIOGRAPHY_HEADERS.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove unwanted Unicode characters."""
    text = raw_text
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\u00a0\u200b]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

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


def _clean_page_text(page_text: str) -> List[str]:
    """Clean one page of extracted text."""
    lines: List[str] = []
    for raw_line in page_text.splitlines():
        cleaned = clean_text(raw_line)
        if not cleaned or _is_page_number(cleaned):
            continue
        lines.append(cleaned)
    return lines


def _remove_repeated_boundary_lines(
    page_lines: List[List[str]],
) -> List[List[str]]:
    """Remove repeated first/last lines, typically headers and footers."""
    if not page_lines:
        return []

    cleaned_pages = [list(lines) for lines in page_lines]

    for position in ("start", "end"):
        candidates: List[str] = []
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


def _normalize_whitespace(page_lines: List[List[str]]) -> str:
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


def _has_meaningful_text(text: str) -> bool:
    """Decide whether native extraction returned enough useful text."""
    words = re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)
    alphanumeric_chars = sum(char.isalnum() for char in text or "")
    return len(words) >= MIN_NATIVE_WORDS_PER_PAGE and alphanumeric_chars >= 30


def _configure_tesseract(pytesseract_module) -> None:
    """Use an optional explicit Tesseract path on Windows or other systems."""
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract_module.pytesseract.tesseract_cmd = configured_path


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
        return False    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    language: str = DEFAULT_OCR_LANGUAGE,
) -> str:
    """Render one PDF page and extract text with Tesseract."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        from src.errors import OCR_DEPENDENCIES_MISSING

        raise OCRDependencyError(OCR_DEPENDENCIES_MISSING) from exc

    _configure_tesseract(pytesseract)

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
            )
            return pytesseract.image_to_string(
                image,
                lang=language,
                config="--oem 3 --psm 3",
            ).strip()
    except pytesseract.TesseractNotFoundError as exc:
        from src.errors import OCR_TESSERACT_NOT_FOUND

        raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc


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


def _format_table_as_text(table: List[List[Optional[str]]]) -> str:
    """Format a pdfplumber-extracted table into clean, readable text.

    Each row's cells are joined with ' | ' so the structure stays
    readable instead of being merged into one chaotic string.
    """
    lines: List[str] = []
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
) -> List[str]:
    """Helper running in a subprocess to extract text from a single PDF page."""
    import io

    import pdfplumber

try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[page_index]


            if not _has_meaningful_text(native_text):
                if _is_blank_scanned_page(pdf_bytes, page_index, dpi=ocr_dpi):
                    return []


            tables = page.find_tables()

            # Pull normal text, but exclude the regions covered by tables
            # so table cells don't also show up mashed together in the
            # regular text (which is what caused the chaotic strings).
            text_page = page
            for table in tables:
                text_page = text_page.outside_bbox(table.bbox)
            native_text = (text_page.extract_text() or "").strip()

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

            if not _has_meaningful_text(selected_text):

                selected_text = _ocr_pdf_page(
                    pdf_bytes,
                    page_index,
                    dpi=ocr_dpi,
                    language=ocr_language,
                )

            return _clean_page_text(selected_text)    except OCRDependencyError:
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


def extract_texts_parallel(
    files_dict: Dict[str, bytes],
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    session_id: Optional[str] = None,
) -> tuple[Dict[str, str], Dict[str, Exception]]:
    """
    Extract text from multiple files in parallel using ProcessPoolExecutor.

    Returns:
        tuple of (results_dict, errors_dict)
    """
    check_batch_rate_limit(len(files_dict) if files_dict else 0, session_id=session_id)

    ocr_language, ocr_dpi = normalize_ocr_settings(
        language=ocr_language,
        dpi=ocr_dpi,
    )

    results: Dict[str, str] = {}
    errors: Dict[str, Exception] = {}

    if not files_dict:
        return results, errors

    if len(files_dict) == 1 or not _should_use_parallel():
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

        with ProcessPoolExecutor() as executor:
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


def extract_pdf_metadata(file: PDFInput) -> Dict[str, str]:
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

    return metadata


def extract_text_from_pdf(
    file: PDFInput,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
) -> str:
    """Extract PDF text and OCR only pages with insufficient native text.

    Text-based PDFs continue to use pdfplumber. Fully scanned and mixed PDFs
    are handled page by page, allowing OCR results to enter the unchanged
    chunking, embedding and FAISS pipeline.
    """
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

    page_lines: List[List[str]] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
    except Exception as exc:
        logger.error(f"[document_parser] Error reading PDF: {exc}")
        return ""

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
            page_lines = [
                _parse_pdf_page(
                    pdf_bytes,
                    page_index,
                    ocr_dpi,
                    ocr_language,
                )
                for page_index in range(num_pages)
            ]
    else:
        page_lines = [
            _parse_pdf_page(
                pdf_bytes,
                page_index,
                ocr_dpi,
                ocr_language,
            )
            for page_index in range(num_pages)
        ]

    if not page_lines:
        return ""

    cleaned_pages = _remove_repeated_boundary_lines(page_lines)
    return _normalize_whitespace(cleaned_pages)


def extract_text_from_docx(file: PDFInput) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc_file = io.BytesIO(file) if isinstance(file, bytes) else file
        document = docx.Document(doc_file)
        text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
    except (ValueError, KeyError, OSError) as exc:
        print(f"[document_parser] Error reading DOCX: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading DOCX: {exc}")
    return text.strip()


def extract_text_from_txt(file: PDFInput) -> str:
    """Extract text from a TXT file with UTF-8 fallback."""
    text = ""
    try:
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        elif isinstance(file, bytes):
            text = file.decode("utf-8", errors="ignore")
        else:
            data = file.read()
            text = (
                data.decode("utf-8", errors="ignore")
                if isinstance(data, bytes)
                else data
            )
    except (OSError, UnicodeDecodeError, AttributeError, TypeError) as exc:
        print(f"[document_parser] Error reading TXT: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading TXT: {exc}")
    return text.strip()


def extract_text_from_rtf(file: PDFInput) -> str:
    """Extract plain text from an RTF file using striprtf."""
    text = ""
    try:
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8", errors="ignore") as handle:
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

    try:
        # Fetch the webpage with a user agent to avoid being blocked
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


# --- Markdown (.md) support -------------------------------------------------

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
    """Convert raw Markdown source into plain readable text.

    Fenced code block contents are preserved as-is (fence markers removed);
    headers, lists, blockquotes, horizontal rules, links, images, and
    emphasis markers are stripped down to their underlying text.
    """
    lines = raw_text.splitlines()
    output: List[str] = []
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
        from ebooklib import epub

        epub_file = io.BytesIO(file) if isinstance(file, bytes) else file

        book = epub.read_epub(epub_file)

        text_parts = []

        for item in book.get_items():
            if item.get_type() == 9:
                soup = BeautifulSoup(
                    item.get_content(),
                    "html.parser",
                )

                text_parts.append(soup.get_text(" ", strip=True))

        return "\n\n".join(text_parts).strip()

    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"[document_parser] Error reading EPUB: {exc}")
    except Exception as exc:
        logger.error(f"[document_parser] Error reading EPUB: {exc}")
        return ""


def extract_text_from_md(file: PDFInput) -> str:
    """Extract plain text from a Markdown (.md) file."""
    raw_text = extract_text_from_txt(file)
    if not raw_text:
        return ""
    return strip_markdown_syntax(raw_text)


def extract_text_from_image(
    file: PDFInput, *, ocr_language: str = DEFAULT_OCR_LANGUAGE
) -> str:
    """Extract text from an image (PNG, JPG) using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        from src.errors import OCR_DEPENDENCIES_MISSING
        raise OCRDependencyError(OCR_DEPENDENCIES_MISSING) from exc

    _configure_tesseract(pytesseract)

    file_bytes = _read_pdf_bytes(file)
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(
            image,
            lang=ocr_language,
            config="--oem 3 --psm 3",
        ).strip()
    except pytesseract.TesseractNotFoundError as exc:
        from src.errors import OCR_TESSERACT_NOT_FOUND
        raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc
    except Exception as exc:
        logger.error(f"[document_parser] Error reading image: {exc}")
        return ""


def extract_text(
    file: PDFInput,
    filename: str,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_dpi: int = DEFAULT_OCR_DPI,
) -> str:
    """Route extraction according to a filename extension."""
    ocr_language, ocr_dpi = normalize_ocr_settings(
        language=ocr_language,
        dpi=ocr_dpi,
    )

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        raw = extract_text_from_pdf(file, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
    elif extension == "docx":
        raw = extract_text_from_docx(file)
    elif extension == "doc":
        raw = extract_text_from_doc(file)
    elif extension == "md":
        raw = extract_text_from_md(file)

    elif extension == "rtf":
        raw = extract_text_from_rtf(file)

    elif extension == "epub":
        raw = extract_text_from_epub(file)
    elif extension in ("png", "jpg", "jpeg"):
        raw = extract_text_from_image(file, ocr_language=ocr_language)
    else:
        raw = extract_text_from_txt(file)

    return strip_bibliography(raw)


def extract_texts_from_pdfs(files: list, session_id: Optional[str] = None) -> Dict[str, str]:
    """Legacy compatibility wrapper."""
    return extract_texts(files, session_id=session_id)


def extract_texts(files: list, session_id: Optional[str] = None) -> Dict[str, str]:
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

    raw_texts, errors = extract_texts_parallel(files_dict, session_id=session_id)
    if errors:
        raise next(iter(errors.values()))

    results = {}
    for name in files_dict.keys():
        results[name] = raw_texts.get(name, "")

    return results
