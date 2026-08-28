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

"""src/core/parsers/text_parser.py - Text, RTF, ZIP, EPUB, Markdown, ODT, and URL parsing strategies."""

import io
import logging
import os
import xml.etree.ElementTree
import zipfile
from urllib.parse import urlparse

from src.core.parsers.cleaners import strip_bibliography, strip_markdown_syntax
from src.core.parsers.common import DEFAULT_OCR_DPI, DEFAULT_OCR_LANGUAGE, PDFInput
from src.core.parsers.pdf_parser import _read_pdf_bytes
from src.exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)

RTF_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def _rtf_content_within_limit(file: PDFInput) -> bool:
    """Return whether an RTF input can be safely passed to striprtf.

    Check known sizes before reading/decoding the payload so oversized RTF
    files are rejected before striprtf can allocate parser state for them.
    Seekable streams are inspected without consuming their current position.
    """
    if isinstance(file, str):
        return os.path.getsize(file) <= RTF_MAX_FILE_SIZE_BYTES
    if isinstance(file, bytes):
        return len(file) <= RTF_MAX_FILE_SIZE_BYTES
    if isinstance(file, io.BytesIO):
        return file.getbuffer().nbytes <= RTF_MAX_FILE_SIZE_BYTES

    try:
        current = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(current)
        return size <= RTF_MAX_FILE_SIZE_BYTES
    except (AttributeError, OSError, ValueError):
        return True


class CorruptedArchiveError(RuntimeError):
    """Raised when an uploaded ZIP archive is corrupted or contains unreadable entries."""

    pass


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
    except Exception as exc:
        logger.error(f"[document_parser] Error reading TXT: {exc}")
    return text.strip()


def extract_text_from_rtf(file: PDFInput) -> str:
    """Extract plain text from an RTF file using striprtf.

    RTF inputs are capped at 10 MB to prevent oversized documents from being
    handed to striprtf and causing avoidable memory spikes.
    """
    if not _rtf_content_within_limit(file):
        logger.warning(
            "[document_parser] Rejected RTF input larger than %d bytes",
            RTF_MAX_FILE_SIZE_BYTES,
        )
        return ""

    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError as exc:
        raise UnsupportedFormatError(
            "striprtf is required to process RTF files. Please install striprtf to parse RTF documents."
        ) from exc

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
    except UnsupportedFormatError:
        raise
    except Exception as exc:
        logger.error(f"[document_parser] Error reading RTF: {exc}")
    return text.strip()


def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL using web scraping."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Web scraping dependencies are missing. Install beautifulsoup4 and "
            "requests using: python -m pip install beautifulsoup4 requests"
        ) from exc

    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]) or parsed.scheme not in (
        "http",
        "https",
    ):
        raise ValueError(f"Invalid URL: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return strip_bibliography(text)

    except requests.RequestException as exc:
        raise Exception(f"Failed to fetch URL: {exc}") from exc
    except Exception as exc:
        raise Exception(f"Failed to parse webpage content: {exc}") from exc


def extract_text_from_epub(file: PDFInput) -> str:
    """Extract plain text from an EPUB file."""
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub

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
    """Extract and aggregate text from all valid documents inside a ZIP archive."""
    from src.core.parsers.dispatch import extract_text

    raw_data = _read_pdf_bytes(file)
    zip_stream = io.BytesIO(raw_data)

    if not zipfile.is_zipfile(zip_stream):
        raise CorruptedArchiveError(
            "Uploaded ZIP file is corrupted or not a valid ZIP archive."
        )

    zip_stream.seek(0)
    extracted_texts = []
    corrupted_files = []

    try:
        with zipfile.ZipFile(zip_stream, "r") as archive:
            for member_name in archive.namelist():
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
                logger.warning(
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
    """Extract plain text from an ODT file."""
    try:
        raw_data = _read_pdf_bytes(file)
        text_parts = []

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

    except Exception as exc:
        logger.error(f"[document_parser] Error reading ODT: {exc}")
        return ""
