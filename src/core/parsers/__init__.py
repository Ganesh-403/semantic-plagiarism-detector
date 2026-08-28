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

"""src/core/parsers - Document parsing strategy package."""

from src.core.parsers.cleaners import (
    ENGLISH_STOPWORDS,
    ZERO_WIDTH_CHARS_PATTERN,
    clean_text,
    detect_text_language,
    get_stopwords,
    load_custom_stopwords,
    mask_named_entities_in_text,
    normalize_extended_punctuation,
    normalize_unicode_nfc,
    normalize_unicode_spaces,
    prepare_text_for_embedding,
    remove_ignore_phrases,
    sanitize_unicode_spaces,
    sanitize_zero_width_characters,
    strip_bibliography,
    strip_markdown_syntax,
)
from src.core.parsers.common import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    PDFInput,
    check_batch_rate_limit,
    normalize_ocr_settings,
    validate_ocr_dpi,
    validate_ocr_language,
)
from src.core.parsers.dispatch import (
    ALLOWED_EXTENSIONS,
    extract_text,
    extract_texts,
    get_supported_file_extensions,
    parallel_extract_texts,
)
from src.core.parsers.docx_parser import (
    ParsedDocxText,
    extract_text_from_doc,
    extract_text_from_docx,
)
from src.core.parsers.ocr_parser import (
    OCRDependencyError,
    check_ocr_dependencies,
    extract_text_from_image,
)
from src.core.parsers.pdf_parser import (
    count_pdf_images,
    extract_pdf_metadata,
    extract_text_from_pdf,
    extract_texts_from_pdfs,
    extract_texts_parallel,
)
from src.core.parsers.text_parser import (
    CorruptedArchiveError,
    extract_text_from_epub,
    extract_text_from_md,
    extract_text_from_odt,
    extract_text_from_rtf,
    extract_text_from_txt,
    extract_text_from_url,
    extract_text_from_zip,
)

__all__ = [
    "PDFInput",
    "DEFAULT_OCR_DPI",
    "DEFAULT_OCR_LANGUAGE",
    "SUPPORTED_OCR_LANGUAGES",
    "ALLOWED_EXTENSIONS",
    "ENGLISH_STOPWORDS",
    "ZERO_WIDTH_CHARS_PATTERN",
    "ParsedDocxText",
    "CorruptedArchiveError",
    "OCRDependencyError",
    "check_ocr_dependencies",
    "check_batch_rate_limit",
    "validate_ocr_dpi",
    "validate_ocr_language",
    "normalize_ocr_settings",
    "load_custom_stopwords",
    "get_stopwords",
    "sanitize_zero_width_characters",
    "normalize_unicode_spaces",
    "sanitize_unicode_spaces",
    "normalize_extended_punctuation",
    "normalize_unicode_nfc",
    "detect_text_language",
    "strip_bibliography",
    "clean_text",
    "remove_ignore_phrases",
    "prepare_text_for_embedding",
    "mask_named_entities_in_text",
    "strip_markdown_syntax",
    "count_pdf_images",
    "extract_pdf_metadata",
    "extract_text_from_pdf",
    "extract_texts_from_pdfs",
    "extract_text_from_docx",
    "extract_text_from_doc",
    "extract_text_from_txt",
    "extract_text_from_rtf",
    "extract_text_from_url",
    "extract_text_from_epub",
    "extract_text_from_md",
    "extract_text_from_zip",
    "extract_text_from_odt",
    "extract_text_from_image",
    "extract_text",
    "extract_texts",
    "extract_texts_parallel",
    "parallel_extract_texts",
    "get_supported_file_extensions",
]
