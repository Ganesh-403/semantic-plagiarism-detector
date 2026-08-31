"""
Utils package for semantic plagiarism detector.
"""

# Issue #2781: OS compatibility patches
from .os_compat import apply_asyncio_patches, get_os_platform
from .pdf_report import generate_plagiarism_report
from .text_stats import (
    compute_text_stats,
    count_sentences,
    count_unique_words,
    count_words,
    format_stats_for_pdf,
    get_unique_word_ratio,
)

__all__ = [
    "generate_plagiarism_report",
    "compute_text_stats",
    "count_sentences",
    "count_unique_words",
    "count_words",
    "format_stats_for_pdf",
    "get_unique_word_ratio",
    "apply_asyncio_patches",
    "get_os_platform",
]


"""
Utilities module for the Semantic Plagiarism Detector
"""

from .file_validators import FileValidator, validate_file, validate_filename, get_safe_filename
from .text_processor import TextProcessor, process_text

__all__ = [
    'FileValidator', 'validate_file', 'validate_filename', 'get_safe_filename',
    'TextProcessor', 'process_text'
]
