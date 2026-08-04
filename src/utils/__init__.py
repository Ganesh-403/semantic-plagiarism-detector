"""
Utils package for semantic plagiarism detector.
"""

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
]
