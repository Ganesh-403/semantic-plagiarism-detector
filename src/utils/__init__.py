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
