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
tests/utils/test_warning_list_fuzzy.py
--------------------------------------
Unit tests for fuzzy string matching in filter_warnings() (Issue #2121).

Requires thefuzz library. Tests verify fuzzy matching behavior, threshold
cutoffs, and fallback to exact matching when thefuzz is unavailable.
"""

from unittest.mock import patch

import pytest

from src.utils.warning_list import FUZZY_THRESHOLD, THEFUZZ_AVAILABLE, filter_warnings

# Skip all fuzzy tests if thefuzz is not installed
pytestmark = pytest.mark.skipif(
    not THEFUZZ_AVAILABLE, reason="thefuzz library not installed"
)


class TestFilterWarningsFuzzy:
    """Test suite for fuzzy matching behavior in filter_warnings()."""

    def test_filter_fuzzy_match_typo(self):
        """Verify fuzzy matching tolerates minor typos in the query."""
        warnings = [{"doc_a": "plagiarism_essay_final.pdf", "doc_b": "source.pdf"}]

        # "plagiarism" with a typo
        result = filter_warnings(warnings, "plagirism", use_fuzzy=True)
        assert len(result) == 1

    def test_filter_fuzzy_match_partial(self):
        """Verify fuzzy matching works with partial string matches."""
        warnings = [{"doc_a": "student_john_doe_submission.docx", "doc_b": "wiki.pdf"}]

        # Partial match "john doe"
        result = filter_warnings(warnings, "john doe", use_fuzzy=True)
        assert len(result) == 1

    def test_filter_fuzzy_no_match_too_different(self):
        """Verify completely different strings are filtered out."""
        warnings = [{"doc_a": "alice_essay.pdf", "doc_b": "bob_essay.pdf"}]

        # Completely unrelated query
        result = filter_warnings(warnings, "xyz123quantum", use_fuzzy=True)
        assert len(result) == 0

    def test_filter_fuzzy_respects_threshold(self):
        """Verify matches below FUZZY_THRESHOLD are excluded."""
        warnings = [{"doc_a": "document.pdf", "doc_b": "other.pdf"}]

        # Query that might have a low fuzzy score (e.g., "doc" vs "document")
        # We mock fuzz.partial_ratio to return exactly the threshold - 1
        with patch("src.utils.warning_list.fuzz") as mock_fuzz:
            mock_fuzz.partial_ratio.return_value = FUZZY_THRESHOLD - 1

            result = filter_warnings(warnings, "doc", use_fuzzy=True)
            assert len(result) == 0

    def test_filter_fuzzy_includes_at_threshold(self):
        """Verify matches exactly at FUZZY_THRESHOLD are included."""
        warnings = [{"doc_a": "document.pdf", "doc_b": "other.pdf"}]

        with patch("src.utils.warning_list.fuzz") as mock_fuzz:
            mock_fuzz.partial_ratio.return_value = FUZZY_THRESHOLD

            result = filter_warnings(warnings, "doc", use_fuzzy=True)
            assert len(result) == 1

    def test_filter_fuzzy_disabled_falls_back_to_exact(self):
        """Verify use_fuzzy=False disables fuzzy matching even if available."""
        warnings = [{"doc_a": "plagiarism_essay.pdf", "doc_b": "source.pdf"}]

        # Typo should NOT match when fuzzy is disabled
        result = filter_warnings(warnings, "plagirism", use_fuzzy=False)
        assert len(result) == 0

    def test_filter_fuzzy_checks_both_documents(self):
        """Verify fuzzy matching checks both doc_a and doc_b."""
        warnings = [{"doc_a": "random.pdf", "doc_b": "plagiarism_source.pdf"}]

        # Typo in doc_b should still match
        result = filter_warnings(warnings, "plagirism", use_fuzzy=True)
        assert len(result) == 1

    def test_filter_fuzzy_handles_empty_documents(self):
        """Verify fuzzy matching handles empty document names gracefully."""
        warnings = [{"doc_a": "", "doc_b": ""}]

        result = filter_warnings(warnings, "test", use_fuzzy=True)
        assert len(result) == 0


class TestFilterWarningsFallback:
    """Test suite for fallback behavior when thefuzz is unavailable."""

    def test_fallback_when_thefuzz_missing(self):
        """Verify exact matching is used when thefuzz is not installed."""
        warnings = [{"doc_a": "plagiarism_essay.pdf", "doc_b": "source.pdf"}]

        # Mock THEFUZZ_AVAILABLE to False
        with patch("src.utils.warning_list.THEFUZZ_AVAILABLE", False):
            # Typo should NOT match because fuzzy is unavailable
            result = filter_warnings(warnings, "plagirism", use_fuzzy=True)
            assert len(result) == 0

            # Exact match should still work
            result_exact = filter_warnings(warnings, "plagiarism", use_fuzzy=True)
            assert len(result_exact) == 1
