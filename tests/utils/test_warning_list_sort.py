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
tests/utils/test_warning_list_sort.py
-------------------------------------
Comprehensive unit tests for the sort_warnings() multi-column sorting logic.

Verifies the two-pass stable sort implementation, fallback behavior for
invalid fields, and correct ordering for various data types and edge cases.
Addresses Issue #2122.
"""

import logging

import pytest

from src.utils.warning_list import VALID_SORT_FIELDS, sort_warnings


class TestSortWarningsPrimary:
    """Test suite for primary field sorting behavior."""

    def test_sort_by_similarity_desc_default(self):
        """Verify default sort is by similarity descending (highest first)."""
        warnings = [
            {"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.50},
            {"doc_a": "c.pdf", "doc_b": "d.pdf", "similarity": 0.95},
            {"doc_a": "e.pdf", "doc_b": "f.pdf", "similarity": 0.75},
        ]

        result = sort_warnings(warnings)

        assert result[0]["similarity"] == 0.95
        assert result[1]["similarity"] == 0.75
        assert result[2]["similarity"] == 0.50

    def test_sort_by_similarity_asc(self):
        """Verify primary_desc=False sorts similarity ascending."""
        warnings = [
            {"doc_a": "a.pdf", "similarity": 0.90},
            {"doc_a": "b.pdf", "similarity": 0.40},
            {"doc_a": "c.pdf", "similarity": 0.60},
        ]

        result = sort_warnings(warnings, primary_desc=False)

        assert result[0]["similarity"] == 0.40
        assert result[1]["similarity"] == 0.60
        assert result[2]["similarity"] == 0.90

    def test_sort_by_doc_a_asc(self):
        """Verify alphabetical sorting by doc_a (Issue #2122 requirement)."""
        warnings = [
            {"doc_a": "charlie.pdf", "similarity": 0.5},
            {"doc_a": "alice.pdf", "similarity": 0.5},
            {"doc_a": "bob.pdf", "similarity": 0.5},
        ]

        result = sort_warnings(warnings, primary_field="doc_a", primary_desc=False)

        assert result[0]["doc_a"] == "alice.pdf"
        assert result[1]["doc_a"] == "bob.pdf"
        assert result[2]["doc_a"] == "charlie.pdf"

    def test_sort_by_doc_a_desc(self):
        """Verify reverse alphabetical sorting by doc_a."""
        warnings = [
            {"doc_a": "alice.pdf", "similarity": 0.5},
            {"doc_a": "charlie.pdf", "similarity": 0.5},
            {"doc_a": "bob.pdf", "similarity": 0.5},
        ]

        result = sort_warnings(warnings, primary_field="doc_a", primary_desc=True)

        assert result[0]["doc_a"] == "charlie.pdf"
        assert result[1]["doc_a"] == "bob.pdf"
        assert result[2]["doc_a"] == "alice.pdf"


class TestSortWarningsMultiColumn:
    """Test suite for two-pass stable multi-column sorting (Issue #2122)."""

    def test_sort_primary_and_secondary(self):
        """Verify equal primary values are sorted by secondary field."""
        warnings = [
            {"doc_a": "c.pdf", "similarity": 0.90},
            {"doc_a": "a.pdf", "similarity": 0.90},
            {"doc_a": "b.pdf", "similarity": 0.95},
            {"doc_a": "d.pdf", "similarity": 0.90},
        ]

        # Primary: similarity desc, Secondary: doc_a asc
        result = sort_warnings(
            warnings,
            primary_field="similarity",
            secondary_field="doc_a",
            primary_desc=True,
            secondary_desc=False,
        )

        # 0.95 should be first
        assert result[0]["doc_a"] == "b.pdf"
        # The three 0.90s should be sorted alphabetically by doc_a
        assert result[1]["doc_a"] == "a.pdf"
        assert result[2]["doc_a"] == "c.pdf"
        assert result[3]["doc_a"] == "d.pdf"

    def test_stable_sort_preserves_original_order_for_ties(self):
        """Verify Python's stable sort preserves original order when both keys are equal."""
        warnings = [
            {"doc_a": "same.pdf", "doc_b": "first.pdf", "similarity": 0.80},
            {"doc_a": "same.pdf", "doc_b": "second.pdf", "similarity": 0.80},
            {"doc_a": "same.pdf", "doc_b": "third.pdf", "similarity": 0.80},
        ]

        result = sort_warnings(
            warnings, primary_field="similarity", secondary_field="doc_a"
        )

        # Since doc_a is identical, and similarity is identical, original order should be preserved
        assert result[0]["doc_b"] == "first.pdf"
        assert result[1]["doc_b"] == "second.pdf"
        assert result[2]["doc_b"] == "third.pdf"

    def test_secondary_sort_descending(self):
        """Verify secondary_desc=True reverses the secondary sort order."""
        warnings = [
            {"doc_a": "a.pdf", "similarity": 0.90},
            {"doc_a": "c.pdf", "similarity": 0.90},
            {"doc_a": "b.pdf", "similarity": 0.90},
        ]

        result = sort_warnings(
            warnings,
            primary_field="similarity",
            secondary_field="doc_a",
            primary_desc=True,
            secondary_desc=True,  # Reverse alphabetical
        )

        assert result[0]["doc_a"] == "c.pdf"
        assert result[1]["doc_a"] == "b.pdf"
        assert result[2]["doc_a"] == "a.pdf"


class TestSortWarningsFallbacks:
    """Test suite for invalid field fallback behavior (Issue #2122)."""

    def test_sort_invalid_primary_field_falls_back(self, caplog):
        """Verify passing invalid primary_field defaults to 'similarity'."""
        warnings = [
            {"doc_a": "a.pdf", "similarity": 0.50},
            {"doc_a": "b.pdf", "similarity": 0.90},
        ]

        with caplog.at_level(logging.WARNING):
            result = sort_warnings(warnings, primary_field="invalid_field_xyz")

        # Should fall back to similarity descending
        assert result[0]["similarity"] == 0.90
        assert result[1]["similarity"] == 0.50

        # Verify warning was logged
        assert any(
            "Invalid primary_field" in record.message for record in caplog.records
        )

    def test_sort_invalid_secondary_field_falls_back(self, caplog):
        """Verify passing invalid secondary_field defaults to 'doc_a'."""
        warnings = [
            {"doc_a": "b.pdf", "similarity": 0.90},
            {"doc_a": "a.pdf", "similarity": 0.90},
        ]

        with caplog.at_level(logging.WARNING):
            result = sort_warnings(
                warnings,
                primary_field="similarity",
                secondary_field="invalid_secondary_xyz",
            )

        # Should fall back to doc_a ascending for the tie
        assert result[0]["doc_a"] == "a.pdf"
        assert result[1]["doc_a"] == "b.pdf"

        assert any(
            "Invalid secondary_field" in record.message for record in caplog.records
        )

    def test_sort_both_invalid_fields_fall_back(self, caplog):
        """Verify both fields fall back to defaults when both are invalid."""
        warnings = [
            {"doc_a": "c.pdf", "similarity": 0.80},
            {"doc_a": "a.pdf", "similarity": 0.90},
        ]

        with caplog.at_level(logging.WARNING):
            result = sort_warnings(
                warnings, primary_field="bad1", secondary_field="bad2"
            )

        # Should sort by similarity desc, then doc_a asc
        assert result[0]["similarity"] == 0.90
        assert result[1]["similarity"] == 0.80


class TestSortWarningsEdgeCases:
    """Test suite for edge cases and data quality issues."""

    def test_sort_empty_list_returns_empty(self):
        """Verify empty input returns empty list."""
        assert sort_warnings([]) == []

    def test_sort_handles_missing_primary_key(self):
        """Verify missing primary key defaults to 0.0 for numeric fields."""
        warnings = [
            {"doc_a": "a.pdf", "similarity": 0.90},
            {"doc_a": "b.pdf"},  # Missing similarity
            {"doc_a": "c.pdf", "similarity": 0.50},
        ]

        result = sort_warnings(warnings, primary_field="similarity", primary_desc=True)

        # Missing similarity (0.0) should be last when descending
        assert result[0]["similarity"] == 0.90
        assert result[1]["similarity"] == 0.50
        assert result[2]["doc_a"] == "b.pdf"

    def test_sort_handles_missing_secondary_key(self):
        """Verify missing secondary key defaults to empty string."""
        warnings = [
            {"doc_a": "b.pdf", "similarity": 0.90},
            {"similarity": 0.90},  # Missing doc_a
            {"doc_a": "a.pdf", "similarity": 0.90},
        ]

        result = sort_warnings(
            warnings,
            primary_field="similarity",
            secondary_field="doc_a",
            secondary_desc=False,
        )

        # Empty string should come before "a.pdf" and "b.pdf" in ascending order
        assert "doc_a" not in result[0] or result[0].get("doc_a") == ""
        assert result[1]["doc_a"] == "a.pdf"
        assert result[2]["doc_a"] == "b.pdf"

    def test_sort_handles_non_numeric_primary_values(self):
        """Verify non-numeric primary values are treated as 0.0."""
        warnings = [
            {"doc_a": "a.pdf", "similarity": "invalid_string"},
            {"doc_a": "b.pdf", "similarity": 0.80},
            {"doc_a": "c.pdf", "similarity": None},
        ]

        result = sort_warnings(warnings, primary_field="similarity", primary_desc=True)

        # 0.80 should be first, the invalid/None ones should be tied at 0.0
        assert result[0]["similarity"] == 0.80

    def test_sort_does_not_mutate_original_list(self):
        """Verify the original list is not modified (returns new list)."""
        warnings = [
            {"doc_a": "b.pdf", "similarity": 0.50},
            {"doc_a": "a.pdf", "similarity": 0.90},
        ]
        original_copy = warnings.copy()

        sort_warnings(warnings)

        assert warnings == original_copy

    @pytest.mark.parametrize("field", list(VALID_SORT_FIELDS))
    def test_all_valid_fields_accepted_without_warning(self, field, caplog):
        """Verify all fields in VALID_SORT_FIELDS do not trigger fallback warnings."""
        warnings = [
            {
                "doc_a": "a.pdf",
                "similarity": 0.5,
                "doc_b": "b.pdf",
                "severity": "High",
                "timestamp": "2024-01-01",
            }
        ]

        with caplog.at_level(logging.WARNING):
            sort_warnings(warnings, primary_field=field, secondary_field=field)

        assert not any("Invalid" in record.message for record in caplog.records)
