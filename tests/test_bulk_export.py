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
Comprehensive Unit Tests for Bulk Export with Empty Document Dataset
Issue: #3471
Tests edge cases, data integrity, and robustness of the export function.
"""

from typing import Any, Dict, List

import pytest

# ==============================================================================
# SECTION 1: Defining the Local Export Logic (For testing)
# ==============================================================================


def bulk_export(documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Simulated bulk export function.
    Returns a list of valid documents, filtering out invalid entries.
    """
    if not documents:
        return []

    cleaned_documents = []
    for doc in documents:
        if isinstance(doc, dict) and doc.get("content"):
            cleaned_documents.append(doc)
        elif isinstance(doc, str) and doc.strip():
            cleaned_documents.append({"content": doc})
    return cleaned_documents


# ==============================================================================
# SECTION 2: Testing the Basic Empty Dataset Cases
# ==============================================================================


class TestBulkExportBasicEmpty:
    def test_empty_list_returns_empty(self):
        """Should return an empty list for an empty input list."""
        assert bulk_export([]) == []

    def test_empty_tuple_returns_empty(self):
        """Should return an empty list for an empty tuple."""
        assert bulk_export(()) == []

    def test_empty_dict_returns_empty(self):
        """Should return an empty list for an empty dictionary."""
        assert bulk_export({}) == []

    def test_none_input_raises_error(self):
        """Should raise an error for None input."""
        with pytest.raises(TypeError):
            bulk_export(None)

    def test_integer_input_raises_error(self):
        """Should raise an error for non-iterable integer input."""
        with pytest.raises(TypeError):
            bulk_export(12345)


# ==============================================================================
# SECTION 3: Testing Mixed and Filtered Data
# ==============================================================================


class TestBulkExportFilteredData:
    def test_filters_out_none_values(self):
        """Should filter out None values from the list."""
        result = bulk_export([None, "doc1", None, "doc2"])
        assert "doc1" in [doc["content"] for doc in result]
        assert "doc2" in [doc["content"] for doc in result]

    def test_filters_out_empty_strings(self):
        """Should filter out empty strings."""
        result = bulk_export(["", "valid doc", "   "])
        assert len(result) == 1

    def test_filters_out_whitespace_only(self):
        """Should filter out strings with only whitespace."""
        result = bulk_export(["   ", "valid"])
        assert len(result) == 1

    def test_filters_out_plain_integers(self):
        """Should ignore raw integers in the list."""
        result = bulk_export([1, 2, "valid"])
        assert len(result) == 1

    def test_filters_out_invalid_dicts(self):
        """Should ignore dictionaries without content."""
        result = bulk_export([{"title": "No content"}, {"content": "Valid"}])
        assert len(result) == 1
        assert result[0]["content"] == "Valid"


# ==============================================================================
# SECTION 4: Testing Data Integrity and Output
# ==============================================================================


class TestBulkExportDataIntegrity:
    def test_valid_strings_converted_to_dicts(self):
        """Valid strings should be converted to dictionaries."""
        result = bulk_export(["Hello World"])
        assert result == [{"content": "Hello World"}]

    def test_valid_dicts_preserved(self):
        """Valid dictionaries should be preserved as-is."""
        doc = {"id": 1, "content": "Real Doc"}
        result = bulk_export([doc])
        assert result == [doc]

    def test_result_is_always_list(self):
        """The return type should always be a list."""
        assert isinstance(bulk_export([]), list)
        assert isinstance(bulk_export(["text"]), list)

    def test_does_not_mutate_input(self):
        """The function should not alter the original input list."""
        original = ["doc1", "doc2"]
        bulk_export(original)
        assert original == ["doc1", "doc2"]

    def test_duplicate_documents_preserved(self):
        """Should preserve duplicate documents in the output."""
        result = bulk_export(["doc", "doc"])
        assert len(result) == 2

    def test_large_empty_dataset(self):
        """Should handle a large list with mostly None values."""
        large_input = [None] * 1000
        large_input.append("Valid Doc")
        result = bulk_export(large_input)
        assert len(result) == 1


# ==============================================================================
# SECTION 5: Testing Robustness and Error Handling
# ==============================================================================


class TestBulkExportRobustness:
    def test_no_crash_on_clean_input(self):
        """Should handle a normal list without crashing."""
        result = bulk_export(["doc1", "doc2"])
        assert len(result) == 2

    def test_no_crash_on_mixed_input(self):
        """Should handle mixed valid/invalid inputs without crashing."""
        mixed = [None, "", 123, "valid", {"content": "also valid"}]
        result = bulk_export(mixed)
        assert len(result) == 2

    def test_no_crash_on_nested_empty_list(self):
        """Should handle a list containing an empty list."""
        result = bulk_export([[], "valid"])
        # Nested empty list is invalid, so only 'valid' is returned
        assert len(result) == 1

    def test_handles_bytes_input(self):
        """Should handle byte strings gracefully (based on string check)."""
        result = bulk_export([b"binary doc"])
        # Depending on logic, bytes can be ignored or converted. Here we just ensure no crash.
        assert isinstance(result, list)

    def test_single_valid_document(self):
        """Should correctly handle a single valid document."""
        result = bulk_export(["Only Doc"])
        assert len(result) == 1
        assert result[0]["content"] == "Only Doc"

    def test_single_invalid_document(self):
        """Should return an empty list for a single invalid document."""
        result = bulk_export([""])
        assert result == []
