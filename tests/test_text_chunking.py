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
Comprehensive Property-Based Tests for Text Chunking Boundaries
Issue: #3496
Tests that the chunking logic preserves text, maintains boundaries, and handles edge cases.
"""

import random
import string

import pytest

# ==============================================================================
# SECTION 1: Defining the Chunking Logic (Under Test)
# ==============================================================================


def chunk_text(text: str, max_chunk_size: int = 50) -> list:
    """
    Splits a string into chunks of a maximum size.
    Ensures no chunk is empty and all text is preserved.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    if not text:
        return []

    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i : i + max_chunk_size])
    return chunks


# ==============================================================================
# SECTION 2: Property-Based Core Tests (The Rules!)
# ==============================================================================


class TestPropertyBasedChunking:
    def test_property_chunks_are_not_empty(self):
        """Property: No chunk in the output should ever be empty."""
        text = "This is a long text that will be split into multiple chunks."
        chunks = chunk_text(text, max_chunk_size=10)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_property_all_text_is_preserved(self):
        """Property: Concatenating chunks should return the exact original text."""
        text = "This is a long text that will be split into multiple chunks."
        chunks = chunk_text(text, max_chunk_size=10)
        assert "".join(chunks) == text

    def test_property_chunks_respect_max_size(self):
        """Property: No chunk should be longer than the max_chunk_size."""
        text = "A" * 1000
        max_size = 50
        chunks = chunk_text(text, max_chunk_size=max_size)
        assert all(len(chunk) <= max_size for chunk in chunks)

    def test_property_small_text_returns_single_chunk(self):
        """Property: Text smaller than the max size should return exactly one chunk."""
        text = "Short text"
        chunks = chunk_text(text, max_chunk_size=100)
        assert len(chunks) == 1

    def test_property_exact_size_text(self):
        """Property: Text exactly equal to max size should return one chunk."""
        text = "A" * 50
        chunks = chunk_text(text, max_chunk_size=50)
        assert len(chunks) == 1

    def test_property_just_over_size_text(self):
        """Property: Text just over max size should return two chunks."""
        text = "A" * 51
        chunks = chunk_text(text, max_chunk_size=50)
        assert len(chunks) == 2


# ==============================================================================
# SECTION 3: Edge Cases
# ==============================================================================


class TestChunkingEdgeCases:
    def test_empty_string_returns_empty_list(self):
        """Edge Case: Empty string should return an empty list."""
        assert chunk_text("") == []

    def test_single_character_text(self):
        """Edge Case: Single character should return one chunk."""
        chunks = chunk_text("A", max_chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == "A"

    def test_special_characters_preserved(self):
        """Edge Case: Special characters like newlines should be preserved."""
        text = "Line1\nLine2\nLine3"
        chunks = chunk_text(text, max_chunk_size=5)
        assert "".join(chunks) == text

    def test_unicode_text_preserved(self):
        """Edge Case: Unicode characters should be preserved."""
        text = "Hello, 世界! How are you?"
        chunks = chunk_text(text, max_chunk_size=10)
        assert "".join(chunks) == text

    def test_invalid_chunk_size_raises_error(self):
        """Edge Case: Negative chunk size should raise an error."""
        with pytest.raises(ValueError):
            chunk_text("Test", max_chunk_size=0)

    def test_spaces_are_not_trimmed(self):
        """Edge Case: Spaces should not be trimmed from chunks."""
        text = "Hello World"
        chunks = chunk_text(text, max_chunk_size=5)
        assert "".join(chunks) == text


# ==============================================================================
# SECTION 4: Random Fuzz Testing (The "Property" Part)
# ==============================================================================


class TestRandomizedFuzz:
    def test_fuzz_random_strings_preserved(self):
        """Fuzz: Random strings should always be preserved when joined."""
        random.seed(42)
        for _ in range(100):
            length = random.randint(1, 500)
            random_string = "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=length)
            )

            chunks = chunk_text(random_string, max_chunk_size=25)
            assert "".join(chunks) == random_string

    def test_fuzz_random_chunk_sizes(self):
        """Fuzz: Random chunk sizes should never produce empty chunks."""
        random.seed(99)
        for _ in range(50):
            text = "A" * random.randint(1, 200)
            size = random.randint(1, 100)
            chunks = chunk_text(text, max_chunk_size=size)
            assert all(len(chunk) > 0 for chunk in chunks)

    def test_fuzz_random_text_with_newlines(self):
        """Fuzz: Random text with newlines should be perfectly preserved."""
        random.seed(10)
        for _ in range(50):
            length = random.randint(10, 200)
            random_string = "".join(
                random.choices(["a", "b", "c", "\n", "\t"], k=length)
            )
            chunks = chunk_text(random_string, max_chunk_size=15)
            assert "".join(chunks) == random_string
