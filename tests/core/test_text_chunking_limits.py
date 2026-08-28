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
tests/core/test_text_chunking_limits.py
---------------------------------------
Edge case and stress tests for text chunking safety limits.

Ensures the chunking pipeline is robust against memory exhaustion
attacks via extremely large document uploads.
"""

import logging

from src.core.text_chunking import chunk_by_sentences


class TestChunkingMemorySafety:
    """Stress tests for memory safety and limit enforcement."""

    def test_extremely_large_document_respects_limit(self):
        """Verify a 10,000 sentence document is capped at max_chunks."""
        # Simulate a massive document
        text = (
            ". ".join(
                [
                    f"This is sentence number {i} in a massive document"
                    for i in range(10000)
                ]
            )
            + "."
        )

        chunks = chunk_by_sentences(text, max_chunks=50, min_chunk_length=10)
        assert len(chunks) == 50

    def test_max_chunks_one_returns_single_chunk(self):
        """Verify max_chunks=1 returns exactly one chunk."""
        text = "First. Second. Third. Fourth. Fifth."
        chunks = chunk_by_sentences(text, max_chunks=1, min_chunk_length=1)
        assert len(chunks) == 1

    def test_warning_contains_limit_value(self, caplog):
        """Verify the warning message includes the actual limit value."""
        text = ". ".join([f"S{i}" for i in range(20)]) + "."
        limit = 3

        with caplog.at_level(logging.WARNING):
            chunk_by_sentences(text, max_chunks=limit, min_chunk_length=1)

        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(str(limit) in msg for msg in warning_msgs)

    def test_chunking_preserves_sentence_integrity_at_limit(self):
        """Verify sentences are not split mid-word when hitting the limit."""
        sentences = [f"UniqueWord{i} is here." for i in range(20)]
        text = " ".join(sentences)

        chunks = chunk_by_sentences(text, max_chunks=5, min_chunk_length=1)

        # Verify no chunk contains a partial word like "UniqueWor"
        for chunk in chunks:
            assert "UniqueWor " not in chunk
            # Every word should be complete
            words = chunk.split()
            for word in words:
                assert not word.endswith("UniqueWor")

    def test_min_chunk_length_interaction_with_max_chunks(self):
        """Verify min_chunk_length filtering happens before max_chunks counting."""
        # Create text where many chunks would be filtered by min_length
        text = "A. B. C. This is a longer sentence that passes the filter."

        # If min_length filters the short ones, we might not hit max_chunks
        chunks = chunk_by_sentences(text, max_chunks=10, min_chunk_length=20)

        # Should only contain the long sentence
        assert len(chunks) == 1
        assert "longer sentence" in chunks[0]
