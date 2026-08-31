"""
test_chunk_text_truncation_issue_2908.py
-----------------------------------------
Unit test suite for Issue #2908:
Validates that chunk_text truncates oversized input text to max_chunk_capacity (max_chunks * chunk_size)
immediately after logging a warning message, preventing memory spikes or infinite loops.
"""

import logging
from src.core.text_chunking import chunk_text


def test_chunk_text_truncates_oversized_text(caplog):
    """Verify chunk_text truncates input text exceeding max_chunk_capacity."""
    huge_text = "word " * 5000  # 25000 characters
    chunk_size = 50
    max_chunks = 10
    max_capacity = max_chunks * chunk_size  # 500 characters

    with caplog.at_level(logging.WARNING):
        chunks = chunk_text(
            huge_text,
            chunk_size=chunk_size,
            chunk_overlap=0,
            min_words=1,
            max_chunks=max_chunks,
        )

    # Verify warning log was captured
    assert "exceeded chunk capacity limit; text was truncated" in caplog.text

    # Verify that total length of generated chunks does not exceed max_capacity
    total_chunk_length = sum(len(c) for c in chunks)
    assert total_chunk_length <= max_capacity
    assert len(chunks) <= max_chunks
