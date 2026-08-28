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
tests/security/test_watermarking.py
-----------------------------------
Comprehensive unit tests for the document watermarking and leak tracking system.
"""

import hashlib

import pytest

from src.core.watermark_extractor import extract_watermark, strip_watermarks
from src.db.watermark_logs_db import (
    identify_leak_source,
    initialize_watermark_db,
    log_watermark_generation,
)
from src.security.watermark_engine import (
    ZWC_END,
    ZWC_START,
    embed_watermark,
    generate_watermark_id,
)


class TestWatermarkEngine:
    """Test suite for watermark embedding logic."""

    def test_generate_watermark_id_deterministic(self):
        """Verify the same user/doc pair always generates the same ID."""
        id1 = generate_watermark_id("user_1", "hash_abc")
        id2 = generate_watermark_id("user_1", "hash_abc")
        assert id1 == id2
        assert len(id1) == 32

    def test_embed_watermark_append(self):
        """Verify watermark is appended correctly."""
        text = "This is a confidential exam."
        watermarked, wm_id = embed_watermark(text, "user_1", "hash_abc", "append")

        assert watermarked.startswith(text)
        assert watermarked.endswith(ZWC_END)
        assert len(watermarked) > len(text)

    def test_embed_watermark_distribute(self):
        """Verify watermark is inserted in the middle for distribute strategy."""
        text = "A" * 100
        watermarked, wm_id = embed_watermark(text, "user_1", "hash_abc", "distribute")

        # The watermark should be somewhere in the middle, not at the very end
        assert not watermarked.endswith(ZWC_END)
        assert ZWC_END in watermarked

    def test_invisible_to_standard_len(self):
        """Verify that stripping ZWCs returns the original text length."""
        text = "Hello World"
        watermarked, _ = embed_watermark(text, "u1", "h1")
        cleaned = strip_watermarks(watermarked)
        assert cleaned == text


class TestWatermarkExtractor:
    """Test suite for watermark extraction and decoding."""

    def test_extract_valid_watermark(self):
        """Verify a valid embedded watermark can be extracted and decoded."""
        text = "Confidential data here."
        user_id = "student_42"
        doc_hash = "deadbeef"

        watermarked, original_id = embed_watermark(text, user_id, doc_hash)
        extracted_id = extract_watermark(watermarked)

        assert extracted_id == original_id

    def test_extract_from_truncated_text(self):
        """Verify extraction works even if the end marker is stripped."""
        text = "Some text."
        watermarked, original_id = embed_watermark(text, "u1", "h1")

        # Simulate a platform stripping the end marker
        truncated = watermarked.replace(ZWC_END, "")
        extracted_id = extract_watermark(truncated)

        assert extracted_id == original_id

    def test_extract_returns_none_for_clean_text(self):
        """Verify clean text returns None."""
        assert extract_watermark("Just normal text.") is None

    def test_extract_returns_none_for_empty(self):
        """Verify empty text returns None."""
        assert extract_watermark("") is None


class TestWatermarkDB:
    """Test suite for the watermark logging database."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "test_wm.db"
        initialize_watermark_db(db_path)
        return db_path

    def test_log_and_identify_leak(self, temp_db):
        """Verify a logged watermark can be used to identify a leak."""
        wm_id = "a" * 32
        assert log_watermark_generation(wm_id, "user_99", "doc_hash_1", db_path=temp_db)

        leak_info = identify_leak_source(wm_id, db_path=temp_db)
        assert leak_info is not None
        assert leak_info["user_id"] == "user_99"
        assert leak_info["document_hash"] == "doc_hash_1"

    def test_identify_unknown_leak(self, temp_db):
        """Verify identifying an unlogged watermark returns None."""
        assert identify_leak_source("unknown_id", db_path=temp_db) is None
