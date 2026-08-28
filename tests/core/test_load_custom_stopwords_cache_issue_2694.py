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
tests/core/test_load_custom_stopwords_cache_issue_2694.py
---------------------------------------------------------
Unit tests verifying that `load_custom_stopwords` in `src.core.document_parser`
is properly cached using `@functools.lru_cache(maxsize=1)`.

Closes Issue #2694:
Description: In src/core/document_parser.py, get_stopwords() calls
load_custom_stopwords(), which reads from the disk every time it's invoked.
This causes severe disk I/O bottlenecks when parsing thousands of chunks.
Acceptance Criteria: Apply @functools.lru_cache(maxsize=1) to load_custom_stopwords.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.document_parser import get_stopwords, load_custom_stopwords


@pytest.fixture(autouse=True)
def clear_stopwords_lru_cache():
    """Ensure the LRU cache is cleared before and after each test."""
    if hasattr(load_custom_stopwords, "cache_clear"):
        load_custom_stopwords.cache_clear()
    yield
    if hasattr(load_custom_stopwords, "cache_clear"):
        load_custom_stopwords.cache_clear()


def test_load_custom_stopwords_has_lru_cache():
    """Verify that load_custom_stopwords has lru_cache wrapper with maxsize=1."""
    assert hasattr(load_custom_stopwords, "cache_info")
    assert hasattr(load_custom_stopwords, "cache_clear")
    info = load_custom_stopwords.cache_info()
    assert info.maxsize == 1


def test_load_custom_stopwords_caches_disk_reads():
    """Verify that multiple invocations read from disk once and hit cache on subsequent calls."""
    stopwords_content = "custom_stopword_1\ncustom_stopword_2\ncustom_stopword_3\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(stopwords_content)
        tmp_path = tmp.name

    try:
        load_custom_stopwords.cache_clear()
        initial_info = load_custom_stopwords.cache_info()
        assert initial_info.hits == 0

        # First call: cache miss, reads from disk
        res1 = load_custom_stopwords(file_path=tmp_path)
        assert res1 == frozenset(
            {"custom_stopword_1", "custom_stopword_2", "custom_stopword_3"}
        )

        # Second and third calls: cache hits
        res2 = load_custom_stopwords(file_path=tmp_path)
        res3 = load_custom_stopwords(file_path=tmp_path)

        assert res2 == res1
        assert res3 == res1

        current_info = load_custom_stopwords.cache_info()
        assert current_info.hits == 2
        assert current_info.misses >= 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_get_stopwords_uses_cached_custom_stopwords(monkeypatch):
    """Verify that get_stopwords() leverages cached load_custom_stopwords."""
    stopwords_content = "custom_domain_word\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(stopwords_content)
        tmp_path = tmp.name

    monkeypatch.setenv("STOPWORDS_FILE", tmp_path)
    try:
        load_custom_stopwords.cache_clear()

        # Call get_stopwords multiple times
        combined_1 = get_stopwords()
        combined_2 = get_stopwords()

        assert "custom_domain_word" in combined_1
        assert combined_1 == combined_2

        info = load_custom_stopwords.cache_info()
        assert info.hits >= 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_custom_stopwords_cache_clear():
    """Verify cache_clear resets cache and forces a re-read."""
    stopwords_content = "initial_stopword\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(stopwords_content)
        tmp_path = tmp.name

    try:
        load_custom_stopwords.cache_clear()
        res1 = load_custom_stopwords(file_path=tmp_path)
        assert "initial_stopword" in res1

        # Clear cache and verify cache_info is reset
        load_custom_stopwords.cache_clear()
        info = load_custom_stopwords.cache_info()
        assert info.hits == 0
        assert info.misses == 0

        res2 = load_custom_stopwords(file_path=tmp_path)
        assert res2 == res1
        info_after = load_custom_stopwords.cache_info()
        assert info_after.misses == 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)
