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

from pathlib import Path

SOURCE = Path("src/core/cross_lingual.py")
TESTS = Path("tests/core/test_cross_lingual.py")


def test_translation_memory_uses_sha256_sentence_hashing():
    source = SOURCE.read_text(encoding="utf-8")

    assert "class TranslationMemoryCache:" in source
    assert "hashlib.sha256(" in source
    assert "sentence: str" in source


def test_cache_lookup_occurs_before_translator_call():
    source = SOURCE.read_text(encoding="utf-8")

    cache_lookup = source.index("cached_translation = cache.get(")
    translator_call = source.index(
        "translated_text = translator_fn(",
        cache_lookup,
    )
    assert cache_lookup < translator_call


def test_identical_sentence_cache_hit_test_exists():
    source = TESTS.read_text(encoding="utf-8")

    assert "test_translation_memory_cache_hits_for_identical_sentence" in source
    assert "assert len(calls) == 1" in source
