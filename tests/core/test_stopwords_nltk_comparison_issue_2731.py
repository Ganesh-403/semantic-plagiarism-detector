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
test_stopwords_nltk_comparison_issue_2731.py
---------------------------------------------
Unit test suite for Issue #2731:
Compares the custom ENGLISH_STOPWORDS set against NLTK's English stopwords corpus,
computes set differences (intentionally added vs. omitted stopwords), and prints
them for documentation.
"""

from __future__ import annotations

import pytest

from src.core.stopwords import ENGLISH_STOPWORDS
from src.stopwords import ENGLISH_STOPWORDS as ROOT_ENGLISH_STOPWORDS


def test_compare_english_stopwords_against_nltk(capsys):
    """Compute and document set difference between custom ENGLISH_STOPWORDS and NLTK english corpus."""
    try:
        import nltk

        try:
            from nltk.corpus import stopwords

            nltk_english = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords

            nltk_english = set(stopwords.words("english"))
    except Exception as e:
        pytest.skip(f"NLTK or stopwords corpus unavailable: {e}")

    custom_set = set(ENGLISH_STOPWORDS)

    # Words present in custom ENGLISH_STOPWORDS but omitted/absent in NLTK
    added_in_custom = custom_set - nltk_english

    # Words present in NLTK english corpus but omitted in custom ENGLISH_STOPWORDS
    omitted_from_custom = nltk_english - custom_set

    common_words = custom_set & nltk_english

    print("\n" + "=" * 60)
    print("STOPWORDS COMPARISON: Custom ENGLISH_STOPWORDS vs. NLTK Corpus")
    print("=" * 60)
    print(f"Total Custom ENGLISH_STOPWORDS count: {len(custom_set)}")
    print(f"Total NLTK English stopwords count:   {len(nltk_english)}")
    print(f"Common stopwords count:              {len(common_words)}")
    print("-" * 60)
    print(f"Custom Stopwords Intentionally Added ({len(added_in_custom)} words):")
    print(sorted(added_in_custom))
    print("-" * 60)
    print(f"NLTK Stopwords Omitted from Custom Set ({len(omitted_from_custom)} words):")
    print(sorted(omitted_from_custom))
    print("=" * 60)

    # Assertions for verification
    assert isinstance(custom_set, set)
    assert len(custom_set) > 0
    assert len(nltk_english) > 0
    assert isinstance(added_in_custom, set)
    assert isinstance(omitted_from_custom, set)


def test_root_and_core_stopwords_consistency():
    """Verify consistency between root stopwords and core stopwords modules."""
    assert set(ENGLISH_STOPWORDS) == set(ROOT_ENGLISH_STOPWORDS)
