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

import pytest

from src.utils.diff_highlighter import MARK_OPEN_TAG, highlight_overlap


def test_highlight_overlap_empty_strings():
    """Empty inputs must return ("", "") without raising an exception."""
    assert highlight_overlap("", "") == ("", "")


def test_highlight_overlap_empty_first():
    assert highlight_overlap("", "some text") == ("", "some text")


def test_highlight_overlap_empty_second():
    assert highlight_overlap("some text", "") == ("some text", "")


def test_highlight_overlap_punctuation_and_spaces():
    """Strings containing only punctuation/spaces should return escaped text with no highlights."""
    assert highlight_overlap("! ? .", "!!!") == ("! ? .", "!!!")
    assert highlight_overlap("   ", " ") == ("   ", " ")


def test_highlight_overlap_single_character():
    """Single-character strings shouldn't be highlighted due to min_match_length default."""
    assert highlight_overlap("a", "b") == ("a", "b")
    assert highlight_overlap("a", "a") == ("a", "a")


def test_highlight_overlap_identical_10_word_sentence():
    """Identical 10-word sentences must receive full wrapping."""
    sentence = "one two three four five six seven eight nine ten"
    expected_output = f"{MARK_OPEN_TAG}{sentence}</mark>"
    assert highlight_overlap(sentence, sentence) == (expected_output, expected_output)
