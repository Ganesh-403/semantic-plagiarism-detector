"""Lexical diversity regressions against the real text statistics utility."""

import pytest
from src.utils.text_stats import get_unique_word_ratio


@pytest.mark.parametrize(("text", "expected"), [
    ("", 0.0), (None, 0.0), (" \t\n", 0.0),
    ("the quick brown fox jumps over the lazy dog", 8 / 9),
    ("version 1 version 2 version 3", 4 / 6),
    ("hello hello hello hello", 0.25), ("cat dog cat bird", 0.75),
    ("yes no yes no", 0.5), ("word " * 100, 0.01),
    ("Hello hello HELLO", 1 / 3), ("test! Test, test.", 1 / 3),
    ("hello@world hello@world", 0.5), ("猫猫狗", 2 / 3),
    ("café CAFÉ", 0.5), ("unique", 1.0),
])
def test_unique_word_ratio(text, expected):
    assert get_unique_word_ratio(text) == pytest.approx(expected)
