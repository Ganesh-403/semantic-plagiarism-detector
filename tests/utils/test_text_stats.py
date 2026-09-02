from src.utils.text_stats import (
    count_words,
    format_text_stats,
    get_char_count,
    get_reading_time_minutes,
    get_word_count,
    get_sentence_count,
    get_syllable_count,
    get_readability_metrics,
    get_text_stats,
)

"""Unit tests for src/utils/text_stats.py.

This module could not be collected at all until Issue #2556: line 41 asserted
against ``2.0.0``, which is not a valid Python number, so the whole file was a
syntax error. That in turn hid two functions that raised ``NameError`` on every
call -- see ``get_reading_time_minutes`` and ``get_readability_metrics``.
"""

import pytest

from src.utils.text_stats import (
    count_syllables,
    count_syllables_in_word,
    count_words,
    format_text_stats,
    get_char_count,
    get_readability_metrics,
    get_reading_time_minutes,
    get_sentence_count,
    get_syllable_count,
    get_text_stats,
)


def test_count_syllables_in_word():
    assert count_syllables_in_word("apple") == 2
    assert count_syllables_in_word("table") == 2
    assert count_syllables_in_word("blue") == 1
    assert count_syllables_in_word("the") == 1


def test_count_syllables():
    # Regular words
    assert count_syllables("apple") == 2
    assert count_syllables("table") == 2
    assert count_syllables("make") == 1
    assert count_syllables("free") == 1
    assert count_syllables("blue") == 1
    assert count_syllables("syllable") == 3
    assert count_syllables("beauty") == 2
    
    # Edge cases
    assert count_syllables("") == 0
    assert count_syllables("   ") == 0
    assert count_syllables("123!") == 0
    assert count_syllables("a") == 1
    assert count_syllables("me") == 1
    assert count_syllables("the") == 1


def test_count_words():
    assert count_words("This is a test.") == 4
    assert count_words("") == 0
    assert count_words("   Spaces   ") == 1
    assert count_words("hello,world") == 2
    assert count_words("hello-world") == 2
    assert count_words("HELLO hello") == 2
    # CJK word count assertions
    assert count_words("这是一个测试") == 6
    assert count_words("Hello, 这是一个测试!") == 7
    assert count_words("日本語漢字") == 5


def test_get_word_count_matches_count_words():
    """Regression test for #2005: get_word_count() is now an alias for
    count_words() and both entry points must produce identical results,
    including on edge cases (punctuation, contractions, mixed case,
    numbers, hyphenation, empty/whitespace-only input)."""
    cases = [
        "",
        "   ",
        "This is a test.",
        "   Spaces   ",
        "Hello, world! How are you?",
        "don't stop believin'",
        "MiXeD CaSe Words HERE",
        "line1\nline2\ttabbed",
        "numbers 123 and 456 mixed",
        "hyphenated-word another_one",
        "...!!! ???",
    ]
    for text in cases:
        assert get_word_count(text) == count_words(text), (
            f"Mismatch for input {text!r}: "
            f"get_word_count={get_word_count(text)}, "
            f"count_words={count_words(text)}"
        )


def test_get_char_count():
    assert get_char_count("abc") == 3
    assert get_char_count("") == 0


def test_get_reading_time_minutes():
    assert get_reading_time_minutes("word " * 100) == 0.5
    assert get_reading_time_minutes("word " * 400) == 2.0
    assert get_reading_time_minutes("") == 0.1


def test_get_reading_time_minutes_returns_a_float():
    """The declared return type, and what the 0.1 floor implies."""
    assert isinstance(get_reading_time_minutes("word " * 100), float)


def test_get_reading_time_minutes_floors_short_text():
    """Any non-empty text reports a visible duration rather than 0."""
    assert get_reading_time_minutes("one two three") == 0.1


def test_get_sentence_count():
    assert get_sentence_count("Hello world. How are you? Fine!") == 3
    assert get_sentence_count("") == 0
    # count_sentences floors non-empty text at 1: a run of words with no
    # terminator is still one sentence. Only empty/whitespace text scores 0.
    assert get_sentence_count("No punctuation") == 1
    assert get_sentence_count("Dr. Smith arrived. He stayed.") == 2


def test_get_sentence_count_with_dotted_acronyms():
    """Issue #3709: interior periods of dotted acronyms are not sentence breaks."""
    assert get_sentence_count("Visit the U.S.A. today.") == 1
    assert get_sentence_count("The U.S.A. economy grew. Inflation fell.") == 2
    assert get_sentence_count("She works at the C.I.A.S. now. He does not.") == 2


def test_get_sentence_count_with_decimals():
    """Issue #3709: a period between digits is a decimal point, not a break."""
    assert get_sentence_count("Pi is 3.14. It is exact.") == 2
    assert get_sentence_count("The score was 9.75 out of 10.0 overall") == 1


def test_count_sentence_without_ending_punctuation():
    assert get_sentence_count("The cat sat on the mat") == 1


def test_get_syllable_count():
    assert get_syllable_count("the") == 1
    assert get_syllable_count("") == 0


@pytest.mark.xfail(
    reason=(
        "count_syllables_in_word subtracts one for any trailing 'e' that is not "
        "preceded by a consonant + 'l', so 'science' scores 1 instead of 2. The "
        "heuristic feeds the Flesch reading-ease and grade-level scores, so "
        "changing it is a behaviour change with app-wide blast radius and does "
        "not belong in a parse fix. Tracked as a follow-up to #2556."
    ),
    strict=True,
)
def test_get_syllable_count_handles_trailing_ce():
    assert get_syllable_count("science") == 2


def test_get_readability_metrics():
    # Empty / safe handling
    ease, grade = get_readability_metrics("")
    assert ease == 0.0
    assert grade == 0.0

    text = "This is a simple sentence."
    ease, grade = get_readability_metrics(text)
    assert isinstance(ease, float)
    assert isinstance(grade, float)


def test_format_text_stats():
    text = "This is a test sentence."
    stats = format_text_stats(text)
    assert "**Words:** 5" in stats
    assert "**Characters:** 24" in stats
    # 5 words at 200 wpm rounds to 0.0, so the 0.1 floor applies.
    assert "**Est. Reading Time:** 0.1 min" in stats
    assert "**Flesch Reading Ease:**" in stats
    assert "**Flesch-Kincaid Grade:**" in stats


def test_get_text_stats():
    # Empty text
    stats = get_text_stats("")
    assert stats == {
        "words": 0,
        "characters": 0,
        "sentences": 0,
        "syllables": 0,
        "reading_ease": 0.0,
        "grade_level": 0.0,
        "reading_time": 0,
    }

    # Whitespace-only. get_text_stats documents that whitespace-only input
    # returns the all-zero default, so 'characters' is 0 and not len(text).
    stats = get_text_stats("   \n   ")
    assert stats == {
        "words": 0,
        "characters": 0,
        "sentences": 0,
        "syllables": 0,
        "reading_ease": 0.0,
        "grade_level": 0.0,
        "reading_time": 0,
    }

    # Punctuation-only
    stats = get_text_stats("... !!! ???")
    assert stats["words"] == 0
    assert stats["reading_ease"] == 0.0
    assert stats["grade_level"] == 0.0
