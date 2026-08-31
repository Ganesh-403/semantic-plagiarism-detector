import pytest
from src.utils.diff_highlighter import highlight_overlap, MARK_OPEN_TAG

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
