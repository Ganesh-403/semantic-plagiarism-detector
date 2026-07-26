from src.utils.text_stats import (format_text_stats, get_char_count,
                                  get_reading_time_minutes, get_word_count)


def test_get_word_count():
    assert get_word_count("This is a test.") == 4
    assert get_word_count("") == 0
    assert get_word_count("   Spaces   ") == 1

def test_get_char_count():
    assert get_char_count("abc") == 3
    assert get_char_count("") == 0

def test_get_reading_time_minutes():
    assert get_reading_time_minutes("word " * 100) == 1
    assert get_reading_time_minutes("word " * 400) == 2
    assert get_reading_time_minutes("") == 1

def test_format_text_stats():
    text = "This is a test sentence."
    stats = format_text_stats(text)
    assert "**Words:** 5" in stats
    assert "**Characters:** 24" in stats
    assert "**Est. Reading Time:** 1 min" in stats
