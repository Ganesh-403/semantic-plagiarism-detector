"""
text_stats.py
-------------
Text statistics and analysis utilities for document comparison.

Provides functions to compute various text metrics such as word count,
sentence count, and unique word ratio. These metrics are used in
plagiarism reports to provide additional context about compared documents.
"""

import logging
import re
from typing import Dict, List


def count_words(text: str) -> int:
    """
    Count the number of words in the given text.

    Words are defined as sequences of characters separated by whitespace.
    Punctuation is stripped before counting.

    Args:
        text: The text to analyze

    Returns:
        Number of words in the text
    """
    if not text:
        return 0

    # Remove punctuation and split on whitespace
    words = re.findall(r"\b\w+\b", text.lower())
    return len(words)


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in the given text.

    Sentences are identified by periods, exclamation marks, and question marks.
    Also handles common abbreviations to avoid over-counting.

    Args:
        text: The text to analyze

    Returns:
        Number of sentences in the text
    """
    if not text:
        return 0

    # Common abbreviations that end with period but aren't sentence ends
    abbreviations = [
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "inc.",
        "ltd.",
        "corp.",
        "co.",
        "fig.",
        "tbl.",
        "art.",
        "no.",
        "pp.",
        "vol.",
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "oct.",
        "nov.",
        "dec.",
        "u.s.",
        "u.k.",
        "e.u.",
        "n.a.v.o.",
    ]

    # Replace abbreviations temporarily
    text_lower = text.lower()
    for abbr in abbreviations:
        text_lower = text_lower.replace(abbr, abbr.replace(".", ""))

    # Count sentence-ending punctuation
    sentence_endings = re.findall(r"[.!?]+", text_lower)
    return len(sentence_endings)


def count_unique_words(text: str) -> int:
    """
    Count the number of unique words in the given text.

    Word comparison is case-insensitive.

    Args:
        text: The text to analyze

    Returns:
        Number of unique words in the text
    """
    if not text:
        return 0

    # Get all words (lowercase)
    words = re.findall(r"\b\w+\b", text.lower())

    # Count unique words
    unique_words = set(words)
    return len(unique_words)


def get_unique_word_ratio(text: str) -> float:
    """
    Calculate the unique word ratio (types/tokens ratio).

    This measures lexical diversity - the ratio of unique words to total words.
    Higher values indicate more diverse vocabulary.

    Args:
        text: The text to analyze

    Returns:
        Unique word ratio (0.0 to 1.0)
    """
    if not text:
        return 0.0

    total_words = count_words(text)
    if total_words == 0:
        return 0.0

    unique_words = count_unique_words(text)
    return unique_words / total_words


def compute_text_stats(text: str) -> Dict[str, float]:
    """
    Compute all text statistics for the given text.

    Args:
        text: The text to analyze

    Returns:
        Dictionary containing:
            - word_count: Total number of words
            - sentence_count: Total number of sentences
            - unique_word_count: Number of unique words
            - unique_word_ratio: Ratio of unique words to total words (0-1)
    """
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    unique_word_count = count_unique_words(text)
    unique_word_ratio = get_unique_word_ratio(text)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_word_count": unique_word_count,
        "unique_word_ratio": unique_word_ratio,
    }


def format_stats_for_pdf(stats: Dict[str, float]) -> List[List[str]]:
    """
    Format text statistics for display in a PDF table.

    Args:
        stats: Dictionary of text statistics from compute_text_stats()

    Returns:
        List of rows for a ReportLab Table, each row is [Metric, Value]
    """
    return [
        ["Word Count", str(stats["word_count"])],
        ["Sentence Count", str(stats["sentence_count"])],
        ["Unique Words", str(stats["unique_word_count"])],
        ["Unique Word Ratio", f"{stats['unique_word_ratio']:.2%}"],
    ]



logger = logging.getLogger(__name__)


def get_word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def get_char_count(text: str) -> int:
    return len(text)


def get_reading_time_minutes(text: str) -> int:
    # Average reading speed is roughly 200-250 words per minute.
    # We'll use 200 for a conservative estimate.
    word_count = get_word_count(text)
    minutes = max(1, round(word_count / 200))
    return minutes


def format_text_stats(text: str) -> str:
    words = get_word_count(text)
    chars = get_char_count(text)
    time = get_reading_time_minutes(text)
    reading_ease, grade_level = get_readability_metrics(text)
    return f"**Words:** {words} | **Characters:** {chars} | **Est. Reading Time:** {time} min | **Flesch Reading Ease:** {reading_ease} | **Flesch-Kincaid Grade:** {grade_level}"


def count_syllables_in_word(word: str) -> int:
    """Estimate the syllable count of a single word using basic heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    word = "".join([c for c in word if c.isalpha()])
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    is_prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not is_prev_vowel:
            count += 1
        is_prev_vowel = is_vowel

    if word.endswith("e"):
        count -= 1

    if count <= 0:
        count = 1

    return count


def get_syllable_count(text: str) -> int:
    """Return the total syllable count for the text."""
    words = re.findall(r"\w+", text)
    return sum(count_syllables_in_word(w) for w in words)


def get_sentence_count(text: str) -> int:
    """Return the total sentence count for the text."""
    sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]
    return max(1, len(sentences)) if text.strip() else 0


def get_readability_metrics(text: str) -> tuple[float, float]:
    """Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.

    Returns (flesch_reading_ease, flesch_kincaid_grade).
    """
    words = get_word_count(text)
    sentences = get_sentence_count(text)
    syllables = get_syllable_count(text)

    if words == 0 or sentences == 0:
        if words == 0:
            logger.debug("Word count is zero, returning 0.0 for readability metrics.")
        return 0.0, 0.0

    reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    grade_level = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    return round(reading_ease, 2), round(grade_level, 2)


def get_text_stats(text: str) -> dict[str, int | float]:
    """Calculate and return all text statistics in a structured dictionary.

    Returns default values with zeroes when text is empty or only whitespace.
    """
    if not text or not text.strip():
        logger.debug(
            "Empty or whitespace-only text provided. Returning default zeroes."
        )
        return {
            "words": 0,
            "characters": 0,
            "sentences": 0,
            "syllables": 0,
            "reading_ease": 0.0,
            "grade_level": 0.0,
            "reading_time": 0,
        }

    words = get_word_count(text)
    chars = get_char_count(text)
    sentences = get_sentence_count(text)
    syllables = get_syllable_count(text)
    reading_ease, grade_level = get_readability_metrics(text)
    reading_time = get_reading_time_minutes(text)

    return {
        "words": words,
        "characters": chars,
        "sentences": sentences,
        "syllables": syllables,
        "reading_ease": reading_ease,
        "grade_level": grade_level,
        "reading_time": reading_time,
    }
