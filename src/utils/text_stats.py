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
text_stats.py
-------------
Text statistics and analysis utilities for document comparison.

Provides functions to compute various text metrics such as word count,
sentence count, and unique word ratio. These metrics are used in
plagiarism reports to provide additional context about compared documents.
"""

from __future__ import annotations

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


# Common abbreviations whose trailing period does not end a sentence. Matched
# whole-word only — see ``_ABBREVIATION_RE`` for why that matters.
ABBREVIATIONS = (
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "inc",
    "ltd",
    "corp",
    "co",
    "fig",
    "tbl",
    "art",
    "no",
    "pp",
    "vol",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

# An abbreviation only counts when the letters before the period form a whole
# word. Without the leading boundary, "co." matches inside "disco." and "no."
# inside "casino.", which swallows the real sentence break that follows.
_ABBREVIATION_RE = re.compile(
    r"\b(?:" + "|".join(ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)

# Dotted acronyms such as "U.S.", "e.u." or "N.A.V.O." — any run of two or more
# single letters each followed by a period. Written generically so the list
# above does not need an entry per acronym.
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")

# A period between two digits is a decimal point ("3.14", "v2.5"), never a
# sentence break.
_DECIMAL_POINT_RE = re.compile(r"(?<=\d)\.(?=\d)")

# Placeholder substituted for periods that must not be counted. A private-use
# code point is used so it cannot collide with anything in the document, and so
# masking never joins two words together the way deleting the period would.
_PROTECTED_PERIOD = "\ue000"

_SENTENCE_ENDING_RE = re.compile(r"[.!?]+")


def _mask_non_terminal_periods(text: str) -> str:
    """Replace periods that do not end a sentence with a placeholder.

    Covers abbreviations, dotted acronyms and decimal points. The surrounding
    letters are preserved so that word boundaries elsewhere in the text are
    unaffected — only the period itself is swapped out.
    """
    masked = _DECIMAL_POINT_RE.sub(_PROTECTED_PERIOD, text)
    masked = _DOTTED_ACRONYM_RE.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        masked,
    )
    masked = _ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        masked,
    )
    return masked


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in the given text.

    Sentences are identified by periods, exclamation marks, and question marks.
    Periods belonging to common abbreviations ("Dr."), dotted acronyms ("U.S.")
    and decimal numbers ("3.14") are excluded so they do not inflate the count.

    Non-empty text always counts as at least one sentence, so a run of words
    with no terminal punctuation is reported as a single sentence rather than
    none.

    Args:
        text: The text to analyze

    Returns:
        Number of sentences in the text
    """
    if not text or not text.strip():
        return 0

    masked = _mask_non_terminal_periods(text)

    # Count sentence-ending punctuation. Consecutive marks ("?!", "...") are a
    # single break because the pattern matches them as one run.
    sentence_endings = _SENTENCE_ENDING_RE.findall(masked)
    return max(1, len(sentence_endings))


# Alias get_sentence_count to count_sentences for backward compatibility
get_sentence_count = count_sentences


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


def compute_text_stats(text: str) -> dict[str, float]:
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


def format_stats_for_pdf(stats: dict[str, float]) -> list[list[str]]:
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
    """Deprecated alias for :func:`count_words`.

    Kept for backward compatibility with existing callers/tests. New code
    should call :func:`count_words` directly.
    """
    return count_words(text)


def get_char_count(text: str) -> int:
    return len(text)


def get_reading_time_minutes(text: str) -> float:
    """Estimate reading time in minutes.

    Average reading speed is roughly 200-250 words per minute; 200 is used as
    a conservative estimate. The result is rounded to one decimal place and
    floored at 0.1 so that any non-empty text reports a visible duration
    rather than "0 min".
    """
    word_count = count_words(text)
    return max(0.1, round(word_count / 200, 1))


def format_text_stats(text: str) -> str:
    words = count_words(text)
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
        is_consonant_le = (
            len(word) >= 3 and word.endswith("le") and word[-3] not in vowels
        )
        if not is_consonant_le:
            count -= 1

    if count <= 0:
        count = 1

    return count


def get_syllable_count(text: str) -> int:
    """Return the total syllable count for the text."""
    words = re.findall(r"\w+", text)
    return sum(count_syllables_in_word(w) for w in words)


def get_readability_metrics(text: str) -> tuple[float, float]:
    """Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.

    Returns (flesch_reading_ease, flesch_kincaid_grade).
    """
    words = count_words(text)
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

    words = count_words(text)
    chars = get_char_count(text)
    sentences = count_sentences(text)
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
