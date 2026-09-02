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


# Token patterns shared by :func:`count_words`.
#
# ``_WORD_RE`` is the historical rule: a maximal run of word characters
# (letters, digits, underscore). A hyphen is not a word character, so it ends
# the token and "state-of-the-art" is counted as four separate words.
#
# ``_HYPHENATED_WORD_RE`` also keeps single internal hyphens inside a token, so
# "state-of-the-art" counts as one lexical unit. ``[\w]`` is just an explicit
# spelling of ``\w``; the ``(?:-[\w]+)*`` tail matches zero or more "-word"
# groups that follow the first run.
_WORD_RE = re.compile(r"\b\w+\b")
_HYPHENATED_WORD_RE = re.compile(r"\b[\w]+(?:-[\w]+)*\b")


def count_words(text: str, count_hyphenated_as_single: bool = False) -> int:
    """
    Count the number of words in the given text.

    Words are defined as sequences of characters separated by whitespace.
    Punctuation is stripped before counting. For CJK scripts, individual
    characters are counted as individual word units.

    Args:
        text: The text to analyze
        count_hyphenated_as_single: When ``True``, a hyphenated compound such
            as ``"state-of-the-art"`` is counted as a single word rather than
            one word per hyphen-separated part. Defaults to ``False`` so the
            original behaviour is unchanged for every existing caller.

    Returns:
        Number of words in the text. With ``count_hyphenated_as_single=True``,
        ``"state-of-the-art"`` contributes 1 to the count instead of 4.
    """
    if not text:
        return 0

    pattern = _HYPHENATED_WORD_RE if count_hyphenated_as_single else _WORD_RE

    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if cjk_chars:
        # Replace CJK characters with space to avoid merging adjacent English words
        non_cjk_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = pattern.findall(non_cjk_text.lower())
        return len(cjk_chars) + len(words)

    # Remove punctuation and split on whitespace
    words = pattern.findall(text.lower())
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

# Spaced ellipsis such as ". . ." or ". . . ." — periods separated by whitespace
# that represent a pause or omitted text rather than multiple sentence breaks.
_SPACED_ELLIPSIS_RE = re.compile(r"(?:\.\s+)+\.")

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

    Covers abbreviations, dotted acronyms, decimal points, and spaced ellipses.
    The surrounding letters are preserved so that word boundaries elsewhere in
    the text are unaffected — only the period itself is swapped out.
    """
    masked = _SPACED_ELLIPSIS_RE.sub(_PROTECTED_PERIOD, text)
    masked = _DECIMAL_POINT_RE.sub(_PROTECTED_PERIOD, masked)
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
    Periods belonging to common abbreviations ("Dr."), dotted acronyms ("U.S."),
    decimal numbers ("3.14"), and spaced ellipses (". . .") are excluded so they
    do not inflate the count.

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

    Word comparison is case-insensitive. Populates the set directly using
    re.finditer to avoid intermediate list allocation for large texts (Issue #3706).

    Args:
        text: The text to analyze

    Returns:
        Number of unique words in the text
    """
    if not text:
        return 0

    cjk_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars:
        non_cjk_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = set(m.group(0).lower() for m in re.finditer(r"\b\w+\b", non_cjk_text))
        return len(cjk_chars | words)

    unique_words = set(m.group(0).lower() for m in re.finditer(r"\b\w+\b", text))
    return len(unique_words)


def get_unique_words_set(text: str) -> set[str]:
    """Return the set of unique lowercased words extracted from text via generator iteration.

    Args:
        text: The text to analyze.

    Returns:
        set[str]: Set of distinct word tokens.
    """
    if not text:
        return set()

    cjk_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars:
        non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = set(m.group(0).lower() for m in re.finditer(r"\b\w+\b", non_cjk))
        return cjk_chars | words

    return set(m.group(0).lower() for m in re.finditer(r"\b\w+\b", text))


def iter_word_tokens(text: str) -> Iterable[str]:
    """Lazily yield lowercased word tokens without allocating a full list in memory.

    Args:
        text: The text to analyze.

    Yields:
        str: Individual lowercased word tokens.
    """
    if not text:
        return
    for match in re.finditer(r"\b\w+\b", text):
        yield match.group(0).lower()


def compute_vocabulary_richness(text: str) -> dict[str, float]:
    """Compute advanced lexical diversity indices based on unique and total word counts.

    Calculates:
        - types: Unique word count (V)
        - tokens: Total word count (N)
        - ttr: Type-Token Ratio (V / N)
        - guiraud_r: Guiraud's R index (V / sqrt(N))
        - herdan_c: Herdan's C index (log(V) / log(N))

    Args:
        text: The text to analyze.

    Returns:
        dict[str, float]: Dictionary of computed lexical richness metrics.
    """
    import math

    tokens = count_words(text)
    if tokens == 0:
        return {
            "tokens": 0,
            "types": 0,
            "ttr": 0.0,
            "guiraud_r": 0.0,
            "herdan_c": 0.0,
        }

    types = count_unique_words(text)
    ttr = types / tokens
    guiraud_r = types / math.sqrt(tokens) if tokens > 0 else 0.0
    herdan_c = math.log(types) / math.log(tokens) if tokens > 1 and types > 0 else 1.0

    return {
        "tokens": tokens,
        "types": types,
        "ttr": round(ttr, 4),
        "guiraud_r": round(guiraud_r, 4),
        "herdan_c": round(herdan_c, 4),
    }


def batch_count_unique_words(texts: list[str]) -> list[int]:
    """Compute unique word counts across a batch list of documents with generator efficiency.

    Args:
        texts: List of document strings.

    Returns:
        list[int]: List of unique word counts corresponding to each document.
    """
    return [count_unique_words(t) for t in texts]


def stream_word_frequencies(text: str) -> dict[str, int]:
    """Compute word occurrence frequencies using iterative token stream.

    Args:
        text: Input text string.

    Returns:
        dict[str, int]: Frequency map of lowercased word tokens.
    """
    from collections import Counter
    if not text:
        return {}
    return dict(Counter(iter_word_tokens(text)))


def compute_hapax_legomena(text: str) -> list[str]:
    """Identify words that appear exactly once in the entire text (Hapax Legomena).

    A high proportion of hapax legomena often correlates with sophisticated vocabulary
    or unique stylistic fingerprinting useful for authorship attribution.

    Args:
        text: Input text to analyze.

    Returns:
        list[str]: Sorted list of single-occurrence words.
    """
    freqs = stream_word_frequencies(text)
    return sorted([word for word, count in freqs.items() if count == 1])


def compute_hapax_dislegomena(text: str) -> list[str]:
    """Identify words that appear exactly twice in the text (Hapax Dislegomena).

    Args:
        text: Input text to analyze.

    Returns:
        list[str]: Sorted list of double-occurrence words.
    """
    freqs = stream_word_frequencies(text)
    return sorted([word for word, count in freqs.items() if count == 2])


def compute_yule_k_characteristic(text: str) -> float:
    """Calculate Yule's Characteristic K metric for vocabulary richness.

    Yule's K is a mathematically invariant measure of vocabulary diversity
    that remains relatively independent of total text length.

    Formula:
        K = 10^4 * (sum(i^2 * V_i) - N) / N^2
        where V_i is the number of words occurring i times, and N is total tokens.

    Args:
        text: Input text to analyze.

    Returns:
        float: Yule's K characteristic value (higher indicates lower lexical diversity).
    """
    from collections import Counter

    tokens = count_words(text)
    if tokens <= 1:
        return 0.0

    freqs = stream_word_frequencies(text)
    spectrum = Counter(freqs.values())

    s1 = sum(count * freq for freq, count in spectrum.items())
    s2 = sum(count * (freq ** 2) for freq, count in spectrum.items())

    if s1 <= 1:
        return 0.0

    k = 10000.0 * (s2 - s1) / (s1 ** 2)
    return round(k, 2)


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


def count_letters(text: str) -> int:
    """Count the total number of letters (alphanumeric characters) in the text.

    Used by readability formulas such as the Coleman-Liau index.

    Args:
        text: The text to analyze.

    Returns:
        int: Total letter count.
    """
    if not text:
        return 0
    # Standard Coleman-Liau counts letters/alphanumerics as characters
    letters = re.findall(r"[a-zA-Z0-9]", text)
    return len(letters)


def compute_coleman_liau_index(text: str) -> float:
    """Calculate the Coleman-Liau readability index for the given text.

    The Coleman-Liau index is formulaic and calculates the US school grade level
    required to comprehend the text based on character and sentence counts per 100 words,
    without requiring syllable dictionaries or tokenizers.

    Formula:
        L = (letters / words) * 100
        S = (sentences / words) * 100
        CLI = 0.0588 * L - 0.296 * S - 15.8

    Args:
        text: The text to analyze.

    Returns:
        float: Estimated grade level, rounded to 2 decimal places (or 0.0 if text is empty).
    """
    if not text or not text.strip():
        return 0.0

    words = count_words(text)
    if words == 0:
        return 0.0

    sentences = count_sentences(text)
    letters = count_letters(text)

    # Average number of letters per 100 words
    l_val = (letters / words) * 100.0
    # Average number of sentences per 100 words
    s_val = (sentences / words) * 100.0

    cli = 0.0588 * l_val - 0.296 * s_val - 15.8
    return round(cli, 2)


def get_coleman_liau_grade_description(score: float) -> str:
    """Return a human-readable grade level description for a Coleman-Liau index score.

    Args:
        score: Coleman-Liau index numerical score.

    Returns:
        str: Descriptive education level / grade bracket.
    """
    if score <= 0.0:
        return "Before Grade 1 (Kindergarten / Early)"
    if score < 6.0:
        return f"Elementary School (Grade {round(score)})"
    if score < 9.0:
        return f"Middle School (Grade {round(score)})"
    if score < 13.0:
        return f"High School (Grade {round(score)})"
    if score < 17.0:
        return "College / Undergraduate Level"
    return "Graduate / Professional Level"


def compare_readability_scores(
    text_a: str, text_b: str
) -> dict[str, Any]:
    """Compare readability metrics between two texts to assist in plagiarism evaluation.

    Args:
        text_a: First document text.
        text_b: Second document text.

    Returns:
        Dictionary comparing Coleman-Liau scores, grade level differences, and similarity.
    """
    cli_a = compute_coleman_liau_index(text_a)
    cli_b = compute_coleman_liau_index(text_b)
    diff = abs(cli_a - cli_b)

    return {
        "doc_a_readability": cli_a,
        "doc_b_readability": cli_b,
        "absolute_difference": round(diff, 2),
        "shares_grade_level": round(cli_a) == round(cli_b),
        "doc_a_level": get_coleman_liau_grade_description(cli_a),
        "doc_b_level": get_coleman_liau_grade_description(cli_b),
    }


def explain_readability_grade_breakdown(text: str) -> dict[str, Any]:
    """Provide a granular breakdown of letters, words, sentences, and CLI terms.

    Args:
        text: The text to analyze.

    Returns:
        dict: Detailed breakdown containing L, S, CLI, and intermediate counts.
    """
    words = count_words(text)
    sentences = count_sentences(text)
    letters = count_letters(text)

    if words == 0:
        return {
            "words": 0,
            "sentences": 0,
            "letters": 0,
            "letters_per_100_words": 0.0,
            "sentences_per_100_words": 0.0,
            "coleman_liau_index": 0.0,
            "grade_bracket": get_coleman_liau_grade_description(0.0),
        }

    l_val = (letters / words) * 100.0
    s_val = (sentences / words) * 100.0
    cli = round(0.0588 * l_val - 0.296 * s_val - 15.8, 2)

    return {
        "words": words,
        "sentences": sentences,
        "letters": letters,
        "letters_per_100_words": round(l_val, 2),
        "sentences_per_100_words": round(s_val, 2),
        "coleman_liau_index": cli,
        "grade_bracket": get_coleman_liau_grade_description(cli),
    }


def batch_compute_readability(texts: list[str]) -> list[dict[str, Any]]:
    """Compute Coleman-Liau readability metrics for a batch list of texts.

    Args:
        texts: List of text strings to analyze.

    Returns:
        list[dict]: List of results with index, word_count, CLI score, and description.
    """
    results = []
    for idx, text in enumerate(texts):
        cli = compute_coleman_liau_index(text)
        results.append({
            "index": idx,
            "word_count": count_words(text),
            "readability_score": cli,
            "grade_level": get_coleman_liau_grade_description(cli),
        })
    return results


def assess_readability_homogeneity(documents: dict[str, str]) -> dict[str, Any]:
    """Assess whether a set of suspected documents share similar readability grade profiles.

    Args:
        documents: Dictionary mapping document identifier/name to text content.

    Returns:
        dict: Mean CLI, variance, min, max, and pairwise consistency flag.
    """
    if not documents:
        return {
            "document_count": 0,
            "mean_readability": 0.0,
            "min_readability": 0.0,
            "max_readability": 0.0,
            "readability_range": 0.0,
            "is_homogeneous": True,
            "per_document": {},
        }

    per_doc = {name: compute_coleman_liau_index(text) for name, text in documents.items()}
    scores = list(per_doc.values())
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    # Documents within 2 grade levels are considered stylistically homogeneous
    is_homogeneous = score_range <= 2.0

    return {
        "document_count": len(documents),
        "mean_readability": round(mean_score, 2),
        "min_readability": round(min_score, 2),
        "max_readability": round(max_score, 2),
        "readability_range": round(score_range, 2),
        "is_homogeneous": is_homogeneous,
        "per_document": per_doc,
    }


def compute_text_stats(text: str) -> dict[str, float]:
    """
    Compute all text statistics for the given text.

    Issue #3704: Includes 'readability_score' calculated via the Coleman-Liau index.

    Args:
        text: The text to analyze

    Returns:
        Dictionary containing:
            - word_count: Total number of words
            - sentence_count: Total number of sentences
            - unique_word_count: Number of unique words
            - unique_word_ratio: Ratio of unique words to total words (0-1)

            - avg_word_length: Average number of characters per word
            - avg_sentence_length: Average number of words per sentence

            - readability_score: Coleman-Liau readability index grade level

    """
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    unique_word_count = count_unique_words(text)
    unique_word_ratio = get_unique_word_ratio(text)
    readability_score = compute_coleman_liau_index(text)

    words = re.findall(r"\w+", text)

    avg_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0
        else 0.0
    )

    avg_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0.0
    )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_word_count": unique_word_count,
        "unique_word_ratio": unique_word_ratio,

        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,

        "readability_score": readability_score,
    }


def format_stats_for_pdf(stats: dict[str, float]) -> list[list[str]]:
    """
    Format text statistics for display in a PDF table.

    Args:
        stats: Dictionary of text statistics from compute_text_stats()

    Returns:
        List of rows for a ReportLab Table, each row is [Metric, Value]
    """
    rows = [
        ["Word Count", str(stats["word_count"])],
        ["Sentence Count", str(stats["sentence_count"])],
        ["Unique Words", str(stats["unique_word_count"])],
        ["Unique Word Ratio", f"{stats['unique_word_ratio']:.2%}"],
    ]
    if "readability_score" in stats:
        rows.append(["Readability Grade (CLI)", f"{stats['readability_score']:.2f}"])
    return rows


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


def count_syllables(word: str) -> int:
    """
    Estimate the syllable count of a single English word.

    Uses regex-based vowel group counting with silent 'e' deductions.

    Args:
        word: The word to analyze

    Returns:
        Estimated syllable count (at least 1 for non-empty words)
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Strip non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    # Count vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)

    # Silent 'e' deductions
    if word.endswith("e"):
        vowels = "aeiouy"
        is_consonant_le = (
            len(word) >= 3 and word.endswith("le") and word[-3] not in vowels
        )
        if not is_consonant_le:
            count -= 1

    # Return at least 1 syllable for non-empty words
    return max(1, count)


def count_syllables_in_word(word: str) -> int:
    """Estimate the syllable count of a single word using basic heuristics."""
    return count_syllables(word)


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
