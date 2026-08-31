"""
src/utils/diff_highlighter.py
-----------------------------
Utilities for highlighting overlapping text segments between two documents.

Provides HTML rendering functions that visually emphasize matching phrases,
words, or character sequences to help instructors quickly identify plagiarized
content in side-by-side comparison views.
"""

from __future__ import annotations

import functools
import html
import re
from typing import Callable, Optional, Tuple

from src.core.config import DEFAULT_DIFF_MIN_MATCH_LENGTH

#: Inline style applied to every highlighted run. Kept as a module constant so
#: the markup is identical everywhere and testable without a regex.
MARK_OPEN_TAG = (
    '<mark style="background-color: #fef08a; padding: 2px 4px; border-radius: 3px;">'
)

#: CJK ranges called out in #3213: Han ideographs plus Hiragana/Katakana.
_CJK_RANGE = r"\u4e00-\u9fff\u3040-\u30ff"

#: Token scanner used everywhere tokens must line up with character offsets.
#:
#: Chinese and Japanese write without spaces, so the previous ``\b\w+\b``
#: scanner swallowed an entire sentence into one token and partial overlaps
#: could never be seen. Here every CJK character stands alone — the n-gram
#: window in ``highlight_overlap`` thereby becomes a *character* n-gram for
#: such text — while non-CJK runs keep the original word grouping. The word
#: branch excludes the CJK ranges explicitly rather than relying on ``\w``
#: (which matches CJK and would otherwise absorb them); alternation order
#: makes the single-character branch win first regardless.
_TOKEN_SPAN_RE = re.compile(rf"[{_CJK_RANGE}]|[^\W{_CJK_RANGE}]+")


def _tokenize(text: str) -> list[str]:
    """Split *text* into lowercased tokens.

    Non-CJK runs are grouped into words exactly as before. Each CJK
    character (Han, Hiragana, Katakana) becomes its own token because
    those scripts separate words with neither spaces nor punctuation,
    which turns the n-gram window in :func:`highlight_overlap` into a
    character n-gram match (#3213).

    Args:
        text: Raw document text.

    Returns:
        The tokens, in order.
    """
    return [match.group().lower() for match in _TOKEN_SPAN_RE.finditer(text)]


#: Vowels used by the fallback stemmer's consonant-doubling cleanup.
_VOWELS = frozenset("aeiou")


def _fallback_stem(token: str) -> str:
    """Conservative Porter-style suffix stripper for when NLTK is unavailable.

    Handles the regular English inflections that matter most for fuzzy
    overlap (-s/-es, -ied/-ies, -ed, -ing) well enough to keep the feature
    useful; it is deliberately *not* a full Porter implementation. Words of
    three characters or fewer are returned untouched so short tokens and
    acronyms are never mangled.
    """
    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        candidate = token[:-1]
        return candidate if len(candidate) >= 3 else token

    for suffix in ("ing", "edly", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if not any(char in _VOWELS for char in base):
                return token
            # Undo the doubled final consonant ("running" -> "runn" -> "run"),
            # except where doubling is part of the stem ("falling" -> "fall").
            if (
                len(set(base[-2:])) == 1
                and base[-1] not in _VOWELS
                and base[-1] not in ("l", "s", "z")
            ):
                base = base[:-1]
            return base

    return token


@functools.lru_cache(maxsize=1)
def _get_stem_function() -> Callable[[str], str]:
    """Resolve the word-stemming function used by fuzzy overlap matching.

    Prefers ``nltk.stem.PorterStemmer`` (NLTK is a core dependency) and falls
    back to :func:`_fallback_stem` when NLTK cannot be imported, so enabling
    stemming never crashes a minimal install.

    Returns:
        A callable mapping a lowercased word token to its stem.
    """
    try:
        from nltk.stem import PorterStemmer  # type: ignore

        return PorterStemmer().stem
    except Exception:
        return _fallback_stem


def _covered_word_ranges(
    words: list[str],
    other_ngrams: set[tuple[str, ...]],
    window: int,
) -> list[tuple[int, int]]:
    """Return the merged word ranges of *words* that also occur in the other text.

    A word position is part of a match exactly when some window of ``window``
    consecutive words containing it also appears in the other document. Testing
    each window against a set of the other document's windows costs one hash
    lookup, so the whole scan is linear in the number of words.

    Args:
        words: Tokens of the document being highlighted.
        other_ngrams: Every ``window``-sized tuple of tokens from the other
            document.
        window: Match length in words.

    Returns:
        Half-open ``(start, end)`` word ranges, sorted and already merged, so
        no two ranges touch or overlap and ``<mark>`` tags cannot nest.
    """
    ranges: list[tuple[int, int]] = []

    for index in range(len(words) - window + 1):
        if tuple(words[index : index + window]) not in other_ngrams:
            continue

        # Positions are visited left to right, so a new window either extends
        # the range we are already inside or starts a fresh one.
        if ranges and index <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], index + window))
        else:
            ranges.append((index, index + window))

    return ranges


def _apply_marks(
    text: str,
    word_ranges: list[tuple[int, int]],
    css_class: Optional[str] = None,
) -> str:
    """Wrap the given word ranges of *text* in ``<mark>`` tags.

    Args:
        text: The original, unescaped document text.
        word_ranges: Half-open, sorted, non-overlapping word ranges.
        css_class: Optional CSS class name to apply to ``<mark>`` tags.
            When provided, ``<mark class="...">`` is rendered instead of
            inline styles.

    Returns:
        HTML-escaped text with the matching runs wrapped in ``<mark>``.
    """
    if not word_ranges:
        return html.escape(text)

    # Must use the same scanner as _tokenize so word indices line up with
    # the token indices the ranges refer to.
    word_positions = [(m.start(), m.end()) for m in _TOKEN_SPAN_RE.finditer(text)]
    if not word_positions:
        return html.escape(text)

    open_tag = f'<mark class="{html.escape(css_class)}">' if css_class else MARK_OPEN_TAG

    result: list[str] = []
    last_end = 0

    for start_word, end_word in word_ranges:
        if start_word >= len(word_positions):
            continue

        char_start = word_positions[start_word][0]
        char_end = word_positions[min(end_word - 1, len(word_positions) - 1)][1]

        result.append(html.escape(text[last_end:char_start]))
        result.append(open_tag)
        result.append(html.escape(text[char_start:char_end]))
        result.append("</mark>")

        last_end = char_end

    result.append(html.escape(text[last_end:]))
    return "".join(result)


def highlight_overlap(
    text_a: str,
    text_b: str,
    min_match_length: int = DEFAULT_DIFF_MIN_MATCH_LENGTH,
    use_stemming: bool = False,
    css_class: Optional[str] = None,
) -> tuple[str, str]:
    """Highlight overlapping sequences between two text strings.

    Identifies common word sequences of at least `min_match_length` words
    and wraps them in HTML <mark> tags with a distinct background color or
    custom CSS class.
    This helps instructors visually identify plagiarized phrases while
    ignoring common stop words and short coincidental matches.

    The previous implementation compared every start offset in one document
    against every start offset in the other and then walked each pair forward,
    which is cubic on the input this function exists for — two documents that
    are largely the same. Two 3,000-word documents took 24 seconds and blocked
    the Streamlit script thread for the whole time.

    A word is now part of a match exactly when some window of
    ``min_match_length`` consecutive words containing it also appears in the
    other document. That is the same set of highlighted positions the old
    nested scan produced once its overlapping ranges were merged — the merge
    was already collapsing every redundant alignment — but it costs one hash
    lookup per word instead of one walk per offset pair.

    Args:
        text_a: The first document's text chunk.
        text_b: The second document's text chunk.
        min_match_length: Minimum number of consecutive words required to
                         constitute a "match". Defaults to 4 to avoid
                         highlighting common phrases like "in the" or "and the".
                         Values below 1 are treated as 1. For CJK text every
                         character is a token, so this counts characters.

    Returns:
        A tuple of two HTML strings (highlighted_a, highlighted_b) with
        matching sequences wrapped in <mark> tags. Returns escaped HTML
        to prevent XSS vulnerabilities.

    Examples:
        Two shared words fall short of the four-word default, so nothing is
        marked:

        >>> a, b = highlight_overlap("the quick brown fox", "a quick brown dog")
        >>> "<mark" in a or "<mark" in b
        False

        Lowering the threshold picks the shared run up in both documents:

        >>> a, b = highlight_overlap("the quick brown fox", "a quick brown dog", 2)
        >>> "<mark" in a and "<mark" in b
        True
        >>> "quick brown" in a and "quick brown" in b
        True

        Chinese has no spaces, so it used to tokenize into one opaque
        sentence-sized token per document and nothing ever matched. Each
        character is now its own token, and the shared opening run is found:

        >>> a, b = highlight_overlap(
        ...     "今天天气很好我们一起去公园",
        ...     "今天天气很好他们待在家里",
        ...     min_match_length=5,
        ... )
        >>> "<mark" in a and "<mark" in b
        True
        >>> "今天天气很好" in a and "今天天气很好" in b
        True
    """
    if not text_a or not text_b:
        return html.escape(text_a or ""), html.escape(text_b or "")

    words_a = _tokenize(text_a)
    words_b = _tokenize(text_b)

    if not words_a or not words_b:
        return html.escape(text_a), html.escape(text_b)

    window = max(1, min_match_length)
    if len(words_a) < window or len(words_b) < window:
        return html.escape(text_a), html.escape(text_b)

    if use_stemming:
        # Stemming only rewrites token *values*; it never changes how many
        # tokens there are, so the ranges computed below still line up with
        # the original character positions that _apply_marks highlights.
        stem = _get_stem_function()
        words_a = [stem(word) for word in words_a]
        words_b = [stem(word) for word in words_b]

    ngrams_a = {
        tuple(words_a[i : i + window]) for i in range(len(words_a) - window + 1)
    }
    ngrams_b = {
        tuple(words_b[j : j + window]) for j in range(len(words_b) - window + 1)
    }

    # Each document is highlighted against the *other* document's windows, so
    # the index built from B decides what gets marked in A and vice versa.
    ranges_a = _covered_word_ranges(words_a, ngrams_b, window)
    ranges_b = _covered_word_ranges(words_b, ngrams_a, window)

    return _apply_marks(text_a, ranges_a, css_class=css_class), _apply_marks(
        text_b, ranges_b, css_class=css_class
    )


def _escape_text(text: str) -> str:
    """Escape HTML and Markdown syntax characters."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for m_char in ["*", "_", "~", "`", "#", "[", "]", "(", ")", "|", "{", "}"]:
        escaped = escaped.replace(m_char, f"\\{m_char}")
    return escaped


def _sanitize_color(color: str, fallback: str = "rgba(250, 204, 21, 0.3)") -> str:
    """Sanitize CSS color value to prevent HTML/CSS/JS injection."""
    if not isinstance(color, str):
        return fallback
    color_trimmed = color.strip()
    if any(c in color_trimmed for c in ("'", '"', ";", "<", ">")):
        return fallback
    if re.match(
        r"^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*[\d\.]+%?)?\s*\)$",
        color_trimmed,
    ):
        return color_trimmed
    if any(c in color_trimmed for c in ("(", ")")):
        return fallback
    if re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$|^[a-zA-Z]+$", color_trimmed):
        return color_trimmed
    return fallback


def _build_html(
    tokens: list[str],
    highlight_mask: list[bool],
    theme_colors: dict[str, str] | None = None,
) -> str:
    """Build the final HTML string by grouping highlighted tokens inside <mark> tags."""
    parts = []
    in_highlight = False

    for token, should_highlight in zip(tokens, highlight_mask):
        escaped_token = _escape_text(token)

        if should_highlight:
            if not in_highlight:
                raw_bg = (
                    theme_colors.get("warning_soft", "rgba(250, 204, 21, 0.3)")
                    if theme_colors
                    else "rgba(250, 204, 21, 0.3)"
                )
                highlight_bg = _sanitize_color(raw_bg)

                parts.append(
                    f"<mark style='background-color: {highlight_bg}; "
                    "color: inherit; padding: 1px 3px; border-radius: 3px;'>"
                )
                in_highlight = True

            parts.append(escaped_token)
        else:
            if in_highlight:
                parts.append("</mark>")
                in_highlight = False
            parts.append(escaped_token)

    if in_highlight:
        parts.append("</mark>")

    return "".join(parts)
