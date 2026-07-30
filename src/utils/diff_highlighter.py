"""Diff highlighting utility for exact matches between text chunks."""

import difflib
import re


def highlight_overlap(
    text_a: str,
    text_b: str,
    min_match_len: int = 10,
    theme_colors: dict | None = None,
) -> tuple[str, str]:
    """Compare two text chunks at the word/token level and wrap exact matching

    substrings in a visually styled HTML <mark> tag.

    Args:
        text_a: First text chunk.
        text_b: Second text chunk.
        min_match_len: Minimum matching character length (trimmed) to qualify for highlighting.

    Returns:
        Tuple of (highlighted_html_a, highlighted_html_b).
    """
    if not text_a or not text_b:
        return (
            _escape_text(text_a or ""),
            _escape_text(text_b or ""),
        )

    # Tokenise into words and non-words to preserve spacing and punctuation
    tokens_a = re.findall(r"\w+|\W+", text_a)
    tokens_b = re.findall(r"\w+|\W+", text_b)

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    matching_blocks = matcher.get_matching_blocks()

    highlight_a = [False] * len(tokens_a)
    highlight_b = [False] * len(tokens_b)

    for match in matching_blocks:
        if match.size == 0:
            continue

        match_tokens = tokens_a[match.a : match.a + match.size]
        match_str = "".join(match_tokens)

        # Highlight sequence if it is long enough and contains alphanumeric words
        if len(match_str.strip()) >= min_match_len and any(
            c.isalnum() for c in match_str
        ):
            for i in range(match.a, match.a + match.size):
                highlight_a[i] = True
            for i in range(match.b, match.b + match.size):
                highlight_b[i] = True

    return (
    _build_html(tokens_a, highlight_a, theme_colors),
    _build_html(tokens_b, highlight_b, theme_colors),
)


def _escape_text(text: str) -> str:
    """Escape HTML and Markdown syntax characters."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for m_char in ["*", "_", "~", "`", "#", "[", "]", "(", ")"]:
        escaped = escaped.replace(m_char, f"\\{m_char}")
    return escaped


def _build_html(
    tokens: list[str],
    highlight_mask: list[bool],
    theme_colors: dict | None = None,
) -> str:
    """Build the final HTML string by grouping highlighted tokens inside <mark> tags."""
    parts = []
    in_highlight = False

    for token, should_highlight in zip(tokens, highlight_mask):
        escaped_token = _escape_text(token)

        if should_highlight:
            if not in_highlight:
                highlight_bg = (
                    theme_colors.get("warning_soft", "rgba(250, 204, 21, 0.3)")
                    if theme_colors
                    else "rgba(250, 204, 21, 0.3)"
                )

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
