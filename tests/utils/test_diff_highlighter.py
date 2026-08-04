"""Unit tests for the diff highlighting utility."""

from src.utils.diff_highlighter import highlight_overlap


def test_highlight_overlap_exact_match():
    """Verify that exact matching sub-segments are highlighted in yellow."""
    text_a = "This is a very long sequence of words that is matching exactly here."
    text_b = "This is another sequence of words that is matching exactly here."

    html_a, html_b = highlight_overlap(text_a, text_b, min_match_len=15)

    # The matching part "sequence of words that is matching exactly here." is 47 characters.
    # It should be wrapped in the styled <mark> tag.
    mark_style = (
        "style='background-color: rgba(250, 204, 21, 0.3); "
        "color: inherit; padding: 1px 3px; border-radius: 3px;'"
    )
    assert f"<mark {mark_style}>" in html_a
    assert f"<mark {mark_style}>" in html_b
    assert "matching exactly here." in html_a
    assert "matching exactly here." in html_b


def test_highlight_overlap_no_match():
    """Verify that strings with no overlap are returned fully escaped but unhighlighted."""
    text_a = "Abc def ghi"
    text_b = "Xyz opq rst"

    html_a, html_b = highlight_overlap(text_a, text_b, min_match_len=5)

    assert "<mark" not in html_a
    assert "<mark" not in html_b
    assert html_a == "Abc def ghi"
    assert html_b == "Xyz opq rst"


def test_highlight_overlap_below_threshold():
    """Verify that matches below the minimum character length threshold are ignored."""
    text_a = "Match here"
    text_b = "Match also"

    # Match "Match" is 5 characters, which is below min_match_len of 10.
    html_a, html_b = highlight_overlap(text_a, text_b, min_match_len=10)

    assert "<mark" not in html_a
    assert "<mark" not in html_b


def test_highlight_overlap_html_escaping():
    """Verify that HTML special characters are escaped to prevent XSS."""
    text_a = "Match <b>bold HTML</b> and check code & characters"
    text_b = "Match <b>bold HTML</b> and check other & symbols"

    html_a, _ = highlight_overlap(text_a, text_b, min_match_len=10)

    assert "<b>" not in html_a
    assert "&amp;" in html_a
    assert "&lt;b&gt;" in html_a


def test_highlight_overlap_markdown_escaping():
    """Verify that Markdown formatting characters are escaped to prevent rendering bugs."""
    text_a = "Match *bold markdown* and _italic_ and [link]()"
    text_b = "Match *bold markdown* and _italic_ and [other link]()"

    html_a, _ = highlight_overlap(text_a, text_b, min_match_len=10)

    # Markdown characters should be backslash-escaped
    assert "\\*" in html_a
    assert "\\_" in html_a
    assert "\\[" in html_a


def test_highlight_overlap_empty_inputs():
    """Verify that empty inputs are handled gracefully without crashing."""
    html_a, html_b = highlight_overlap("", "some text")
    assert html_a == ""
    assert html_b == "some text"
