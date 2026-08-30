"""Tests for Markdown header chunk boundaries (Issue #4000)."""

from src.core.text_chunking import (
    chunk_by_sentences,
    chunk_text,
    split_at_markdown_headers,
)


def test_split_at_markdown_headers_forces_section_breaks():
    text = (
        "Intro paragraph before any heading.\n"
        "# First Section\n"
        "Body under first.\n"
        "## Nested\n"
        "More body.\n"
        "### Deeper\n"
        "Tail."
    )
    parts = split_at_markdown_headers(text)
    assert len(parts) == 4
    assert parts[0].startswith("Intro")
    assert parts[1].startswith("# First Section")
    assert parts[2].startswith("## Nested")
    assert parts[3].startswith("### Deeper")


def test_chunk_text_does_not_join_heading_to_prior_prose():
    text = (
        "Unrelated preamble that should stay alone from the heading.\n\n"
        "# Methods\n"
        "The methods section describes the approach used in this study."
    )
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20, min_words=3)
    texts = [c.text if hasattr(c, "text") else c for c in chunks]
    # Heading must not share a chunk with the preamble.
    for t in texts:
        if t.lstrip().startswith("# Methods"):
            assert "Unrelated preamble" not in t
            break
    else:
        raise AssertionError("expected a chunk starting at # Methods")


def test_chunk_by_sentences_respects_markdown_headers():
    text = (
        "Preamble sentence one. Preamble sentence two.\n"
        "# Results\n"
        "Result sentence one. Result sentence two."
    )
    chunks = chunk_by_sentences(text, max_chunks=20, min_words=2, min_chunk_length=5)
    heading_chunk = next(c for c in chunks if "# Results" in c)
    assert "Preamble" not in heading_chunk
