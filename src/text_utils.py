"""Text utilities for the semantic plagiarism detector.

Pure helpers used when normalizing document snippets before embedding.
"""


def slugify(text):
    """Return a lowercase, dash-separated slug for ids and cache keys."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def truncate(text, limit=200, suffix="..."):
    """Truncate `text` to `limit` characters, appending `suffix` if cut."""
    if len(text) <= limit:
        return text
    return text[: limit - len(suffix)] + suffix


def word_count(text):
    """Return the number of whitespace-delimited words in `text`."""
    return len(text.split()) if text else 0


def strip_citations(text):
    """Remove bracketed and parenthetical citation markers from `text`.

    Handles forms like ``[1]``, ``[1, 2]`` and ``(Author, 2020)`` so that
    boilerplate references don't pollute document embeddings.
    """
    import re
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\d{4}[^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text):
    """Split `text` into sentences on `.`, `!` and `?` boundaries."""
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


if __name__ == "__main__":
    assert slugify("Hello World!") == "hello-world"
    assert truncate("x" * 50, 10) == "x" * 7 + "..."
    assert word_count("a b c") == 3
    assert strip_citations("See [1] and (Smith, 2020) for details.") == "See and for details."
    assert sentence_split("Hello world! How are you?") == ["Hello world!", "How are you?"]
    print("text_utils self-test passed")
