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


if __name__ == "__main__":
    assert slugify("Hello World!") == "hello-world"
    assert truncate("x" * 50, 10) == "x" * 7 + "..."
    assert word_count("a b c") == 3
    print("text_utils self-test passed")
