"""Targeted tests for APA, IEEE, and MLA citation extraction.

Issue #2033.
"""

import re

from src.core.citation_extractor import extract_citations


def _assert_citation(citation, *, author, year, title):
    assert citation["author"] == author
    assert citation["year"] == year
    assert citation["title"] == title
    assert re.fullmatch(r"[0-9a-f]{64}", citation["hash"])


def test_apa_citation():
    """APA bibliography entries extract author, year, title, and hash."""
    citations = extract_citations("Smith, J. A. (2020). A study. Journal.")

    assert len(citations) == 1
    _assert_citation(
        citations[0],
        author="Smith, J. A.",
        year="2020",
        title="A study",
    )


def test_ieee_citation():
    """IEEE bibliography entries extract author, year, title, and hash."""
    citations = extract_citations('[1] J. Smith, "A study," Journal, 2020.')

    assert len(citations) == 1
    _assert_citation(
        citations[0],
        author="J. Smith",
        year="2020",
        title="A study",
    )


def test_mla_citation():
    """MLA bibliography entries extract author, year, title, and hash."""
    citations = extract_citations('Smith, John. "A study." Journal, 2020.')

    assert len(citations) == 1
    _assert_citation(
        citations[0],
        author="Smith, John",
        year="2020",
        title="A study",
    )
