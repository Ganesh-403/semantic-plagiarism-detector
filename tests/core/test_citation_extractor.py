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
