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

import pytest

from src.citation_integrity_verifier import CitationIntegrityVerifier


def test_citation_integrity_verifier():
    verifier = CitationIntegrityVerifier()
    citations = [
        {
            "doi": "10.1038/s41586-021-03819-2",
            "title": "AlphaFold Protein Structure",
            "year": "2021",
        },
        {
            "doi": "",
            "title": "Fake Unknown Paper",
            "year": "2024",
        },
        {
            "doi": "10.9999/fake-doi-123",
            "title": "Hallucinated AI Paper",
            "year": "2025",
        },
    ]

    result = verifier.verify_manuscript_citations(citations)
    assert result["totalCitationsAnalyzed"] == 3
    assert result["validCitationsCount"] == 1
    assert result["flaggedCitationsCount"] == 2
    assert result["citationIntegrityScorePct"] == 33.33
