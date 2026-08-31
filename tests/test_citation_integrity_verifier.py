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
