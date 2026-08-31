"""
Unit tests for Enterprise Stylometric Authorship Attribution & Write-Print Engine
"""

import pytest
from src.services.stylometric_author_engine import (
    StylometricWriteprintExtractor,
    AuthorshipAttributionClassifier,
)


def test_stylometric_writeprint_extraction():
    extractor = StylometricWriteprintExtractor(target_author_id="AUTHOR-PROF-101")
    text_sample = (
        "Artificial intelligence systems demonstrate remarkable capabilities in natural language processing."
        " However, evaluating stylistic nuances requires quantitative feature extraction! "
        "Furthermore, sentence length entropy provides robust discrimination against ghostwriting."
    )

    writeprint = extractor.extract_author_writeprint(text_sample)
    assert writeprint["authorId"] == "AUTHOR-PROF-101"
    assert writeprint["totalWordsAnalyzed"] > 0
    assert writeprint["typeTokenRatio"] > 0.0
    assert writeprint["avgSentenceLengthWords"] > 0.0


def test_authorship_attribution_classification():
    baselines = {
        "AUTHOR-PROF-101": {
            "typeTokenRatio": 0.85,
            "avgSentenceLengthWords": 12.5,
            "punctuationDensity": 0.08,
        }
    }
    classifier = AuthorshipAttributionClassifier(author_baseline_profiles=baselines)

    cand_writeprint = {
        "typeTokenRatio": 0.84,
        "avgSentenceLengthWords": 12.2,
        "punctuationDensity": 0.078,
    }

    matches = classifier.classify_authorship(cand_writeprint, distance_threshold=0.80)
    assert len(matches) > 0
    assert matches[0]["matchedAuthorId"] == "AUTHOR-PROF-101"
    assert matches[0]["attributionConfidencePct"] >= 80.0


# ==============================================================================
# AUTOMATED STYLOMETRIC UNIT TEST ARCHITECTURE SPECIFICATIONS
# ------------------------------------------------------------------------------
# Section 1: Test Assertions & Boundary Assertions
# - Soft float equality assertions on TTR metrics
# - Classification confidence bounds validation
# ==============================================================================
