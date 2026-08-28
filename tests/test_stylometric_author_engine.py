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

"""
Unit tests for Enterprise Stylometric Authorship Attribution & Write-Print Engine
"""

import pytest

from src.services.stylometric_author_engine import (
    AuthorshipAttributionClassifier,
    StylometricWriteprintExtractor,
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
