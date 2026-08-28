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

# semantic-plagiarism-detector/tests/test_paraphrase_fingerprinter.py

import pytest

from src.core.paraphrase_fingerprinter import ParaphraseFingerprinter
from src.db.tool_signatures_db import ToolSignaturesDB


def test_feature_extraction():
    sample_text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming education."
    features = ParaphraseFingerprinter.extract_fingerprint(sample_text)

    assert "sentence_length_variance" in features
    assert "synonym_entropy" in features
    assert isinstance(features["synonym_entropy"], float)


def test_tool_signature_matching():
    db = ToolSignaturesDB()
    # Test features matching Quillbot Standard signature profile
    features = {
        "synonym_entropy": 4.2,
        "sentence_length_variance": 12.5,
        "burstiness_index": 0.3,
    }
    match = db.match_signature(features)

    assert "attributed_tool" in match
    assert "confidence_score" in match
    assert match["attributed_tool"] == "Quillbot_Standard"
    assert match["confidence_score"] > 0.8
