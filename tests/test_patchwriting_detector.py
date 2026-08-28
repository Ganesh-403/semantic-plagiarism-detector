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

# semantic-plagiarism-detector/tests/test_patchwriting_detector.py

import pytest

from src.core.patchwriting_detector import PatchwritingDetector
from src.core.pos_normalizer import POSNormalizer


def test_pos_extraction_accuracy():
    text = "The quick brown fox jumps."
    tags = POSNormalizer.extract_pos_sequence(text)
    assert len(tags) > 0
    assert isinstance(tags, list)


def test_syntactic_similarity_scoring():
    source = "The diligent engineer designed a scalable distributed system."
    # Patchwritten version: swapped nouns/verbs with structural equivalence
    student = "The active architect created a modular robust platform."

    result = PatchwritingDetector.compute_syntactic_similarity(source, student)

    assert "similarity_score" in result
    assert "ngram_similarity" in result
    assert 0.0 <= result["similarity_score"] <= 1.0
    # High structural similarity expected due to matching POS sequence (DET ADJ NOUN VERB...)
    assert result["similarity_score"] > 0.4
