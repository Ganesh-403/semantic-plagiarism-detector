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

from src.ocr.neural_alignment_engine import NeuralAlignmentEngine
from src.ocr.pipeline_orchestrator import OcrPipelineOrchestrator


def test_sanitization_removes_invisible_characters():
    engine = NeuralAlignmentEngine()
    obfuscated_input = "E\u200Bv\u200Ba\u200Bs\u200Bi\u200Bo\u200Bn Text"

    clean_output, changes_made = engine.sanitize_and_clean_text(obfuscated_input)

    assert clean_output == "Evasion Text"
    assert changes_made == 5


def test_alignment_vector_precision_scoring():
    engine = NeuralAlignmentEngine()
    source = "The quick brown fox jumps over the lazy dog"
    target = "The quick brown fox jumps over the lazy hound"

    score = engine.compute_alignment_vectors(source, target)

    # 8 identical words out of 10 unique elements total cross-union = 80.0%
    assert score == 80.00


def test_orchestrator_pipeline_execution():
    orchestrator = OcrPipelineOrchestrator()
    mock_bbox = {"x0": 10, "y0": 20, "x1": 150, "y1": 80}

    result = orchestrator.process_multimodal_block(
        mock_db_client=None,
        document_id="doc_uuid_12345",
        page=1,
        raw_extracted_text="Sample text content block.",
        bbox=mock_bbox,
        reference_text="Sample text content block.",
    )

    assert result["success"] is True
    assert result["alignment_score"] == 100.00
