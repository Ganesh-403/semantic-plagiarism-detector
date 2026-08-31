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
    features = {"synonym_entropy": 4.2, "sentence_length_variance": 12.5, "burstiness_index": 0.3}
    match = db.match_signature(features)
    
    assert "attributed_tool" in match
    assert "confidence_score" in match
    assert match["attributed_tool"] == "Quillbot_Standard"
    assert match["confidence_score"] > 0.8
