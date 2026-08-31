# semantic-plagiarism-detector/tests/test_patchwriting_detector.py

import pytest
from src.core.pos_normalizer import POSNormalizer
from src.core.patchwriting_detector import PatchwritingDetector

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
