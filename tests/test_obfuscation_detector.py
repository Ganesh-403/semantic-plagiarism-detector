import pytest
from src.security.obfuscation_detector import ObfuscationDetector

def test_clean_text_produces_zero_score():
    detector = ObfuscationDetector()
    clean_text = "The quick brown fox jumps over the lazy dog. Pristine academic writing samples."
    metrics = detector.analyze_text(clean_text)
    
    assert metrics["obfuscation_score"] == 0.0
    assert not metrics["is_flagged"]
    assert len(metrics["invisible_indices"]) == 0

def test_zero_width_character_injection_detection():
    detector = ObfuscationDetector()
    # Injects zero-width space characters (\u200B) between letters
    obfuscated_text = "P\u200Bl\u200Ba\u200Bg\u200Bi\u200Ba\u200Br\u200Bi\u200Bs\u200Bm"
    metrics = detector.analyze_text(obfuscated_text)
    
    assert metrics["obfuscation_score"] > 0.0
    assert metrics["is_flagged"]
    assert len(metrics["invisible_indices"]) == 9

def test_cyrillic_homoglyph_substitution_detection():
    detector = ObfuscationDetector()
    # Uses Cyrillic 'а' (U+0430) inside the word 'cat'
    mixed_script_text = "The c\u0430t slept on the rug." 
    metrics = detector.analyze_text(mixed_script_text)
    
    assert metrics["obfuscation_score"] > 0.0
    assert len(metrics["homoglyph_indices"]) == 1
