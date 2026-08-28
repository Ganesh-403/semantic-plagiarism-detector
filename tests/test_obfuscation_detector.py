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
