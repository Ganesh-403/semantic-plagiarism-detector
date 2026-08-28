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
tests/core/test_detect_chunk_language.py
-----------------------------------------
Unit tests for detect_chunk_language() with various languages and edge cases.

Fixes #2250
"""

from __future__ import annotations

from src.core.cross_lingual import detect_chunk_language


def test_detects_english_returns_en():
    text = (
        "The student submitted their assignment to the teacher. "
        "The professor reviewed the work and provided detailed feedback."
    )
    assert detect_chunk_language(text) == "en"


def test_detects_spanish():
    text = (
        "El estudiante presentó su trabajo al profesor. "
        "La universidad evaluó los resultados del examen con mucho cuidado."
    )
    assert detect_chunk_language(text) == "es"


def test_detects_french():
    text = (
        "Le professeur a corrigé les copies des étudiants. "
        "Les résultats ont été publiés sur le tableau d'affichage de l'université."
    )
    assert detect_chunk_language(text) == "fr"


def test_detects_german():
    text = (
        "Der Student hat seine Hausarbeit rechtzeitig abgegeben. "
        "Die Universität bewertet die Leistungen der Studierenden sehr sorgfältig."
    )
    assert detect_chunk_language(text) == "de"


def test_detects_chinese():
    text = "这是一个关于人工智能的研究论文，探讨了机器学习的最新进展和应用。"
    assert detect_chunk_language(text) == "zh"


def test_detects_japanese():
    text = "人工知能は教育分野においても重要な役割を果たしています。学生の学習を支援するシステムが開発されています。"
    assert detect_chunk_language(text) == "ja"


def test_empty_string_returns_default():
    assert detect_chunk_language("") == "en"


def test_none_input_returns_default():
    assert detect_chunk_language(None) == "en"


def test_non_string_input_returns_default():
    assert detect_chunk_language(123) == "en"
    assert detect_chunk_language([]) == "en"


def test_too_short_text_returns_default():
    assert detect_chunk_language("hola") == "en"
    assert detect_chunk_language("bonjour monde") == "en"


def test_numbers_only_returns_default():
    assert detect_chunk_language("123 456 789 1011 1213") == "en"


def test_whitespace_only_returns_default():
    assert detect_chunk_language("     ") == "en"


def test_returns_string_type():
    result = detect_chunk_language("some text here to analyze language")
    assert isinstance(result, str)


def test_returns_valid_iso_code():
    text = "The student submitted the assignment to the professor for review."
    result = detect_chunk_language(text)
    assert len(result) <= 3
    assert result.isalpha()


def test_mixed_language_does_not_crash():
    text = "The student dit que le travail était muy difícil para todos."
    result = detect_chunk_language(text)
    assert isinstance(result, str)
