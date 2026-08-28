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

from __future__ import annotations

from src.core.cross_lingual import (
    detect_language,
    prepare_chunks_for_embedding,
    prepare_documents_for_embedding,
    prepare_text_for_embedding,
)


def test_detects_english_text():
    text = (
        "Artificial intelligence helps teachers provide faster feedback "
        "and personalise classroom learning."
    )
    assert detect_language(text) == "en"


def test_detects_hindi_text():
    text = (
        "कृत्रिम बुद्धिमत्ता शिक्षकों को विद्यार्थियों के लिए व्यक्तिगत "
        "शिक्षण सामग्री तैयार करने में सहायता करती है।"
    )
    assert detect_language(text) == "hi"


def test_english_text_is_not_translated():
    calls = []

    def fake_translator(*args, **kwargs):
        calls.append((args, kwargs))
        return "should not be used"

    result = prepare_text_for_embedding(
        "Artificial intelligence supports modern education.",
        detector=lambda _: "en",
        translator=fake_translator,
    )

    assert result["original_text"] == result["embedding_text"]
    assert result["detected_language"] == "en"
    assert result["translated"] is False
    assert calls == []


def test_non_english_text_is_translated_for_embedding_only():
    original = "La inteligencia artificial ayuda a los profesores."

    result = prepare_text_for_embedding(
        original,
        detector=lambda _: "es",
        translator=lambda text, target_lang: f"[translated {text}]",
    )

    assert result["original_text"] == original
    assert result["detected_language"] == "es"
    assert result["translated"] is True
    assert (
        result["embedding_text"]
        == "[translated La inteligencia artificial ayuda a los profesores.]"
    )


def test_detect_language_with_chunk_record():
    text = "Bonjour le monde"
    lang = detect_language(text)
    assert lang == "fr"


def test_prepare_documents_for_embedding_merges_by_language():
    """Prepare documents for embedding groups by detected language."""
    docs = {
        "english": "This is an English document.",
        "spanish": "Este es un documento en español.",
        "hindi": "यह एक हिंदी दस्तावेज़ है।",
    }

    result = prepare_documents_for_embedding(docs)

    assert isinstance(result, dict)
    for doc_name in docs:
        assert doc_name in result


def test_prepare_chunks_for_embedding():
    """Prepare chunks for embedding."""
    chunks = {
        "doc1": [
            "Artificial intelligence is transforming education.",
            "Machine learning models can predict student outcomes.",
        ],
        "doc2": [
            "Data science is a multidisciplinary field.",
        ],
    }

    result = prepare_chunks_for_embedding(chunks)

    assert isinstance(result, dict)
    for doc_name in chunks:
        assert doc_name in result
