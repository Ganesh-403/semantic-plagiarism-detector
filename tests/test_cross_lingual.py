from __future__ import annotations

import pytest

from src.core.cross_lingual import (
    detect_language,
    prepare_chunks_for_embedding,
    prepare_documents_for_embedding,
    prepare_text_for_embedding,
)


@pytest.fixture(autouse=True)
def isolate_translation_cache(monkeypatch):
    from src.core import cross_lingual
    monkeypatch.setattr(cross_lingual, "TRANSLATION_MEMORY_CACHE", cross_lingual.TranslationMemoryCache())
    monkeypatch.setattr(cross_lingual, "get_cached_translation", lambda *args: None)
    monkeypatch.setattr(cross_lingual, "save_translation", lambda *args, **kwargs: None)


def test_detects_english_text():
    text = (
        "Artificial intelligence helps teachers provide faster feedback "
        "and personalise classroom learning."
    )
    assert detect_language(text) == ("en", True)


def test_detects_hindi_text():
    text = (
        "कृत्रिम बुद्धिमत्ता शिक्षकों को विद्यार्थियों के लिए व्यक्तिगत "
        "शिक्षण सामग्री तैयार करने में सहायता करती है।"
    )
    assert detect_language(text) == ("hi", True)


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
    text = "Bonjour le monde, les enfants jouent dans le jardin."
    lang, confident = detect_language(text)
    assert confident
    assert lang == "fr"


def test_prepare_documents_for_embedding_merges_by_language(monkeypatch):
    """Prepare documents for embedding groups by detected language."""
    docs = {
        "english": "This is an English document.",
        "spanish": "Este es un documento en español.",
        "hindi": "यह एक हिंदी दस्तावेज़ है।",
    }

    monkeypatch.setattr("src.core.cross_lingual.translate_text", lambda text, **kwargs: "English translation")
    result, metadata = prepare_documents_for_embedding({name: [text] for name, text in docs.items()})

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

    result, metadata = prepare_chunks_for_embedding(chunks["doc1"])

    assert result == chunks["doc1"]
    assert len(metadata) == 2
