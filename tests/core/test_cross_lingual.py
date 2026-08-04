from __future__ import annotations


import logging

from src.core.cross_lingual import (detect_language,
                                    prepare_chunks_for_embedding,
                                    prepare_documents_for_embedding,
                                    prepare_text_for_embedding)

from src.core.cross_lingual import (
    detect_language,
    prepare_chunks_for_embedding,
    prepare_documents_for_embedding,
    prepare_text_for_embedding,
    translate_to_english,
)


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


def test_low_confidence_logs_warning(monkeypatch, caplog):
    """A detection confidence below 0.7 should trigger a warning log."""

    class FakeLang:
        lang = "fr"
        prob = 0.5

    monkeypatch.setattr(
        "src.core.cross_lingual.detect_langs", lambda text: [FakeLang()]
    )

    with caplog.at_level(logging.WARNING):
        detect_language(
            "This is a sufficiently long snippet of text for detection to run."
        )

    assert any(
        "Low language detection confidence" in record.message
        for record in caplog.records
    )

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
        translator=lambda text, **_: "Artificial intelligence helps teachers.",
    )

    assert result["original_text"] == original
    assert result["embedding_text"] == ("Artificial intelligence helps teachers.")
    assert result["detected_language"] == "es"
    assert result["translated"] is True
    assert result["translation_failed"] is False


def test_translation_failure_falls_back_to_original():
    original = "L'intelligence artificielle aide les enseignants."

    def broken_translator(*args, **kwargs):
        raise RuntimeError("network unavailable")

    result = prepare_text_for_embedding(
        original,
        detector=lambda _: "fr",
        translator=broken_translator,
    )

    assert result["original_text"] == original
    assert result["embedding_text"] == original
    assert result["translated"] is False
    assert result["translation_failed"] is True


def test_short_or_empty_text_is_safe():
    assert detect_language("") == ("en", False)
    assert detect_language("12345") == ("en", False)

    result = prepare_text_for_embedding("")
    assert result["embedding_text"] == ""
    assert result["translated"] is False


def test_detect_language_low_confidence(caplog):
    """Verify that low-confidence detections return 'en', is_confident=False and log warnings."""
    from unittest.mock import patch
    from langdetect.language import Language
    import logging

    with patch("src.core.cross_lingual.detect_langs") as mock_detect_langs:
        mock_detect_langs.return_value = [Language("fr", 0.5)]

        with caplog.at_level(logging.WARNING):
            lang, confident = detect_language("some text in french but low confidence")

        assert lang == "en"
        assert confident is False
        assert any("Low-confidence language detection" in record.message for record in caplog.records)


def test_detect_language_high_confidence():
    """Verify that high-confidence detections return the correct language and is_confident=True."""
    from unittest.mock import patch
    from langdetect.language import Language

    with patch("src.core.cross_lingual.detect_langs") as mock_detect_langs:
        mock_detect_langs.return_value = [Language("fr", 0.9)]
        lang, confident = detect_language("some text in the french language")

        assert lang == "fr"
        assert confident is True


def test_chunk_preparation_preserves_original_order():
    chunks = ["English paragraph", "Texto en español"]
    translations = iter(["English paragraph", "Text in Spanish"])

    # Exercise the public chunk helper by monkeypatching through the module.
    import src.core.cross_lingual as module

    original_prepare = module.prepare_text_for_embedding
    try:

        def fake_prepare(text):
            translated = next(translations)
            return {
                "original_text": text,
                "embedding_text": translated,
                "detected_language": "en" if text.startswith("English") else "es",
                "translated": text != translated,
                "translation_failed": False,
            }

        module.prepare_text_for_embedding = fake_prepare
        embedding_chunks, metadata = prepare_chunks_for_embedding(chunks)
    finally:
        module.prepare_text_for_embedding = original_prepare

    assert embedding_chunks == ["English paragraph", "Text in Spanish"]
    assert [item["original_text"] for item in metadata] == chunks


def test_document_preparation_does_not_mutate_source_chunks(monkeypatch):
    source = {
        "english.pdf": ["AI supports education."],
        "spanish.pdf": ["La IA apoya la educación."],
    }

    def fake_prepare(text):
        if text.startswith("La "):
            return {
                "original_text": text,
                "embedding_text": "AI supports education.",
                "detected_language": "es",
                "translated": True,
                "translation_failed": False,
            }
        return {
            "original_text": text,
            "embedding_text": text,
            "detected_language": "en",
            "translated": False,
            "translation_failed": False,
        }

    monkeypatch.setattr(
        "src.core.cross_lingual.prepare_text_for_embedding",
        fake_prepare,
    )

    aligned, metadata = prepare_documents_for_embedding(source)

    assert source["spanish.pdf"][0] == "La IA apoya la educación."
    assert aligned["spanish.pdf"][0] == "AI supports education."
    assert metadata["spanish.pdf"][0]["translated"] is True


def test_translate_to_english_confidence():
    """Verify translate_to_english returns dictionary with confidence metrics."""
    # 1. English text -> confidence should be 1.0
    english_input = "Artificial intelligence supports modern education."
    res_en = translate_to_english(english_input)
    assert res_en["confidence"] == 1.0
    assert res_en["source_language"] == "en"
    assert res_en["translated_text"] == english_input

    # 2. Non-English text with mock translator
    spanish_input = "La inteligencia artificial ayuda a los profesores en el aula."

    def mock_translator(text, target_lang="en", source_lang="es"):
        return "Artificial intelligence helps teachers in the classroom."

    res_es = translate_to_english(spanish_input, translator=mock_translator)
    assert res_es["source_language"] == "es"
    assert isinstance(res_es["confidence"], float)
    assert 0.0 <= res_es["confidence"] <= 1.0
    assert res_es["translated_text"] == "Artificial intelligence helps teachers in the classroom."
