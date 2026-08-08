from __future__ import annotations

from src.core.cross_lingual import (TranslationMemoryCache,
                                    detect_language,
                                    prepare_chunks_for_embedding,
                                    prepare_documents_for_embedding,
                                    prepare_text_for_embedding)


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
        lang, confident = detect_language("some text in french")
        
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



def test_translation_memory_cache_hits_for_identical_sentence():
    cache = TranslationMemoryCache()
    calls = []
    sentence = "La inteligencia artificial ayuda a los profesores."

    def fake_translator(text, **kwargs):
        calls.append((text, kwargs))
        return "Artificial intelligence helps teachers."

    first = prepare_text_for_embedding(
        sentence,
        detector=lambda _: "es",
        translator=fake_translator,
        translation_cache=cache,
    )
    second = prepare_text_for_embedding(
        sentence,
        detector=lambda _: "es",
        translator=fake_translator,
        translation_cache=cache,
    )

    assert first["embedding_text"] == second["embedding_text"]
    assert first["translated"] is True
    assert second["translated"] is True
    assert len(calls) == 1
    assert len(cache) == 1


def test_translation_cache_keys_include_language_pair():
    cache = TranslationMemoryCache()
    calls = []

    def fake_translator(text, **kwargs):
        calls.append(kwargs["source_lang"])
        return f"translated from {kwargs['source_lang']}"

    spanish = prepare_text_for_embedding(
        "shared sentence",
        detector=lambda _: "es",
        translator=fake_translator,
        translation_cache=cache,
    )
    french = prepare_text_for_embedding(
        "shared sentence",
        detector=lambda _: "fr",
        translator=fake_translator,
        translation_cache=cache,
    )

    assert spanish["embedding_text"] == "translated from es"
    assert french["embedding_text"] == "translated from fr"
    assert calls == ["es", "fr"]
    assert len(cache) == 2


def test_failed_translation_is_not_cached():
    cache = TranslationMemoryCache()
    calls = []

    def failing_translator(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("translation unavailable")

    for _ in range(2):
        result = prepare_text_for_embedding(
            "Texte français répétitif.",
            detector=lambda _: "fr",
            translator=failing_translator,
            translation_cache=cache,
        )
        assert result["translation_failed"] is True

    assert len(calls) == 2
    assert len(cache) == 0


def test_translation_cache_clear_removes_entries():
    cache = TranslationMemoryCache()
    cache.set(
        "Hola",
        "Hello",
        source_lang="es",
        target_lang="en",
    )

    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_english_text_does_not_enter_translation_cache():
    cache = TranslationMemoryCache()

    result = prepare_text_for_embedding(
        "Artificial intelligence supports education.",
        detector=lambda _: "en",
        translator=lambda *_args, **_kwargs: "unused",
        translation_cache=cache,
    )

    assert result["translated"] is False
    assert len(cache) == 0
