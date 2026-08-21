from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pytest

from src.core.cross_lingual import (
    TranslationMemoryCache,
    back_translate_chunk,
    detect_chunk_language,
    detect_language,
    prepare_chunks_for_embedding,
    prepare_documents_for_embedding,
    prepare_text_for_embedding,
    verify_semantic_fidelity,
)
from src.db.translation_cache import clear_translation_cache, init_translation_cache

# ── Issue #1956 Cache Fixture ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_cache():
    """Initialize and clear the translation cache before/after each test."""
    init_translation_cache()
    clear_translation_cache()
    yield
    clear_translation_cache()


# ── Original Language Detection Tests ─────────────────────────────────────────


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
    import logging
    from unittest.mock import patch

    from langdetect.language import Language

    with patch("src.core.cross_lingual.detect_langs") as mock_detect_langs:
        mock_detect_langs.return_value = [Language("fr", 0.5)]

        with caplog.at_level(logging.WARNING):
            lang, confident = detect_language("some text in french but low confidence")

        assert lang == "en"
        assert confident is False
        assert any(
            "Low-confidence language detection" in record.message
            for record in caplog.records
        )


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


# ── Issue #1956: Lightweight Language Detection Tests ─────────────────────────


class TestDetectChunkLanguage:
    """Tests for lightweight language detection heuristics."""

    def test_detect_spanish(self):
        text = "El rápido zorro marrón salta sobre el perro perezoso."
        assert detect_chunk_language(text) == "es"

    def test_detect_french(self):
        text = "Le renard brun rapide saute par-dessus le chien paresseux."
        assert detect_chunk_language(text) == "fr"

    def test_detect_german(self):
        text = "Der schnelle braune Fuchs springt über den faulen Hund."
        assert detect_chunk_language(text) == "de"

    def test_detect_chinese(self):
        text = "快速的棕色狐狸跳过懒狗。"
        assert detect_chunk_language(text) == "zh"

    def test_detect_english_default(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert detect_chunk_language(text) == "en"

    def test_english_with_articles_not_spanish(self):
        """English text with common articles must stay 'en', not flip to 'es'."""
        text = "a plan and a goal for the team next year"
        assert detect_chunk_language(text) == "en"

    def test_empty_text_returns_english(self):
        assert detect_chunk_language("") == "en"
        assert detect_chunk_language(None) == "en"


# ── Issue #1956: Back-Translation & Cache Tests ───────────────────────────────


class TestBackTranslateChunk:
    """Tests for the back-translation and caching logic."""

    def test_english_text_unchanged(self):
        text = "This is already in English."
        result = back_translate_chunk(text, source_lang="en")
        assert result == text

    @patch("src.core.cross_lingual.save_translation")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    def test_cache_miss_triggers_translation(self, mock_get, mock_save):
        text = "El zorro marrón."
        result = back_translate_chunk(text, source_lang="es", use_cache=True)

        # Should call save_translation after mock translation
        mock_save.assert_called_once()
        assert "[Translated from es]" in result

    @patch("src.core.cross_lingual.save_translation")
    @patch(
        "src.core.cross_lingual.get_cached_translation", return_value="The brown fox."
    )
    def test_cache_hit_returns_cached_value(self, mock_get, mock_save):
        text = "El zorro marrón."
        result = back_translate_chunk(text, source_lang="es", use_cache=True)

        assert result == "The brown fox."
        mock_save.assert_not_called()

    def test_cache_disabled_skips_lookup(self):
        text = "El zorro marrón."
        # With use_cache=False, it should bypass cache and translate directly
        result = back_translate_chunk(text, source_lang="es", use_cache=False)
        assert "[Translated from es]" in result


# ── Issue #2222: Add Italian and Portuguese language detection heuristics ─────


class TestItalianPortugueseLanguageDetection:
    """Test suite for Italian and Portuguese language detection (Issue #2222)."""

    def test_detects_italian_text(self):
        """Verify Italian text is correctly identified."""
        italian_text = "Il gatto è sul tavolo e la donna legge un libro in biblioteca"

        result = detect_chunk_language(italian_text)

        assert result == "it"

    def test_detects_italian_with_mixed_case(self):
        """Verify Italian detection works with mixed case."""
        italian_text = "IL GATTO È SUL TAVOLO E LA DONNA LEGGE UN LIBRO"

        result = detect_chunk_language(italian_text)

        assert result == "it"

    def test_detects_italian_academic_text(self):
        """Verify Italian detection works with academic text."""
        italian_text = (
            "La ricerca dimostra che il metodo sperimentale è fondamentale "
            "per la validazione delle ipotesi scientifiche in chimica"
        )

        result = detect_chunk_language(italian_text)

        assert result == "it"

    def test_detects_portuguese_text(self):
        """Verify Portuguese text is correctly identified."""
        portuguese_text = "O gato está na mesa e a mulher lê um livro na biblioteca"

        result = detect_chunk_language(portuguese_text)

        assert result == "pt"

    def test_detects_portuguese_with_mixed_case(self):
        """Verify Portuguese detection works with mixed case."""
        portuguese_text = "O GATO ESTÁ NA MESA E A MULHER LÊ UM LIVRO"

        result = detect_chunk_language(portuguese_text)

        assert result == "pt"

    def test_detects_portuguese_academic_text(self):
        """Verify Portuguese detection works with academic text."""
        portuguese_text = (
            "A pesquisa demonstra que o método experimental é fundamental "
            "para a validação das hipóteses científicas em química"
        )

        result = detect_chunk_language(portuguese_text)

        assert result == "pt"

    def test_italian_vs_spanish_distinction(self):
        """Verify Italian and Spanish are distinguished correctly."""
        italian = "Il gatto è sul tavolo e la donna legge"
        spanish = "El gato está en la mesa y la mujer lee"

        assert detect_chunk_language(italian) == "it"
        assert detect_chunk_language(spanish) == "es"

    def test_portuguese_vs_spanish_distinction(self):
        """Verify Portuguese and Spanish are distinguished correctly."""
        portuguese = "O gato está na mesa e a mulher lê"
        spanish = "El gato está en la mesa y la mujer lee"

        assert detect_chunk_language(portuguese) == "pt"
        assert detect_chunk_language(spanish) == "es"

    def test_italian_low_stop_word_density_returns_english(self):
        """Verify Italian text with low stop word density defaults to English."""
        # Text with very few Italian stop words
        low_density_text = "cat dog house car tree"

        result = detect_chunk_language(low_density_text)

        # Should default to English when stop word density < 10%
        assert result == "en"

    def test_portuguese_low_stop_word_density_returns_english(self):
        """Verify Portuguese text with low stop word density defaults to English."""
        # Text with very few Portuguese stop words
        low_density_text = "computer programming algorithm data structure"

        result = detect_chunk_language(low_density_text)

        # Should default to English when stop word density < 10%
        assert result == "en"

    def test_italian_in_heuristics_dict(self):
        """Verify Italian pattern exists in _LANGUAGE_HEURISTICS."""
        from src.core.cross_lingual import _LANGUAGE_HEURISTICS

        assert "it" in _LANGUAGE_HEURISTICS
        assert isinstance(_LANGUAGE_HEURISTICS["it"], type(re.compile("")))

    def test_portuguese_in_heuristics_dict(self):
        """Verify Portuguese pattern exists in _LANGUAGE_HEURISTICS."""
        from src.core.cross_lingual import _LANGUAGE_HEURISTICS

        assert "pt" in _LANGUAGE_HEURISTICS
        assert isinstance(_LANGUAGE_HEURISTICS["pt"], type(re.compile("")))

    def test_italian_pattern_matches_stop_words(self):
        """Verify Italian regex pattern matches common stop words."""
        from src.core.cross_lingual import _LANGUAGE_HEURISTICS

        pattern = _LANGUAGE_HEURISTICS["it"]

        # Test common Italian stop words
        assert pattern.search("il gatto")
        assert pattern.search("la donna")
        assert pattern.search("di Roma")
        assert pattern.search("e poi")
        assert pattern.search("che cosa")

    def test_portuguese_pattern_matches_stop_words(self):
        """Verify Portuguese regex pattern matches common stop words."""
        from src.core.cross_lingual import _LANGUAGE_HEURISTICS

        pattern = _LANGUAGE_HEURISTICS["pt"]

        # Test common Portuguese stop words
        assert pattern.search("o gato")
        assert pattern.search("a mulher")
        assert pattern.search("de Lisboa")
        assert pattern.search("em casa")
        assert pattern.search("que fazer")

    def test_short_text_returns_english(self):
        """Verify very short text (< 3 words) defaults to English."""
        short_italian = "Il gatto"
        short_portuguese = "O gato"

        # Should default to English for very short text
        assert detect_chunk_language(short_italian) == "en"
        assert detect_chunk_language(short_portuguese) == "en"

    def test_empty_text_returns_english(self):
        """Verify empty text defaults to English."""
        assert detect_chunk_language("") == "en"
        assert detect_chunk_language(None) == "en"

    @pytest.mark.parametrize(
        "text,expected_lang",
        [
            ("Il professore spiega la teoria", "it"),
            ("La studentessa studia per l'esame", "it"),
            ("O professor explica a teoria", "pt"),
            ("A estudante estuda para o exame", "pt"),
            ("El profesor explica la teoría", "es"),
            ("Le professeur explique la théorie", "fr"),
            ("Der Professor erklärt die Theorie", "de"),
        ],
    )
    def test_academic_phrases_parametrized(self, text, expected_lang):
        """Verify academic phrases in multiple languages are detected correctly."""
        # Add more context to meet minimum word count
        full_text = f"{text} nella università oggi"

        result = detect_chunk_language(full_text)

        assert result == expected_lang


# ── Issue #1956 & #2249: Semantic Fidelity Verification Tests ─────────────────


class TestVerifySemanticFidelity:
    """Tests for embedding similarity verification (Issue #2249)."""

    def test_fidelity_none_embedding(self):
        """Test that None embeddings return 0.0."""
        assert verify_semantic_fidelity(None, np.array([1.0, 0.0])) == 0.0
        assert verify_semantic_fidelity(np.array([1.0, 0.0]), None) == 0.0
        assert verify_semantic_fidelity(None, None) == 0.0

    def test_fidelity_empty_embedding(self):
        """Test that empty arrays return 0.0."""
        assert verify_semantic_fidelity(np.array([]), np.array([1.0, 0.0])) == 0.0
        assert verify_semantic_fidelity(np.array([1.0, 0.0]), np.array([])) == 0.0
        assert verify_semantic_fidelity(np.array([]), np.array([])) == 0.0

    def test_fidelity_zero_norm(self):
        """Test that zero-norm (all-zero) vectors return 0.0."""
        zero_vec = np.array([0.0, 0.0])
        valid_vec = np.array([1.0, 2.0])
        assert verify_semantic_fidelity(zero_vec, valid_vec) == 0.0
        assert verify_semantic_fidelity(valid_vec, zero_vec) == 0.0
        assert verify_semantic_fidelity(zero_vec, zero_vec) == 0.0

    def test_fidelity_identical_vectors(self):
        """Test that identical vectors return 1.0 (cosine similarity = 1)."""
        vec = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(verify_semantic_fidelity(vec, vec), rel=1e-5) == 1.0

    def test_fidelity_orthogonal_vectors(self):
        """Test that orthogonal vectors return 0.0 (cosine similarity = 0)."""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        assert pytest.approx(verify_semantic_fidelity(vec_a, vec_b), rel=1e-5) == 0.0


class TestBackTranslateChunkRealTranslation:
    """Test suite for real translation call implementation (Issue #2219)."""

    @patch("src.core.cross_lingual.translate_text")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="es")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_calls_translate_text_with_correct_args(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify translate_text is called with correct arguments."""
        mock_translate.return_value = "Hello world"

        result = back_translate_chunk("Hola mundo", source_lang="es")

        mock_translate.assert_called_once_with(
            "Hola mundo",
            target_lang="en",
            source_lang="es",
        )
        assert result == "Hello world"

    @patch("src.core.cross_lingual.translate_text")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="fr")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_saves_translation_to_cache(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify successful translation is saved to cache."""
        mock_translate.return_value = "Good morning"

        result = back_translate_chunk("Bonjour", source_lang="fr", use_cache=True)

        mock_save.assert_called_once_with(
            "Bonjour",
            "fr",
            "en",
            "Good morning",
        )
        assert result == "Good morning"

    @patch("src.core.cross_lingual.translate_text", side_effect=Exception("API Error"))
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="de")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_fallback_on_translation_failure(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify fallback to original text when translation fails."""
        original_text = "Guten Tag"

        result = back_translate_chunk(original_text, source_lang="de")

        # Should return original text, not raise exception
        assert result == original_text
        # Should not save failed translation to cache
        mock_save.assert_not_called()

    @patch("src.core.cross_lingual.translate_text", return_value="")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="it")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_fallback_on_empty_translation(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify fallback to original text when translation returns empty string."""
        original_text = "Buongiorno"

        result = back_translate_chunk(original_text, source_lang="it")

        assert result == original_text
        mock_save.assert_not_called()

    @patch("src.core.cross_lingual.translate_text", return_value=None)
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="pt")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_fallback_on_none_translation(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify fallback to original text when translation returns None."""
        original_text = "Bom dia"

        result = back_translate_chunk(original_text, source_lang="pt")

        assert result == original_text
        mock_save.assert_not_called()

    @patch("src.core.cross_lingual.translate_text", return_value="  Hello  ")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="es")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_strips_whitespace_from_translation(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify whitespace is stripped from translation result."""
        result = back_translate_chunk("Hola", source_lang="es")

        assert result == "Hello"
        # Cache should store stripped version
        mock_save.assert_called_once_with("Hola", "es", "en", "Hello")

    @patch("src.core.cross_lingual.translate_text")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="es")
    @patch("src.core.cross_lingual.get_cached_translation", return_value=None)
    @patch("src.core.cross_lingual.save_translation")
    def test_skips_cache_when_disabled(
        self, mock_save, mock_cache, mock_detect, mock_translate
    ):
        """Verify cache is skipped when use_cache=False."""
        mock_translate.return_value = "Hello"

        result = back_translate_chunk("Hola", source_lang="es", use_cache=False)

        # Should not check cache
        mock_cache.assert_not_called()
        # Should not save to cache
        mock_save.assert_not_called()
        assert result == "Hello"

    @patch("src.core.cross_lingual.translate_text")
    @patch("src.core.cross_lingual.detect_chunk_language", return_value="en")
    def test_skips_translation_for_target_language(self, mock_detect, mock_translate):
        """Verify no translation occurs when source is already target language."""
        text = "Hello world"

        result = back_translate_chunk(text, source_lang="en")

        # Should not call translate_text
        mock_translate.assert_not_called()
        # Should return original text unchanged
        assert result == text

    def test_handles_empty_input(self):
        """Verify empty input returns empty string."""
        assert back_translate_chunk("") == ""
        assert back_translate_chunk(None) == ""

    def test_handles_non_string_input(self):
        """Verify non-string input returns empty string."""
        assert back_translate_chunk(123) == ""
        assert back_translate_chunk([]) == ""

    @patch("src.core.cross_lingual.translate_text", return_value="Hello world")
    @patch(
        "src.core.cross_lingual.get_cached_translation",
        return_value="Cached translation",
    )
    def test_prefers_cache_over_translation(self, mock_cache, mock_translate):
        """Verify cached translation is preferred over new translation."""
        result = back_translate_chunk("Hola mundo", source_lang="es", use_cache=True)

        # Should return cached value
        assert result == "Cached translation"
        # Should not call translate_text
        mock_translate.assert_not_called()
