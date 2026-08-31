"""
Tests for Cross-Lingual Plagiarism Detection Engine.

Comprehensive test suite covering language detection, embedding,
cross-language comparison, and result generation.
"""

import numpy as np

from src.core.cross_lingual_detector import (
    LANGUAGE_NAMES,
    CrossLingualConfig,
    CrossLingualDetector,
    CrossLingualResult,
    LanguageMatch,
)


class TestLanguageDetection:
    """Tests for language detection."""

    def setup_method(self):
        self.detector = CrossLingualDetector()

    def test_detect_english(self):
        """Test English detection."""
        text = "The quick brown fox jumps over the lazy dog. This is a sample English text."
        lang = self.detector.detect_language(text)
        assert lang == "en"

    def test_detect_spanish(self):
        """Test Spanish detection."""
        text = "El gato se sentó en la mesa. La comida está deliciosa hoy."
        lang = self.detector.detect_language(text)
        assert lang == "es"

    def test_detect_french(self):
        """Test French detection."""
        text = "Le chat est sur la table. Les enfants jouent dans le jardin."
        lang = self.detector.detect_language(text)
        assert lang == "fr"

    def test_detect_german(self):
        """Test German detection."""
        text = "Der Hund ist groß und braun. Das Haus hat eine Tür und ein Fenster."
        lang = self.detector.detect_language(text)
        assert lang == "de"

    def test_empty_text(self):
        """Test empty text defaults to English."""
        lang = self.detector.detect_language("")
        assert lang == "en"


class TestEmbedding:
    """Tests for text embedding."""

    def setup_method(self):
        self.config = CrossLingualConfig(enable_cache=True)
        self.detector = CrossLingualDetector(self.config)

    def test_embed_text_returns_vector(self):
        """Test embedding returns numpy array."""
        embedding = self.detector.embed_text("Hello world", "en")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0

    def test_embed_caching(self):
        """Test embedding caching works."""
        emb1 = self.detector.embed_text("Test text", "en")
        emb2 = self.detector.embed_text("Test text", "en")
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        """Test different texts produce different embeddings."""
        emb1 = self.detector.embed_text("Hello world")
        emb2 = self.detector.embed_text("Goodbye universe")
        assert not np.array_equal(emb1, emb2)

    def test_clear_cache(self):
        """Test cache clearing."""
        self.detector.embed_text("Test", "en")
        assert len(self.detector._cache) > 0
        self.detector.clear_cache()
        assert len(self.detector._cache) == 0


class TestSimilarityComputation:
    """Tests for similarity computation."""

    def setup_method(self):
        self.detector = CrossLingualDetector()

    def test_identical_embeddings(self):
        """Test identical embeddings produce similarity 1.0."""
        emb = np.random.rand(10).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        matrix = self.detector.compare_across_languages([emb], [emb])
        assert abs(matrix[0, 0] - 1.0) < 1e-6

    def test_orthogonal_embeddings(self):
        """Test orthogonal embeddings produce similarity ~0."""
        emb1 = np.array([1, 0, 0, 0], dtype=np.float32)
        emb2 = np.array([0, 1, 0, 0], dtype=np.float32)
        matrix = self.detector.compare_across_languages([emb1], [emb2])
        assert abs(matrix[0, 0]) < 1e-6

    def test_empty_embeddings(self):
        """Test empty input handling."""
        matrix = self.detector.compare_across_languages([], [])
        assert len(matrix) == 0


class TestCrossLingualDetection:
    """Tests for full cross-lingual detection pipeline."""

    def setup_method(self):
        self.config = CrossLingualConfig(enable_cache=False, similarity_threshold=0.5)
        self.detector = CrossLingualDetector(self.config)

    def test_basic_detection(self):
        """Test basic detection flow."""
        documents = {
            "doc_en": ("en", ["This is an English document about machine learning."]),
            "doc_fr": (
                "fr",
                ["Ceci est un document français sur l'apprentissage automatique."],
            ),
        }
        result = self.detector.detect_cross_lingual_plagiarism(documents)
        assert isinstance(result, CrossLingualResult)
        assert result.summary["total_documents"] == 2
        assert result.summary["languages_detected"] == 2

    def test_single_document(self):
        """Test single document handling."""
        documents = {"doc1": ("en", ["Some text here."])}
        result = self.detector.detect_cross_lingual_plagiarism(documents)
        assert result.total_comparisons == 0

    def test_same_language_pair(self):
        """Test same-language comparison."""
        documents = {
            "doc_a": (
                "en",
                ["Machine learning is a subset of artificial intelligence."],
            ),
            "doc_b": ("en", ["Deep learning is a subset of machine learning."]),
        }
        result = self.detector.detect_cross_lingual_plagiarism(documents)
        assert result.summary["total_documents"] == 2


class TestResultSerialization:
    """Tests for result serialization."""

    def test_result_to_dict(self):
        """Test result to_dict."""
        result = CrossLingualResult(
            documents=[],
            matches=[],
            language_distribution={"en": 2},
            total_comparisons=1,
            processing_time=0.5,
            summary={"total_documents": 2},
        )
        d = result.to_dict()
        assert d["summary"]["total_documents"] == 2
        assert d["language_distribution"]["en"] == 2

    def test_match_to_dict(self):
        """Test match to_dict."""
        match = LanguageMatch(
            source_doc="a.pdf",
            source_lang="en",
            source_chunk="Hello",
            target_doc="b.pdf",
            target_lang="es",
            target_chunk="Hola",
            similarity=0.85,
            method="cosine",
            translation_used=True,
            confidence=0.9,
        )
        d = match.to_dict()
        assert d["source_doc"] == "a.pdf"
        assert d["similarity"] == 0.85


class TestConfiguration:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = CrossLingualConfig()
        assert "en" in config.enabled_languages
        assert config.similarity_threshold == 0.65
        assert config.use_translation_bridge is True

    def test_custom_config(self):
        """Test custom config."""
        config = CrossLingualConfig(
            enabled_languages=["en", "es"], similarity_threshold=0.8
        )
        assert len(config.enabled_languages) == 2
        assert config.similarity_threshold == 0.8


class TestSupportedLanguages:
    """Tests for supported languages."""

    def test_supported_languages_list(self):
        """Test supported languages listing."""
        detector = CrossLingualDetector()
        languages = detector.get_supported_languages()
        assert len(languages) >= 10
        assert any(l["code"] == "en" for l in languages)
        assert any(l["code"] == "es" for l in languages)

    def test_language_names(self):
        """Test language names mapping."""
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["es"] == "Spanish"
        assert LANGUAGE_NAMES["fr"] == "French"
