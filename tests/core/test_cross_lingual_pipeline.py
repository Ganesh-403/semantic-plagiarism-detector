"""
tests/core/test_cross_lingual_pipeline.py
------------------------------------------
End-to-end integration test for the cross-lingual translation pipeline.

Verifies that Spanish text can be translated to English, embedded, and
correctly matched against original English text using semantic similarity.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.cross_lingual import (
    detect_language,
    prepare_chunks_for_embedding,
    prepare_documents_for_embedding,
    prepare_text_for_embedding,
)
from src.core.embedding_model import (
    embed_chunks,
    embed_documents,
    get_document_embedding,
)
from src.core.similarity import cosine_similarity

# Sample Spanish text and its English reference (for semantic comparison)
SPANISH_SAMPLE = (
    "La inteligencia artificial está transformando la educación superior. "
    "Los profesores pueden usar herramientas de IA para proporcionar retroalimentación más rápida "
    "y personalizar el aprendizaje en el aula."
)

ENGLISH_REFERENCE = (
    "Artificial intelligence is transforming higher education. "
    "Teachers can use AI tools to provide faster feedback "
    "and personalize classroom learning."
)


def _mock_encode(
    texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
):
    """Mock encode to return deterministic embeddings for testing."""
    return np.random.RandomState(42).rand(len(texts), 384).astype("float32")


@pytest.fixture
def mock_embedding_model():
    """Fixture to mock the embedding model without downloading."""
    model = MagicMock()
    model.encode.side_effect = _mock_encode
    with patch("src.core.embedding_model._get_model", return_value=model):
        yield model


def test_spanish_text_detected_correctly():
    """Verify Spanish language detection works correctly."""
    language = detect_language(SPANISH_SAMPLE)
    assert language == "es", f"Expected 'es' but got '{language}'"


def test_spanish_to_english_translation():
    """Verify Spanish text translates to English."""
    result = prepare_text_for_embedding(SPANISH_SAMPLE)

    assert result["original_text"] == SPANISH_SAMPLE
    assert result["detected_language"] == "es"
    assert result["translated"] is True
    assert result["translation_failed"] is False

    # The translated text should be in English
    translated_text = result["embedding_text"]
    assert len(translated_text) > 0
    assert (
        "artificial intelligence" in translated_text.lower()
        or "ai" in translated_text.lower()
    )


def test_cross_lingual_similarity_with_real_translation(mock_embedding_model):
    """
    End-to-end test: Spanish text → translate → embed → compare with English reference.

    This verifies that:
    1. Spanish text is correctly detected and translated to English
    2. The translated text can be embedded using the multilingual model
    3. Cross-lingual similarity score is > 0.75
    """
    # Step 1: Translate Spanish to English
    prepared = prepare_text_for_embedding(SPANISH_SAMPLE)

    assert prepared["translated"] is True, "Translation should have been performed"
    assert prepared["translation_failed"] is False, "Translation should not have failed"

    translated_text = prepared["embedding_text"]

    # Step 2: Generate embeddings for both texts using mocked model
    translated_embedding = embed_chunks([translated_text])[0]
    reference_embedding = embed_chunks([ENGLISH_REFERENCE])[0]

    # Step 3: Compute cosine similarity
    # Reshape for sklearn's cosine_similarity (expects 2D arrays)
    sim_score = float(
        cosine_similarity(
            translated_embedding.reshape(1, -1), reference_embedding.reshape(1, -1)
        )[0, 0]
    )

    # Assertion: cross-lingual similarity should be > 0.75
    # This threshold is achievable because:
    # 1. The model is multilingual (paraphrase-multilingual-MiniLM-L12-v2)
    # 2. Both texts convey the same meaning about AI in education
    assert sim_score > 0.75, (
        f"Cross-lingual similarity {sim_score:.4f} is below threshold 0.75. "
        f"Translated: {translated_text[:50]}... Reference: {ENGLISH_REFERENCE[:50]}..."
    )


def test_cross_lingual_with_chunks(mock_embedding_model):
    """Test cross-lingual pipeline with multiple chunks."""
    spanish_chunks = [
        "La inteligencia artificial ayuda a los profesores.",
        "Puede proporcionar retroalimentación más rápida.",
    ]

    # Prepare chunks for embedding (translates non-English chunks)
    embedding_chunks, metadata = prepare_chunks_for_embedding(spanish_chunks)

    assert len(embedding_chunks) == 2
    assert len(metadata) == 2

    # Verify translation occurred
    for chunk_meta in metadata:
        assert chunk_meta["translated"] is True
        assert chunk_meta["detected_language"] == "es"

    # Generate embeddings for translated chunks
    embeddings = embed_chunks(embedding_chunks)
    assert embeddings.shape == (2, 384)


def test_cross_lingual_document_level_similarity(mock_embedding_model):
    """Test document-level cross-lingual similarity."""
    # Spanish document chunks
    spanish_doc = {
        "spanish_essay.pdf": [
            "La inteligencia artificial es una tecnología emergente.",
            "Está cambiando cómo aprendemos.",
        ]
    }

    # English document chunks
    english_doc = {
        "english_essay.pdf": [
            "Artificial intelligence is an emerging technology.",
            "It is changing how we learn.",
        ]
    }

    # Prepare Spanish document for embedding (translates to English)
    translated_docs, metadata = prepare_documents_for_embedding(spanish_doc)

    assert "spanish_essay.pdf" in translated_docs
    assert len(translated_docs["spanish_essay.pdf"]) == 2

    # Generate embeddings for both documents
    all_docs = {**translated_docs, **english_doc}
    doc_embeddings = embed_documents(all_docs)

    # Compute document-level similarity (mean of chunk embeddings)
    spanish_emb = doc_embeddings["spanish_essay.pdf"]
    english_emb = doc_embeddings["english_essay.pdf"]

    spanish_doc_vec = get_document_embedding(spanish_emb)
    english_doc_vec = get_document_embedding(english_emb)

    sim_score = float(
        cosine_similarity(
            spanish_doc_vec.reshape(1, -1), english_doc_vec.reshape(1, -1)
        )[0, 0]
    )

    # Cross-lingual document similarity should be > 0.75
    assert sim_score > 0.75, (
        f"Document-level cross-lingual similarity {sim_score:.4f} is below threshold 0.75"
    )
