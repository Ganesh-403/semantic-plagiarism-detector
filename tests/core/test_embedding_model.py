from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.embedding_model as embedding_model
from src.core.embedding_model import (EmbeddingModelManager, embed_chunks,
                                      embed_documents, get_document_embedding)


def _mock_encode(
    texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
):
    return np.random.rand(len(texts), 384).astype("float32")


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode.side_effect = _mock_encode
    with patch("src.core.embedding_model._get_model", return_value=model):
        yield model


def test_embed_chunks_shape(mock_model):
    chunks = ["Hello world.", "Another sentence here for testing purposes."]
    result = embed_chunks(chunks)
    assert result.shape == (2, 384)


def test_embed_chunks_empty():
    result = embed_chunks([])
    assert result.size == 0

def test_embed_empty_text(mock_model):
    """Test embedding a single empty string."""

    result = embedding_model.embed_chunks([""])

    assert result.shape == (1, 384)
    assert result.dtype == np.float32


def test_embed_chunks_returns_float32(mock_model):
    mock_model.encode.side_effect = lambda texts, **kw: np.ones(
        (len(texts), 384), dtype="float32"
    )
    result = embed_chunks(["test chunk"])
    assert result.dtype == np.float32


def test_embed_documents_keys(mock_model):
    docs = {"doc1": ["chunk one", "chunk two"], "doc2": ["another chunk"]}
    result = embed_documents(docs)
    assert set(result.keys()) == {"doc1", "doc2"}
    assert result["doc1"].shape == (2, 384)
    assert result["doc2"].shape == (1, 384)


def test_embed_documents_empty_doc(mock_model, capsys):
    docs = {"empty_doc": []}
    result = embed_documents(docs)
    assert result["empty_doc"].size == 0


def test_get_document_embedding_mean_pool():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = get_document_embedding(emb)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [0.5, 0.5])


def test_get_document_embedding_single_vector():
    vec = np.array([0.1, 0.2, 0.3])
    result = get_document_embedding(vec)
    np.testing.assert_array_equal(result, vec)


def test_get_model_uses_multilingual_default(monkeypatch):
    model = MagicMock()
    sentence_transformer = MagicMock(return_value=model)
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(embedding_model, "SentenceTransformer", sentence_transformer)

    result = embedding_model._get_model()

    assert result is model
    sentence_transformer.assert_called_once_with(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


def test_get_model_uses_environment_override(monkeypatch):
    model = MagicMock()
    sentence_transformer = MagicMock(return_value=model)
    monkeypatch.setenv(
        "SEMANTIC_PLAGIARISM_MODEL", "distiluse-base-multilingual-cased-v2"
    )
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(embedding_model, "SentenceTransformer", sentence_transformer)

    result = embedding_model._get_model()

    assert result is model
    sentence_transformer.assert_called_once_with("distiluse-base-multilingual-cased-v2")


def test_embedding_model_manager_fallback(caplog, monkeypatch):
    """Test that EmbeddingModelManager falls back to lightweight model and logs a warning on failure."""
    import logging

    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    primary = embedding_model._get_model_name()
    fallback = "all-MiniLM-L6-v2"

    def mock_sentence_transformer(model_name):
        if model_name == primary:
            raise RuntimeError("Failed to load primary model")
        return MagicMock()

    monkeypatch.setattr(embedding_model, "SentenceTransformer", mock_sentence_transformer)

    with caplog.at_level(logging.WARNING):
        manager = EmbeddingModelManager.get_instance()
        model = manager.get_model()

        assert model is not None
        # Verify the warning was logged
        assert any(
            f"Primary embedding model {primary} unavailable. Falling back to {fallback}" in record.message
            for record in caplog.records
        )


def test_embedding_model_device_logging(caplog, monkeypatch):
    """Test that EmbeddingModelManager logs device selection when initializing the model."""
    import logging

    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    mock_model_obj = MagicMock()
    mock_model_obj.device = "cpu"

    monkeypatch.setattr(
        embedding_model, "SentenceTransformer", MagicMock(return_value=mock_model_obj)
    )

    with caplog.at_level(logging.INFO):
        manager = EmbeddingModelManager.get_instance()
        model = manager.get_model()

        assert model is mock_model_obj
        expected_log = "Initializing SentenceTransformer model [paraphrase-multilingual-MiniLM-L12-v2] on device [cpu]"
        assert any(
            expected_log in record.message for record in caplog.records
        ), f"Expected device log message not found in: {[r.message for r in caplog.records]}"


def test_detect_device_helper():
    """Test _detect_device helper logic for string device, typed device, and fallback."""
    mock_obj = MagicMock()
    mock_obj.device = "cuda"
    assert embedding_model._detect_device(mock_obj) == "cuda"

    mock_obj_device_type = MagicMock()
    mock_dev = MagicMock()
    mock_dev.type = "mps"
    mock_obj_device_type.device = mock_dev
    assert embedding_model._detect_device(mock_obj_device_type) == "mps"

    assert embedding_model._detect_device(None) in ("cpu", "cuda", "mps")
