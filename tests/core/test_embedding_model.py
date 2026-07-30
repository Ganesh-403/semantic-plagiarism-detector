from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.embedding_model as embedding_model
from src.core.embedding_model import (embed_chunks, embed_documents,
                                      get_document_embedding)


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


def test_embed_chunks_zero_length_text(mock_model):
    """Test embedding when input chunks contain zero-length / empty string."""
    chunks = [""]
    result = embed_chunks(chunks)
    assert result.shape == (1, 384)


def test_embed_chunks_mixed_zero_length(mock_model):
    """Test embedding when zero-length text is mixed with valid text chunks."""
    chunks = ["Valid sentence.", "", "   "]
    result = embed_chunks(chunks)
    assert result.shape == (3, 384)


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
