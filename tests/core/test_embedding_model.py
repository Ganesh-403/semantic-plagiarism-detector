from __future__ import annotations

"""
test_embedding_model.py
-----------------------
Unit tests for the embedding_model module.

Validates:
- Singleton model loading and fallback mechanisms
- Device detection (CPU, CUDA, MPS)
- Mini-batch processing for memory optimization (Issue #920)
- Document and chunk embedding shapes and types
- Edge cases (empty inputs, single chunks, massive batches)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.embedding_model as embedding_model
from src.core.embedding_model import (
    EmbeddingModelManager,
    embed_chunks,
    embed_documents,
    get_document_embedding,
    _detect_device,
)


def _mock_encode(
    texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
):
    """Mock encode function that returns random embeddings of correct shape."""
    return np.random.rand(len(texts), 384).astype("float32")


@pytest.fixture
def mock_model():
    """Fixture to provide a mocked SentenceTransformer model."""
    model = MagicMock()
    model.encode.side_effect = _mock_encode
    with patch("src.core.embedding_model._get_model", return_value=model):
        yield model


# ─── Basic Embedding Tests ─────────────────────────────────────────────────────


def test_embed_chunks_shape(mock_model):
    """Verify output shape matches (N, 384) for N input chunks."""
    chunks = ["Hello world.", "Another sentence here for testing purposes."]
    result = embed_chunks(chunks)
    assert result.shape == (2, 384)


def test_embed_chunks_empty():
    """Empty input list must return an empty array of shape (0, 384)."""
    result = embed_chunks([])
    assert result.size == 0
    assert result.shape == (0, 384)


def test_embed_empty_text(mock_model):
    """Test embedding a single empty string."""
    result = embedding_model.embed_chunks([""])
    assert result.shape == (1, 384)
    assert result.dtype == np.float32


def test_embed_chunks_returns_float32(mock_model):
    """Verify the returned array uses float32 dtype for memory efficiency."""
    mock_model.encode.side_effect = lambda texts, **kw: np.ones(
        (len(texts), 384), dtype="float32"
    )
    result = embed_chunks(["test chunk"])
    assert result.dtype == np.float32


def test_embed_documents_keys(mock_model):
    """Verify embed_documents returns a dict with all original document keys."""
    docs = {"doc1": ["chunk one", "chunk two"], "doc2": ["another chunk"]}
    result = embed_documents(docs)
    assert set(result.keys()) == {"doc1", "doc2"}
    assert result["doc1"].shape == (2, 384)
    assert result["doc2"].shape == (1, 384)


def test_embed_documents_empty_doc(mock_model, capsys):
    """Documents with no chunks must return an empty array, not raise."""
    docs = {"empty_doc": []}
    result = embed_documents(docs)
    assert result["empty_doc"].size == 0
    assert result["empty_doc"].shape == (0, 384)


def test_get_document_embedding_mean_pool():
    """Verify mean pooling across multiple chunk embeddings."""
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = get_document_embedding(emb)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [0.5, 0.5])


def test_get_document_embedding_single_vector():
    """A 1-D array must be returned as-is without mean pooling."""
    vec = np.array([0.1, 0.2, 0.3])
    result = get_document_embedding(vec)
    np.testing.assert_array_equal(result, vec)


# ─── Model Loading & Device Detection Tests ────────────────────────────────────


def test_get_model_uses_multilingual_default(monkeypatch):
    """Verify the default model is the multilingual MiniLM variant."""
    model = MagicMock()
    sentence_transformer = MagicMock(return_value=model)
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(embedding_model, "SentenceTransformer", sentence_transformer)

    result = embedding_model._get_model()

    assert result is model
    sentence_transformer.assert_called_once_with(
        "paraphrase-multilingual-MiniLM-L12-v2", cache_folder=None
    )


def test_get_model_uses_environment_override(monkeypatch):
    """Verify SEMANTIC_PLAGIARISM_MODEL env var overrides the default model."""
    model = MagicMock()
    sentence_transformer = MagicMock(return_value=model)
    monkeypatch.setenv(
        "SEMANTIC_PLAGIARISM_MODEL", "distiluse-base-multilingual-cased-v2"
    )
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(embedding_model, "SentenceTransformer", sentence_transformer)

    result = embedding_model._get_model()

    assert result is model
    sentence_transformer.assert_called_once_with(
        "distiluse-base-multilingual-cased-v2", cache_folder=None
    )


def test_embedding_model_manager_fallback(caplog, monkeypatch):
    """Test that EmbeddingModelManager falls back to lightweight model on failure."""
    import logging

    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    primary = embedding_model._get_model_name()
    fallback = "all-MiniLM-L6-v2"

    def mock_sentence_transformer(model_name, cache_folder=None):
        if model_name == primary:
            raise RuntimeError("Failed to load primary model")
        return MagicMock()

    monkeypatch.setattr(
        embedding_model, "SentenceTransformer", mock_sentence_transformer
    )

    with caplog.at_level(logging.WARNING):
        manager = EmbeddingModelManager.get_instance()
        model = manager.get_model()

    assert model is not None
    assert any(
        f"Primary embedding model {primary} unavailable. Falling back to {fallback}"
        in record.message
        for record in caplog.records
    )


def test_embedding_model_device_logging(caplog, monkeypatch):
    """Test that EmbeddingModelManager logs device selection when initializing."""
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
    assert _detect_device(mock_obj) == "cuda"

    mock_obj_device_type = MagicMock()
    mock_dev = MagicMock()
    mock_dev.type = "mps"
    mock_obj_device_type.device = mock_dev
    assert _detect_device(mock_obj_device_type) == "mps"

    assert _detect_device(None) in ("cpu", "cuda", "mps")


# ─── Mini-Batch Processing Tests (Issue #920) ──────────────────────────────────


def test_embed_chunks_mini_batching_single_batch(mock_model):
    """If chunks <= batch_size, model.encode should be called exactly once."""
    chunks = ["chunk1", "chunk2", "chunk3"]
    embed_chunks(chunks, batch_size=32)
    assert mock_model.encode.call_count == 1


def test_embed_chunks_mini_batching_multiple_batches(mock_model):
    """If chunks > batch_size, model.encode must be called multiple times."""
    chunks = [f"chunk_{i}" for i in range(100)]
    embed_chunks(chunks, batch_size=32)
    # 100 chunks / 32 batch_size = 4 calls (32 + 32 + 32 + 4)
    assert mock_model.encode.call_count == 4


def test_embed_chunks_mini_batching_exact_multiple(mock_model):
    """If chunks is an exact multiple of batch_size, verify call count."""
    chunks = [f"chunk_{i}" for i in range(64)]
    embed_chunks(chunks, batch_size=32)
    assert mock_model.encode.call_count == 2


def test_embed_chunks_mini_batching_batch_size_1(mock_model):
    """batch_size=1 must call model.encode once per chunk."""
    chunks = ["a", "b", "c"]
    embed_chunks(chunks, batch_size=1)
    assert mock_model.encode.call_count == 3


def test_embed_chunks_mini_batching_preserves_order(mock_model):
    """Verify that mini-batching does not scramble the order of embeddings."""

    # Mock encode to return deterministic values based on input text
    def ordered_encode(texts, **kwargs):
        return np.array([[float(hash(t) % 1000)] * 384 for t in texts], dtype="float32")

    mock_model.encode.side_effect = ordered_encode

    chunks = [f"text_{i}" for i in range(50)]
    result = embed_chunks(chunks, batch_size=10)

    # Verify the first embedding matches the first chunk
    expected_first = float(hash("text_0") % 1000)
    assert result[0][0] == expected_first

    # Verify the last embedding matches the last chunk
    expected_last = float(hash("text_49") % 1000)
    assert result[49][0] == expected_last


def test_embed_documents_mini_batching_integration(mock_model):
    """Verify embed_documents correctly delegates to embed_chunks mini-batching."""
    docs = {
        "doc1": [f"chunk_{i}" for i in range(20)],
        "doc2": [f"chunk_{i}" for i in range(30)],
    }
    # Total 50 chunks. With batch_size=15, should be 4 calls (15+15+15+5)
    embed_documents(docs, batch_size=15)
    assert mock_model.encode.call_count == 4


def test_embed_chunks_garbage_collection_trigger(mock_model):
    """Verify gc.collect() is called periodically during large batch processing."""
    chunks = [f"chunk_{i}" for i in range(500)]

    with patch("src.core.embedding_model.gc.collect") as mock_gc:
        embed_chunks(chunks, batch_size=10)
        # gc.collect is called every 10 batches. 500/10 = 50 batches.
        # Calls at batch 10, 20, 30, 40 (i=100, 200, 300, 400)
        assert mock_gc.call_count >= 4


def test_embed_documents_empty_documents_dict(mock_model):
    """An empty dict of documents must return an empty dict without calling encode."""
    result = embed_documents({})
    assert result == {}
    mock_model.encode.assert_not_called()


def test_embed_documents_all_empty_docs(mock_model):
    """If all documents have 0 chunks, encode must not be called."""
    docs = {"doc1": [], "doc2": []}
    result = embed_documents(docs)
    assert result["doc1"].shape == (0, 384)
    assert result["doc2"].shape == (0, 384)
    mock_model.encode.assert_not_called()


def test_embed_chunks_large_batch_memory_mock(mock_model):
    """Simulate a large batch to ensure vstack handles accumulation correctly."""
    chunks = [f"chunk_{i}" for i in range(1000)]

    # Mock encode to return a specific shape
    mock_model.encode.side_effect = lambda texts, **kw: np.ones(
        (len(texts), 384), dtype="float32"
    )

    result = embed_chunks(chunks, batch_size=100)
    assert result.shape == (1000, 384)
    assert mock_model.encode.call_count == 10
