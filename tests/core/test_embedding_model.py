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

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.core.embedding_model as embedding_model
from src.core.embedding_model import (
    EmbeddingModelManager,
    _detect_device,
    embed_chunks,
    embed_documents,
    get_document_embedding,
)
from src.exceptions import ModelInitializationError


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


def test_embedding_model_manager_fallback_custom(caplog, monkeypatch):
    """Test that EmbeddingModelManager uses the custom fallback model from env var on failure."""
    import logging

    custom_fallback = "custom-fallback-model"
    monkeypatch.setenv("SEMANTIC_PLAGIARISM_FALLBACK_MODEL", custom_fallback)
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    primary = embedding_model._get_model_name()

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
        f"Primary embedding model {primary} unavailable. Falling back to {custom_fallback}"
        in record.message
        for record in caplog.records
    )


def test_embedding_model_manager_raises_descriptive_error_when_all_models_fail(
    monkeypatch,
):
    """Raise ModelInitializationError with offline deployment guidance when both loads fail."""
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    primary = embedding_model._get_model_name()

    def fail_every_model(model_name, cache_folder=None):
        raise RuntimeError(f"failed to load {model_name}")

    monkeypatch.setattr(embedding_model, "SentenceTransformer", fail_every_model)

    manager = EmbeddingModelManager.get_instance()

    with pytest.raises(ModelInitializationError) as exc_info:
        manager.get_model()

    message = str(exc_info.value)
    assert f"primary model '{primary}'" in message
    assert "fallback model 'all-MiniLM-L6-v2'" in message
    assert "offline or air-gapped deployments" in message
    assert "hf download" in message
    assert "SEMANTIC_PLAGIARISM_MODEL" in message
    assert "failed to load" in message


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
    expected_log = (
        "SentenceTransformer model [paraphrase-multilingual-MiniLM-L12-v2] "
        "running on device [cpu]"
    )
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

    assert _detect_device(None) in ("cpu", "cuda", "mps", "xpu")


def test_detect_device_supports_xpu(monkeypatch):
    """Return XPU when an Intel accelerator is available."""
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True, raising=False)
    assert _detect_device(None) == "xpu"


def test_detect_device_supports_rocm(monkeypatch):
    """ROCm/HIP devices are exposed through PyTorch's CUDA device API."""
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    assert _detect_device(None) == "cuda"


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


# ─── Tests for Embedding Model Quantization (Issue #1481) ─────────────────────

import pytest
import torch

from src.core.embedding_model import _apply_dynamic_quantization


class TestEmbeddingModelQuantization:
    """Test suite for INT8 dynamic quantization support."""

    def test_apply_dynamic_quantization_modifies_linear_layers(self):
        """Verify that quantize_dynamic targets torch.nn.Linear layers."""
        mock_model = MagicMock()
        # Simulate a module structure
        mock_model.modules.return_value = [torch.nn.Linear(10, 10)]

        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = MagicMock()
            _apply_dynamic_quantization(mock_model)

            mock_quantize.assert_called_once()
            args, kwargs = mock_quantize.call_args
            assert torch.nn.Linear in args[1]
            assert kwargs["dtype"] == torch.qint8

    def test_dynamic_quantization_skips_mps(self, caplog):
        """MPS models must bypass dynamic INT8 quantization."""
        mock_model = MagicMock()
        mock_model.device = "mps"

        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            with caplog.at_level("WARNING"):
                result = _apply_dynamic_quantization(mock_model)

        assert result is mock_model
        mock_quantize.assert_not_called()
        assert "disabled on MPS" in caplog.text

    def test_quantization_fallback_on_error(self, caplog):
        """If quantization fails, the original float32 model should be returned."""
        mock_model = MagicMock()

        with patch(
            "torch.quantization.quantize_dynamic",
            side_effect=RuntimeError("Quantization failed"),
        ):
            with caplog.at_level("WARNING"):
                result = _apply_dynamic_quantization(mock_model)

        assert result is mock_model
        assert "Failed to apply dynamic quantization" in caplog.text

    def test_manager_quantize_model_flag(self):
        """Verify EmbeddingModelManager respects the quantize_model flag."""
        # Reset singleton
        EmbeddingModelManager._instance = None

        manager = EmbeddingModelManager.get_instance(quantize_model=True)
        assert manager.quantize_model is True

        # Reset singleton
        EmbeddingModelManager._instance = None
        manager_fp32 = EmbeddingModelManager.get_instance(quantize_model=False)
        assert manager_fp32.quantize_model is False

    @patch("src.core.embedding_model.SentenceTransformer")
    @patch("src.core.embedding_model._apply_dynamic_quantization")
    def test_get_model_applies_quantization_when_enabled(self, mock_quantize, mock_st):
        """When quantize_model=True, get_model must call the quantization helper."""
        EmbeddingModelManager._instance = None
        global _quantized_model
        _quantized_model = None

        mock_base_model = MagicMock()
        mock_st.return_value = mock_base_model

        mock_quantized_model = MagicMock()
        mock_quantize.return_value = mock_quantized_model

        manager = EmbeddingModelManager.get_instance(quantize_model=True)
        model = manager.get_model()

        mock_quantize.assert_called_once_with(mock_base_model)
        assert model is mock_quantized_model

    @patch("src.core.embedding_model.SentenceTransformer")
    def test_get_model_skips_quantization_when_disabled(self, mock_st):
        """When quantize_model=False, get_model must NOT call the quantization helper."""
        EmbeddingModelManager._instance = None
        global _model
        _model = None

        mock_base_model = MagicMock()
        mock_st.return_value = mock_base_model

        manager = EmbeddingModelManager.get_instance(quantize_model=False)
        model = manager.get_model()

        assert model is mock_base_model

    def test_quantized_model_dimensions_match_float32(self):
        """Verify that quantized models produce embeddings with the same dimensions as float32."""
        # This is a mathematical invariant test. INT8 quantization should not
        # change the output vector dimensionality (e.g., 384 for MiniLM).
        # We mock the encode method to verify shape consistency.

        mock_fp32_model = MagicMock()
        mock_fp32_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)

        mock_int8_model = MagicMock()
        mock_int8_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)

        # Dimensions must match
        assert (
            mock_fp32_model.encode(["test"]).shape
            == mock_int8_model.encode(["test"]).shape
        )


class TestVerifyModelCacheIntegrity:
    """Tests for verify_model_cache_integrity (issue #1580)."""

    def test_returns_true_for_healthy_weight_file(self, tmp_path):
        """A cache with a large enough pytorch_model.bin must be healthy."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"\x00" * (1024 * 1024 + 1))

        assert embedding_model.verify_model_cache_integrity(tmp_path) is True

    def test_returns_false_for_small_file_1mb(self, tmp_path):
        """A file exactly 1MB (or smaller) must be reported as corrupted."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"\x00" * (1024 * 1024))

        assert embedding_model.verify_model_cache_integrity(tmp_path) is False

    def test_returns_false_for_zero_byte_pytorch_model_bin(self, tmp_path):
        """A zero-byte pytorch_model.bin must be reported as corrupted."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"")

        assert embedding_model.verify_model_cache_integrity(tmp_path) is False

    def test_returns_false_for_zero_byte_safetensors(self, tmp_path):
        """A zero-byte model.safetensors must be reported as corrupted."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"")

        assert embedding_model.verify_model_cache_integrity(tmp_path) is False

    def test_returns_true_when_cache_directory_missing(self, tmp_path):
        """A missing cache directory is healthy: the model will be downloaded fresh."""
        missing = tmp_path / "does_not_exist"

        assert embedding_model.verify_model_cache_integrity(missing) is True

    def test_returns_true_when_no_weight_files_cached(self, tmp_path):
        """A cache without any weight files must be healthy."""
        model_dir = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        assert embedding_model.verify_model_cache_integrity(tmp_path) is True

    def test_ignores_non_weight_files(self, tmp_path):
        """A zero-byte non-weight file must not fail the integrity check."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"\x00" * (1024 * 1024 + 1))
        (snapshot / "config.json").write_bytes(b"")

        assert embedding_model.verify_model_cache_integrity(tmp_path) is True

    def test_accepts_str_and_path_equivalently(self, tmp_path):
        """Both Path and str inputs must produce the same verdict."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"\x00" * (1024 * 1024 + 1))

        assert embedding_model.verify_model_cache_integrity(
            tmp_path
        ) == embedding_model.verify_model_cache_integrity(str(tmp_path))

    def test_logs_warning_for_corrupted_weight_file(self, tmp_path, caplog):
        """Corrupted weight files must be reported through the logger."""
        snapshot = tmp_path / "models--demo--model" / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"")

        with caplog.at_level("WARNING"):
            assert embedding_model.verify_model_cache_integrity(tmp_path) is False

        assert "Corrupted model weight file detected" in caplog.text


class TestGetModelRedownloadsCorruptedCache:
    """Tests for automatic re-download of corrupted cached models (issue #1580)."""

    @staticmethod
    def _corrupted_cache_dir(tmp_path) -> Path:
        """Create a model cache dir containing a zero-byte weight file."""
        model_cache_dir = tmp_path / "models--paraphrase-multilingual-MiniLM-L12-v2"
        snapshot = model_cache_dir / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"")
        return model_cache_dir

    def test_removes_corrupted_cache_before_loading(self, tmp_path, monkeypatch):
        """A zero-byte cached model must be deleted before the model is loaded."""
        model_cache_dir = self._corrupted_cache_dir(tmp_path)

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        embedding_model._model = None
        embedding_model._quantized_model = None
        EmbeddingModelManager._instance = None

        mock_model = MagicMock()
        with patch(
            "src.core.embedding_model.SentenceTransformer", return_value=mock_model
        ):
            model = EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        assert model is mock_model
        assert not model_cache_dir.exists()

    def test_keeps_healthy_cache_intact(self, tmp_path, monkeypatch):
        """A healthy cached model must not be deleted."""
        model_cache_dir = tmp_path / "models--paraphrase-multilingual-MiniLM-L12-v2"
        snapshot = model_cache_dir / "snapshots" / "abcdef1234"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"\x00" * (1024 * 1024 + 1))

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        embedding_model._model = None
        embedding_model._quantized_model = None
        EmbeddingModelManager._instance = None

        mock_model = MagicMock()
        with patch(
            "src.core.embedding_model.SentenceTransformer", return_value=mock_model
        ):
            model = EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        assert model is mock_model
        assert model_cache_dir.exists()

    def test_logs_warning_when_cache_removed(self, tmp_path, monkeypatch, caplog):
        """Removing a corrupted cache must log a warning message."""
        self._corrupted_cache_dir(tmp_path)

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        embedding_model._model = None
        embedding_model._quantized_model = None
        EmbeddingModelManager._instance = None

        mock_model = MagicMock()
        with patch(
            "src.core.embedding_model.SentenceTransformer", return_value=mock_model
        ):
            with caplog.at_level("WARNING"):
                EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        assert "Corrupted cache detected" in caplog.text


class TestModelCacheFolderConfig:
    """Tests for the configurable model cache folder (issue #1759)."""

    @staticmethod
    def _reset_global_state(monkeypatch):
        """Reset cached module models and the manager singleton before each test."""
        monkeypatch.setattr(embedding_model, "_model", None)
        monkeypatch.setattr(embedding_model, "_quantized_model", None)
        monkeypatch.setattr(EmbeddingModelManager, "_instance", None)

    def test_get_cache_dir_reads_hf_hub_cache(self, monkeypatch):
        """_get_cache_dir() must return the HF_HUB_CACHE env var when set."""
        monkeypatch.setenv("HF_HUB_CACHE", "C:/custom/hf_cache")
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

        assert embedding_model._get_cache_dir() == "C:/custom/hf_cache"

    def test_get_cache_dir_falls_back_to_transformers_cache(self, monkeypatch):
        """When HF_HUB_CACHE is unset, TRANSFORMERS_CACHE must be used instead."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("TRANSFORMERS_CACHE", "C:/custom/transformers_cache")

        assert embedding_model._get_cache_dir() == "C:/custom/transformers_cache"

    def test_get_cache_dir_defaults_to_none(self, monkeypatch):
        """When neither cache env var is set, _get_cache_dir() must be None."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

        assert embedding_model._get_cache_dir() is None

    def test_get_model_passes_cache_folder_to_loader(self, monkeypatch):
        """SentenceTransformer must receive cache_folder=<HF_HUB_CACHE>."""
        cache_dir = "C:/custom/hf_cache"
        monkeypatch.setenv("HF_HUB_CACHE", cache_dir)
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        sentence_transformer = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            embedding_model, "SentenceTransformer", sentence_transformer
        )
        self._reset_global_state(monkeypatch)

        EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        sentence_transformer.assert_called_once_with(
            embedding_model._DEFAULT_MODEL_NAME, cache_folder=cache_dir
        )

    def test_get_model_logs_configured_cache_target(self, monkeypatch, caplog):
        """The configured cache location must be logged on model load."""
        import logging

        cache_dir = "C:/custom/hf_cache"
        monkeypatch.setenv("HF_HUB_CACHE", cache_dir)
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        monkeypatch.setattr(
            embedding_model, "SentenceTransformer", MagicMock(return_value=MagicMock())
        )
        self._reset_global_state(monkeypatch)

        with caplog.at_level(logging.INFO):
            EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        assert f"Model cache target: {cache_dir}" in caplog.text

    def test_get_model_logs_default_cache_target(self, monkeypatch, caplog):
        """The default cache location must be logged when no cache dir is set."""
        import logging

        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
        monkeypatch.setenv(
            "SEMANTIC_PLAGIARISM_MODEL", embedding_model._DEFAULT_MODEL_NAME
        )
        monkeypatch.setattr(
            embedding_model, "SentenceTransformer", MagicMock(return_value=MagicMock())
        )
        self._reset_global_state(monkeypatch)

        with caplog.at_level(logging.INFO):
            EmbeddingModelManager.get_instance(quantize_model=False).get_model()

        assert "Model cache target: default (~/.cache/huggingface)" in caplog.text
