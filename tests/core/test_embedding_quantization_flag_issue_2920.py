"""Unit tests for EmbeddingModelManager dynamic quantization flag (Issue #2920)."""

from unittest.mock import MagicMock, patch
import pytest

from src.core.embedding_model import EmbeddingModelManager
import src.core.embedding_model as emb_module


@pytest.fixture(autouse=True)
def reset_singleton_state():
    """Ensure singleton and cached models are reset before and after each test."""
    EmbeddingModelManager._instance = None
    emb_module._model = None
    emb_module._quantized_model = None
    yield
    EmbeddingModelManager._instance = None
    emb_module._model = None
    emb_module._quantized_model = None


@patch("src.core.embedding_model.SentenceTransformer")
@patch("src.core.embedding_model._apply_dynamic_quantization")
def test_embedding_model_manager_invokes_dynamic_quantization_when_quantize_true(
    mock_apply_quant, mock_st
):
    """Test that requesting EmbeddingModelManager with quantize_model=True invokes _apply_dynamic_quantization."""
    mock_base_model = MagicMock(name="BaseModel")
    mock_quantized_model = MagicMock(name="QuantizedModel")
    mock_st.return_value = mock_base_model
    mock_apply_quant.return_value = mock_quantized_model

    manager = EmbeddingModelManager.get_instance(quantize_model=True)
    assert manager.quantize_model is True

    model = manager.get_model()

    # Assert _apply_dynamic_quantization was called with the base model
    mock_apply_quant.assert_called_once_with(mock_base_model)
    assert model is mock_quantized_model


@patch("src.core.embedding_model.SentenceTransformer")
@patch("src.core.embedding_model._apply_dynamic_quantization")
def test_embedding_model_manager_skips_dynamic_quantization_when_quantize_false(
    mock_apply_quant, mock_st
):
    """Test that requesting EmbeddingModelManager with quantize_model=False does not invoke _apply_dynamic_quantization."""
    mock_base_model = MagicMock(name="BaseModel")
    mock_st.return_value = mock_base_model

    manager = EmbeddingModelManager.get_instance(quantize_model=False)
    assert manager.quantize_model is False

    model = manager.get_model()

    # Assert _apply_dynamic_quantization was NOT called
    mock_apply_quant.assert_not_called()
    assert model is mock_base_model


@patch("src.core.embedding_model.SentenceTransformer")
@patch("src.core.embedding_model._apply_dynamic_quantization")
def test_embedding_model_manager_singleton_transitions_to_quantized(
    mock_apply_quant, mock_st
):
    """Test that calling get_instance(quantize_model=True) on an existing unquantized singleton upgrades it."""
    mock_base_model = MagicMock(name="BaseModel")
    mock_quantized_model = MagicMock(name="QuantizedModel")
    mock_st.return_value = mock_base_model
    mock_apply_quant.return_value = mock_quantized_model

    # 1. First get unquantized instance
    manager1 = EmbeddingModelManager.get_instance(quantize_model=False)
    assert manager1.quantize_model is False

    # 2. Upgrade singleton with quantize_model=True
    manager2 = EmbeddingModelManager.get_instance(quantize_model=True)
    assert manager2 is manager1
    assert manager2.quantize_model is True

    model = manager2.get_model()
    mock_apply_quant.assert_called_once_with(mock_base_model)
    assert model is mock_quantized_model
